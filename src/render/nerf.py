"""
NeRF differentiable renderer.
References:
https://github.com/bmild/nerf
https://github.com/kwea123/nerf_pl
"""
import torch
import torch.nn.functional as F
import util
import torch.autograd.profiler as profiler
from torch.nn import DataParallel
from dotmap import DotMap
from model import NeuralRenderer
import math 

class _RenderWrapper(torch.nn.Module):
    def __init__(self, net, renderer, simple_output):
        super().__init__()
        self.net = net  # pixelNeRFNet -> outputs: Batch, Ray*Points, 4 (rgba)
        self.renderer = renderer    # self.renderer 
        self.simple_output = simple_output

    def forward(self, rays, val_num=1, want_weights=False, training=False):
        if rays.shape[0] == 0:
            return (
                torch.zeros(0, 3, device=rays.device),
                torch.zeros(0, device=rays.device),
            )
        # coarse, fine 둘다 sampling.. 우선은 coarse로 하고 필요하면 fine으로 바꾸기!
        ###### 여기에서 밑의 함수로 흘러들어간다!
        ###### self.net = NeRFRenderer
        outputs = self.renderer(
            self.net, rays, training, val_num, want_weights=want_weights and not self.simple_output
        )
        featmap = outputs.feat
        return featmap


class NeRFRenderer(torch.nn.Module):
    """
    NeRF differentiable renderer
    :param n_coarse number of coarse (binned uniform) samples
    :param n_fine number of fine (importance) samples
    :param n_fine_depth number of expected depth samples
    :param noise_std noise to add to sigma. We do not use it
    :param depth_std noise for depth samples
    :param eval_batch_size ray batch size for evaluation
    :param white_bkgd if true, background color is white; else black
    :param lindisp if to use samples linear in disparity instead of distance
    :param sched ray sampling schedule. list containing 3 lists of equal length.
    sched[0] is list of iteration numbers,
    sched[1] is list of coarse sample numbers,
    sched[2] is list of fine sample numbers
    """

    def __init__(
        self,
        n_coarse=128,
        n_fine=0,
        n_fine_depth=0,
        noise_std=0.0,
        depth_std=0.01,
        eval_batch_size=100000,
        white_bkgd=False,
        lindisp=False,
        sched=None,  # ray sampling schedule for coarse and fine rays
    ):
        super().__init__()
        self.n_coarse = n_coarse
        self.n_fine = n_fine
        self.n_fine_depth = n_fine_depth

        self.noise_std = noise_std
        self.depth_std = depth_std

        self.white_bkgd = white_bkgd
        self.lindisp = lindisp
        if lindisp:
            print("Using linear displacement rays")
        self.using_fine = n_fine > 0
        self.sched = sched
        if sched is not None and len(sched) == 0:
            self.sched = None
        self.register_buffer(
            "iter_idx", torch.tensor(0, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "last_sched", torch.tensor(0, dtype=torch.long), persistent=True
        )


    def sample_coarse(self, rays):
        """
        Stratified sampling. Note this is different from original NeRF slightly.
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :return (B, Kc)
        """
        device = rays.device
        near, far = rays[:, -2:-1], rays[:, -1:]  # (B, 1)

        step = 1.0 / self.n_coarse
        B = rays.shape[0]
        z_steps = torch.linspace(0, 1 - step, self.n_coarse, device=device)  # (Kc)
        z_steps = z_steps.unsqueeze(0).repeat(B, 1)  # (B, Kc)
        z_steps += torch.rand_like(z_steps) * step
        if not self.lindisp:  # Use linear sampling in depth space
            return near * (1 - z_steps) + far * z_steps  # (B, Kf)
        else:  # Use linear sampling in disparity space
            return 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)  # (B, Kf)

        # Use linear sampling in depth space
        return near * (1 - z_steps) + far * z_steps  # (B, Kc)

    def sample_fine(self, rays, weights):
        """
        Weighted stratified (importance) sample
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param weights (B, Kc)
        :return (B, Kf-Kfd)
        """
        device = rays.device
        B = rays.shape[0]

        weights = weights.detach() + 1e-5  # Prevent division by zero
        pdf = weights / torch.sum(weights, -1, keepdim=True)  # (B, Kc)
        cdf = torch.cumsum(pdf, -1)  # (B, Kc)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (B, Kc+1)

        u = torch.rand(
            B, self.n_fine - self.n_fine_depth, dtype=torch.float32, device=device
        )  # (B, Kf)
        inds = torch.searchsorted(cdf, u, right=True).float() - 1.0  # (B, Kf)
        inds = torch.clamp_min(inds, 0.0)

        z_steps = (inds + torch.rand_like(inds)) / self.n_coarse  # (B, Kf)

        near, far = rays[:, -2:-1], rays[:, -1:]  # (B, 1)
        if not self.lindisp:  # Use linear sampling in depth space
            z_samp = near * (1 - z_steps) + far * z_steps  # (B, Kf)
        else:  # Use linear sampling in disparity space
            z_samp = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)  # (B, Kf)
        return z_samp

    def sample_fine_depth(self, rays, depth):
        """
        Sample around specified depth
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param depth (B)
        :return (B, Kfd)
        """
        z_samp = depth.unsqueeze(1).repeat((1, self.n_fine_depth))
        z_samp += torch.randn_like(z_samp) * self.depth_std
        # Clamp does not support tensor bounds
        z_samp = torch.max(torch.min(z_samp, rays[:, -1:]), rays[:, -2:-1])
        return z_samp

    def composite(self, model, rays, z_samp, training, coarse=True, sb=0):        # 여기서 가져와지는 애들 찾기!
        """     
        Render RGB and depth for each ray using NeRF alpha-compositing formula,
        given sampled positions along each ray (see sample_*)
        :param model should return (B, (r, g, b, sigma)) when called with (B, (x, y, z))
        should also support 'coarse' boolean argument
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param z_samp z positions sampled for each ray (B, K)
        :param coarse whether to evaluate using coarse NeRF
        :param sb super-batch dimension; 0 = disable
        :return weights (B, K), rgb (B, 3), depth (B)
        """
        # nerf.py가 나름 바깥에서 batch 연산을 적용중!
        with profiler.record_function("renderer_composite"):
            B, K = z_samp.shape # B: 512: batcy*ray, K: 64: number of sampled points 
            # z_coarse = (16*16, 64) 

            ###############################################################
            ##################### ray와 sampling의 차이!! ####################
            deltas = z_samp[:, 1:] - z_samp[:, :-1]  # (B, K-1)
            #  if far:
            #      delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # infty (B, 1)
            delta_inf = rays[:, -1:] - z_samp[:, -1:]
            deltas = torch.cat([deltas, delta_inf], -1)  # (B, K)

            #############################################################################
            ##################### ray로 sampled points를 통해 points 만듦!! ################
            # (B, K, 3)
                                            # (512, 64, 1)        (512, 1, 3)
            points = rays[:, None, :3] + z_samp.unsqueeze(2) * rays[:, None, 3:6]    # points: (32768, 3) -> (batch*rays*points, 3)
            points = points.reshape(sb, -1, 3)  # (B*K, 3)
            # generated points along ray
            ################# 두 가지 실험..? 그러면 render할 때도 동일 ray 아니면 고정 ray..? ####
            #############################################################################

            use_viewdirs = hasattr(model, "use_viewdirs") and model.use_viewdirs
            val_all = []
            # points: (4, 8192, 3)
            # eval_batch_size = 25000
            # 아 그냥 eval batch 구분하지 말고 바로 넣기 (어차피 구분 안돼서 들어가고 있었음)
            dim1 = K    # 64: # sampling points for each ray 
            viewdirs = rays[:, None, 3:6].expand(-1, dim1, -1)  # (B, K, 3)     # (512 batch * rays, 64 #sampling points , 3)
            viewdirs = viewdirs.reshape(sb, -1, 3)  # (SB, B'*K, 3)  # (batch, rays*sampling points, 3)
            out = model(points, num_pts=K, coarse=coarse, viewdirs=viewdirs, training=self.training) # pnts, dirs 둘 다 :(4, 8192, 3) <- 뭐야 그냥 그대로 들어가도 되는거였음 
                # 여기서는 model: PixelNeRFNet의 forward함수로 바로 ㄱㄱ!

            points = None
            viewdirs = None

            # 오케... 여기까지가 sampling points 다 살아있는 상태에서 rgba 계산된 결과!!
            # (B*K, 4) OR (SB, B'*K, 4)

            out = out.reshape(B, K, -1)  # (B, K, 4 or 5)   (512, 64, 4) <- (batch*#rays, #points, rgba)

            feats = out[..., :-1]  # (B, K, 3)
            sigmas = out[..., -1]  # (B, K)
            if self.training and self.noise_std > 0.0:
                sigmas = sigmas + torch.randn_like(sigmas) * self.noise_std

            alphas = 1 - torch.exp(-deltas * torch.relu(sigmas))  # (B, K)
            deltas = None
            sigmas = None
            alphas_shifted = torch.cat(
                [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1
            )  # (B, K+1) = [1, a1, a2, ...]
            T = torch.cumprod(alphas_shifted, -1)  # (B)
            weights = alphas * T[:, :-1]  # (B, K)
            alphas = None
            alphas_shifted = None

            feat_final = torch.sum(weights.unsqueeze(-1) * feats, -2)  # (B, 3)
            # compositing 성공!

            # 여기에 neural renderer 넣기 -> net에 최종으로 잘 들어가는지 확인!


            # for white background -> 일단은 빼자!
            # pix_alpha = weights.sum(dim=1)  # (B), pixel alpha
            # rgb_final = rgb_final + 1 - pix_alpha.unsqueeze(-1)  # (B, 3)

            return feat_final

    def forward(
        self, model, rays, training, val_num, want_weights=False,
    ):
        """
        :model nerf model, should return (SB, B, (r, g, b, sigma))
        when called with (SB, B, (x, y, z)), for multi-object:
        SB = 'super-batch' = size of object batch,
        B  = size of per-object ray batch.
        Should also support 'coarse' boolean argument for coarse NeRF.
        :param rays ray spec [origins (3), directions (3), near (1), far (1)] (SB, B, 8)
        :param want_weights if true, returns compositing weights (SB, B, K)
        :return render dict
        """
        with profiler.record_function("renderer_forward"):
            if self.sched is not None and self.last_sched.item() > 0:
                self.n_coarse = self.sched[1][self.last_sched.item() - 1]
                self.n_fine = self.sched[2][self.last_sched.item() - 1]

            assert len(rays.shape) == 3
            superbatch_size = rays.shape[0] # (1, 16*16, 8)
            rays = rays.reshape(-1, 8)  # (SB * B, 8) -> (16*16, 8)
            ray_res = int(math.sqrt(rays.shape[0]/ val_num / superbatch_size))  # -> 16

            z_coarse = self.sample_coarse(rays)  # (B, Kc)  # coarse        # sampled points  -> (16*16, 8) -> 64도 same
            # -> z_coarse = (1024 (4 * 256), 64), rays = (1024 (4 * 256), 8)
            
            out_composite = self.composite(               # given models, rays, z_coars values, -> sampled points along ray! 
                model, rays, z_coarse, training, coarse=True, sb=superbatch_size,
            ).reshape(superbatch_size * val_num, ray_res*ray_res, -1)   # [1]: feat -> (batch*ray, feat_dim) -> 우리는 여기서 2DCNN을 가져와서 돌려야 함!   -> 각 ray가 rgb가 아닌 feature를 가지고 있기 때문!

            composite_permute = out_composite.permute(0, 2, 1).contiguous().reshape(superbatch_size*val_num, -1, ray_res, ray_res)
            coarse_composite = composite_permute.permute(0, 1, 3, 2).contiguous()

            # for visualization -> 정리하기!
            # rgb_np= np.array(rgb.detach().cpu())
            # out_file_name = 'visualization_%010d.png' % it
            # image_grid = make_grid(torch.cat((x_real, image_fake.clamp_(0., 1.), image_swap.clamp_(0., 1.), image_rand.clamp_(0., 1.)), dim=0), nrow=image_fake.shape[0])
            # save_image(image_grid, os.path.join(self.val_vis_dir, out_file_name))

            ################################################################
            ################## 여기까지 해서, 최종 ray 뽑자 ######################
            ################################################################
            ################################################################
            outputs = DotMap(
                feat=self._format_outputs(
                    coarse_composite, superbatch_size, want_weights=want_weights,
                ),  # 이거를 coarse.rgb로 호출할 수 있게 됨!
            )
            return outputs

    def _format_outputs(
        self, rendered_outputs, superbatch_size, want_weights=False,
    ):
        # rendered_outputs: (batch * rays, 3)
        # superbatch로 reshape -> (batch, rays, 3)
        feat_map = rendered_outputs  # 
        # if superbatch_size > 0:
        #     feat_map = feat.reshape(superbatch_size, -1, 128)  # batch, rays, #feat
        return feat_map

    def sched_step(self, steps=1):
        """
        Called each training iteration to update sample numbers
        according to schedule
        """
        if self.sched is None:
            return
        self.iter_idx += steps
        while (
            self.last_sched.item() < len(self.sched[0])
            and self.iter_idx.item() >= self.sched[0][self.last_sched.item()]
        ):
            self.n_coarse = self.sched[1][self.last_sched.item()]
            self.n_fine = self.sched[2][self.last_sched.item()]
            print(
                "INFO: NeRF sampling resolution changed on schedule ==> c",
                self.n_coarse,
                "f",
                self.n_fine,
            )
            self.last_sched += 1

    @classmethod
    def from_conf(cls, conf, white_bkgd=False, lindisp=False, eval_batch_size=100000):
        return cls(
            conf.get_int("n_coarse", 128),
            conf.get_int("n_fine", 0),
            n_fine_depth=conf.get_int("n_fine_depth", 0),
            noise_std=conf.get_float("noise_std", 0.0),
            depth_std=conf.get_float("depth_std", 0.01),
            white_bkgd=conf.get_float("white_bkgd", white_bkgd),
            lindisp=lindisp,
            eval_batch_size=conf.get_int("eval_batch_size", eval_batch_size),
            sched=conf.get_list("sched", None),
        )

    def bind_parallel(self, net, gpus=None, simple_output=False):
        """
        Returns a wrapper module compatible with DataParallel.
        Specifically, it renders rays with this renderer
        but always using the given network instance.
        Specify a list of GPU ids in 'gpus' to apply DataParallel automatically.
        :param net A PixelNeRF network
        :param gpus list of GPU ids to parallize to. If length is 1,
        does not parallelize
        :param simple_output only returns rendered (rgb, depth) instead of the 
        full render output map. Saves data tranfer cost.
        :return torch module
        """
        wrapped = _RenderWrapper(net, self, simple_output=simple_output)
        if gpus is not None and len(gpus) > 1:
            print("Using multi-GPU", gpus)
            wrapped = torch.nn.DataParallel(wrapped, gpus, dim=1)
        return wrapped
