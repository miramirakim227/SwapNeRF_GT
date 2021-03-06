"""
Main model implementation
"""
import torch
from .encoder import ImageEncoder
from .code import PositionalEncoding
from .model_util import make_encoder, make_mlp
import torch.autograd.profiler as profiler
from util import repeat_interleave
import os
import os.path as osp
import warnings
import torch.nn.functional as F
from .neural_renderer import NeuralRenderer


class PixelNeRFNet(torch.nn.Module):
    def __init__(self, conf, decoder, device=None, stop_encoder_grad=False):
        """
        :param conf PyHocon config subtree 'model'
        """
        super().__init__()
        self.encoder = make_encoder(conf["encoder"])    # encoder type  # resnet34, pretrainedTrue, num_layers4
            # encoder 설정 -> ours encoder! 
        self.use_encoder = conf.get_bool("use_encoder", True)  # Image features?        # True

        self.use_xyz = conf.get_bool("use_xyz", False)  # True로 설정되어 있음. use xyz instead of just z 
                                                                                        # True
        assert self.use_encoder or self.use_xyz  # Must use some feature..  둘다 True로 되어있음!

        # Whether to shift z to align in canonical frame.
        # So that all objects, regardless of camera distance to center, will
        # be centered at z=0.
        # Only makes sense in ShapeNet-type setting.
        self.normalize_z = conf.get_bool("normalize_z", True)                       # X -> True

        self.stop_encoder_grad = (
            stop_encoder_grad  # Stop ConvNet gradient (freeze weights)
        )
        self.use_code = conf.get_bool("use_code", False)  # Positional encoding     # True
        self.use_code_viewdirs = conf.get_bool(                                     # False
            "use_code_viewdirs", True
        )  # Positional encoding applies to viewdirs

        # Enable view directions
        self.use_viewdirs = conf.get_bool("use_viewdirs", False)                     # True

        # Global image features?
        self.use_global_encoder = conf.get_bool("use_global_encoder", False)         # False

        d_latent = self.encoder.latent_size if self.use_encoder else 0               
        d_in = 3 if self.use_xyz else 1                                              # 3

        if self.use_viewdirs and self.use_code_viewdirs:
            # Apply positional encoding to viewdirs
            d_in += 3       # True
        if self.use_code and d_in > 0:
            # Positional encoding for x,y,z OR view z
            self.code = PositionalEncoding.from_conf(conf["code"], d_in=d_in)       # True, conf['code']: code setting such as num_freqs, freq_fac, include_input(True)
            d_in = self.code.d_out              # True -> self.num_freqs * 2 * d_in
        if self.use_viewdirs and not self.use_code_viewdirs:
            # Don't apply positional encoding to viewdirs (concat after encoded)
            d_in += 3       # True

        if self.use_global_encoder:
            # Global image feature
            self.global_encoder = ImageEncoder.from_conf(conf["global_encoder"])
            self.global_latent_size = self.global_encoder.latent_size
            d_latent += self.global_latent_size

        d_out = 4

        self.latent_size = self.encoder.latent_size
        self.mlp_coarse = make_mlp(conf["mlp_coarse"], d_in, d_latent, d_out=d_out)     # mlpcoarse: resnet, n_blocks=3, d_hidden=512
        self.mlp_fine = make_mlp(
            conf["mlp_fine"], d_in, d_latent, d_out=d_out, allow_empty=True     # same as above
        )   # resnet 
        # Note: this is world -> camera, and bottom row is omitted
        self.register_buffer("poses", torch.empty(1, 3, 4), persistent=False)
        self.register_buffer("image_shape", torch.empty(2), persistent=False)

        self.d_in = d_in
        self.d_out = d_out
        self.d_latent = d_latent
        self.register_buffer("focal", torch.empty(1, 2), persistent=False)
        # Principal point
        self.register_buffer("c", torch.empty(1, 2), persistent=False)

        self.num_objs = 0
        self.num_views_per_obj = 1
        if device is None:
            device = self.poses.device
        self.decoder = decoder.to(device)

        self.neural_renderer = NeuralRenderer().to(device)

    #######################################################################################
    ################### 여기서부터 잘 집중해서 읽어보기! encode하는 부분에서도 잘 가져오기!! #############
    #####################################################################################

    def encode(self, images, poses, focal, z_bounds=None, c=None):
        """
        :param images (NS, 3, H, W)         (B, 3, H, W) <- Ns: num of views 대신 batch size로 규정!
        NS is number of input (aka source or reference) views
        :param poses (NS, 4, 4)
        :param focal focal length () or (2) or (NS) or (NS, 2) [fx, fy]
        :param z_bounds ignored argument (used in the past)
        :param c principal point None or () or (2) or (NS) or (NS, 2) [cx, cy],
        default is center of image
        """

        self.num_objs = images.size(0)      # images.shape -> since single view, : (4, 1, 3, 128, 128) <- batch ,single view image 

        if len(images.shape) == 5:          # Be consistent with NS = num input views
            assert len(poses.shape) == 4
            assert poses.size(1) == images.size(
                1
            )  # Be consistent with NS = num input views        
            self.num_views_per_obj = images.size(1)
            images = images.reshape(-1, *images.shape[2:])  # 아예 num_views의 흔적을 없앰, (-1(B, NV merged!), 3, 128, 128)
            poses = poses.reshape(-1, 4, 4)             
        else:
            self.num_views_per_obj = 1

        # image : (batch, 3, H, W)로 들어옴
        # pose = (batch, 4, 4)로 만들어줘야 함              # 여기서 예측된 self.poses가 GT poses를 대체함 
        self.shape, self.appearance = self.encoder(images)        # self.latents만 따로 활용하려고 사용함! -> self.latent가 따로 저장됨!, self.latent_scaling도 따로 저장됨
        # 위에까지가 feature 얻는 부분 

        # 여기서부터 가져올 부분 poses -> self.rotmat으로 대체 
        rot = poses[:, :3, :3].transpose(1, 2)  # (B, 3, 3)     # 역행렬 -> 원래의 rotation으로 바꿔준다 
        trans = -torch.bmm(rot, poses[:, :3, 3:])  # (B, 3, 1)      # 이 translation이 의외군.. 
        self.poses = torch.cat((rot, trans), dim=-1)  # (B, 3, 4)

        self.image_shape[0] = images.shape[-1]
        self.image_shape[1] = images.shape[-2]

        # input으로 주어지는 focal type에 따라서 변경 
        # Handle various focal length/principal point formats
        if len(focal.shape) == 0:
            # Scalar: fx = fy = value for all views
            focal = focal[None, None].repeat((1, 2))
        elif len(focal.shape) == 1:
            # Vector f: fx = fy = f_i *for view i*
            # Length should match NS (or 1 for broadcast)
            focal = focal.unsqueeze(-1).repeat((1, 2))
        else:
            focal = focal.clone()
        self.focal = focal.float()     # 가장 마지막의 값들에 -1이 곱해짐 
        self.focal[..., 1] *= -1.0

        if c is None:
            # Default principal point is center of image
            c = (self.image_shape * 0.5).unsqueeze(0)
        elif len(c.shape) == 0:
            # Scalar: cx = cy = value for all views
            c = c[None, None].repeat((1, 2))
        elif len(c.shape) == 1:
            # Vector c: cx = cy = c_i *for view i*
            c = c.unsqueeze(-1).repeat((1, 2))
        self.c = c

        if self.use_global_encoder:     # False
            self.global_encoder(images)


    #######################################################################################
    ################### 여기서부터 잘 집중해서 읽어보기! xyz 만드는 과정도 똑같이 할 것! ################
    #######################################################################################

    ### 여기서부터 잘 집중해서 읽어보기!!! + xyz는 어쨌든 sampling된 query points인데 어떻게 나오게 된 건지 파악하기!
    def forward(self, xyz, num_pts, coarse=True, viewdirs=None, training=False, far=False):  # world space points xyz
        # 어차피 여기 한번밖에 안지나감 괜쫄!
        # xyz: (batch, #rays * #points, 3)
        """
        Predict (r, g, b, sigma) at world space points xyz.
        Please call encode first!
        :param xyz (SB, B, 3)
        SB is batch of objects
        B is batch of points (in rays)
        NS is number of input views
        :return (SB, B, 4) r g b sigma
        """
        with profiler.record_function("model_inference"):       # memory, time tracking tool -> 한번 다 합치고 돌려보기!
            SB, B, _ = xyz.shape       # SB: batch of objects, B: num_rays * num_points -> batch of points in rays -> 리얼 한 세트로 돌리네! 굿! 배치마다의 샘플 속 모든 ray를 포괄!
            ##################################################################################
            # Transform query points into the camera spaces of the input views
            xyz_rot = torch.matmul(self.poses[:, None, :3, :3], xyz.unsqueeze(-1))[     # xyz를 self.poses로 rotate -> transpose했던거에 다시 연산됨..! -> 헐 그러면 이 상태에서 예측하는건가보다 그러면 이게 sampled points고 앞에 camera가 원래 ray에서...!
                                                                                        # 그래야 transformation이 말이 되는 듯.. 근데 그럴거면 왜 transformation을 예측하지..? 
                ..., 0                                                                  # 아무튼 여기가 transform query points into the camera spaces! (self.poses를 곱함!)
            ]
            # 오키.. def encoder에서 생긴 얘가 여기로 들어감!
            xyz = xyz_rot + self.poses[:, None, :3, 3]      # 얘네가 sampling points!     # 아무튼 여기가 transform query points into the camera spaces! (self.poses를 곱함!) 
            # Transform query points into the camera spaces of the input views
            ##################################################################################
            # * Encode the xyz coordinates
            z_feature = xyz_rot.reshape(SB, -1, 3)  # (SB*B, 3) # (65536, 3) -> (SB * num_ray * num_points, 3)  -> (4 * 256 * 64, 3)

            # Viewdirs to input view space
            viewdirs = viewdirs.reshape(SB, B, 3, 1)
            viewdirs = torch.matmul(
                self.poses[:, None, :3, :3], viewdirs   # pose에 viewdir 곱함 <- 위에와 마찬가지로 곱해줌!!
            )  # (SB*NS, B, 3, 1)
            viewdirs = viewdirs.reshape(SB, -1, 3)  # (SB*B, 3)
            
            # Camera frustum culling stuff, currently disabled
            combine_index = None
            dim_size = None

            #######################################################################################
            ################### 여기 앞에까지만! mlp_input을 만드는 과정을 똑같이 할 것! #####################
            #######################################################################################

            # 여기서가 여태까지 샘플링했던 ray, points 전부 해결해주는 곳. 
            # 얘도 뭔가 그냥 아예 batch 단위로 한번에 들어가버림 
            # Run main NeRF network
            # images       dim 맞음! # feature, viewdirs, shape, appearance dimension 맞춰주기 
            feat, sigma = self.decoder(z_feature, viewdirs, self.shape, self.appearance)        # z_feature, viewdirs, : (batch, -1, 3), shape&app : (batch, 256)

            if training:
                sigma += torch.randn_like(sigma)
            sigma_i = sigma.reshape(SB, B // num_pts, num_pts)          # 4, 256, 64
            feat_i = feat.reshape(SB, B // num_pts, num_pts, -1)        # 4, 256, 64, 128

            feat = feat_i
            sigma = F.relu(sigma_i).unsqueeze(-1)
            
            output_list = [feat, sigma]
            output = torch.cat(output_list, dim=-1)
            output = output.reshape(SB, B, -1)   # (4, ray(128)*sampling points(64) -> (8192), feat_dim+1)
        return output
        # 이 윗부분만 잘 집중해서 읽어보기!
        # 아 오키. 여기서나온 각 points의 예측된 rgb와 sigma를 가지고 그다음 ray를 한다!




    def load_weights(self, args, opt_init=False, strict=True, device=None):
        """
        Helper for loading weights according to argparse arguments.
        Your can put a checkpoint at checkpoints/<exp>/pixel_nerf_init to use as initialization.
        :param opt_init if true, loads from init checkpoint instead of usual even when resuming
        """
        # TODO: make backups -> giraffe 코드 보고 백업되게 바꾸기 
        if opt_init and not args.resume:
            return
        ckpt_name = (
            "pixel_nerf_init" if opt_init or not args.resume else "pixel_nerf_latest"
        )
        model_path = "%s/%s/%s" % (args.checkpoints_path, args.name, ckpt_name)

        if device is None:
            device = self.poses.device

        if os.path.exists(model_path):
            print("Load", model_path)
            self.load_state_dict(
                torch.load(model_path, map_location=device), strict=False
            )
        elif not opt_init:
            warnings.warn(
                (
                    "WARNING: {} does not exist, not loaded!! Model will be re-initialized.\n"
                    + "If you are trying to load a pretrained model, STOP since it's not in the right place. "
                    + "If training, unless you are startin a new experiment, please remember to pass --resume."
                ).format(model_path)
            )
        return self

    def save_weights(self, args, opt_init=False):
        """
        Helper for saving weights according to argparse arguments
        :param opt_init if true, saves from init checkpoint instead of usual
        """
        from shutil import copyfile

        ckpt_name = "pixel_nerf_init" if opt_init else "pixel_nerf_latest"
        backup_name = "pixel_nerf_init_backup" if opt_init else "pixel_nerf_backup"

        ckpt_path = osp.join(args.checkpoints_path, args.name, ckpt_name)
        ckpt_backup_path = osp.join(args.checkpoints_path, args.name, backup_name)

        if osp.exists(ckpt_path):
            copyfile(ckpt_path, ckpt_backup_path)
        torch.save(self.state_dict(), ckpt_path)
        return self
