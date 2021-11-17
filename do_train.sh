# CUDA_VISIBLE_DEVICES=1 python train/train.py -n srn_car_exp -c conf/exp/srn.conf -D /root/dataset2/ShapeNet/cars --lr 1e-4 --name check_val_num

# CUDA_VISIBLE_DEVICES=0 python train/train.py -c conf/exp/srn.conf -D /root/dataset2/ShapeNet/cars --lr 5e-5 --name GTCAM_5e-5

# CUDA_VISIBLE_DEVICES=3 python train/train.py -c conf/exp/srn.conf -D /root/dataset2/ShapeNet/cars --lr 5e-5 --name new

# CUDA_VISIBLE_DEVICES=1 python train/train.py -c conf/exp/srn.conf -D /root/dataset2/ShapeNet/cars --lr 1e-4 --recon 50 --name delete --cam 10

# generator = 1e-4

# CUDA_VISIBLE_DEVICES=2 python train/train.py -c conf/exp/srn.conf -D /root/dataset2/ShapeNet/cars --lr 1e-4 --recon 50 --name swapGT --disc_lr 0.01

# CUDA_VISIBLE_DEVICES=0 python train/train.py -c conf/exp/srn.conf -D /root/project/ShapeNet/cars --lr 1e-4 --recon 50 --name swap_D01 --batch_size 32 --disc_lr 0.1

# vis 
CUDA_VISIBLE_DEVICES=0 python train/train.py -c conf/exp/srn.conf -D /root/project/ShapeNet/cars --lr 1e-4 --recon 50 --name epoch3-G-1e-4-D-1e-2 --disc_lr 0.01 --batch_size 32 --epoch-period 3

# epoch3
CUDA_VISIBLE_DEVICES=0 python train/train.py -c conf/exp/srn.conf -D /root/project/ShapeNet/cars --lr 1e-4 --recon 50 --name epoch3-G-1e-4-D-1e-1 --disc_lr 0.1 --batch_size 32 --epoch-period 3