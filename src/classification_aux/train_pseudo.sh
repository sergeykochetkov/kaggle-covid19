CUDA_VISIBLE_DEVICES=0 python train_pseudo.py --cfg configs/eb5_512_deeplabv3plus.yaml --folds 0 --ckpt_in_dir $1 --ckpt_out_dir $2  &
CUDA_VISIBLE_DEVICES=1 python train_pseudo.py --cfg configs/eb6_448_linknet.yaml --folds 0 --ckpt_in_dir $1 --ckpt_out_dir $2
#CUDA_VISIBLE_DEVICES=0 python train_pseudo.py --cfg configs/eb7_512_unetplusplus.yaml --folds 0 1 2 3 4 --ckpt_in_dir $1 --ckpt_out_dir $2
#CUDA_VISIBLE_DEVICES=0 python train_pseudo.py --cfg configs/seresnet152d_512_unet.yaml --folds 0 1 2 3 4 --ckpt_in_dir $1 --ckpt_out_dir $2
