

CUDA_VISIBLE_DEVICES=0,1 python train_chexpert_chest14.py --steps 0 1 --cfg configs/eb3_384_deeplabv3plus.yaml

exit 2

CUDA_VISIBLE_DEVICES=0,1 python train_chexpert_chest14.py --steps 0 1 --cfg configs/eb6_448_linknet.yaml
CUDA_VISIBLE_DEVICES=0,1 python train_chexpert_chest14.py --steps 0 1 --cfg configs/eb7_512_unetplusplus.yaml
CUDA_VISIBLE_DEVICES=0,1 python train_chexpert_chest14.py --steps 0 1 --cfg configs/seresnet152d_512_unet.yaml