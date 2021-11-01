#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

FRAC=0.2

CFG=configs/unittest_eb3_128_deeplabv3plus.yaml

rm chexpert_chest14_pretrain/timm-efficientnet-b3_128_pretrain_step0.pth
rm chexpert_chest14_pretrain/timm-efficientnet-b3_128_pretrain_step1.pth
rm rsnapneu_pretrain/timm-efficientnet-b3_128_deeplabv3plus_rsnapneu.pth
rm checkpoints/timm-efficientnet-b3_128_deeplabv3plus_aux_fold0.pth

python train_chexpert_chest14.py --steps 0 1 --cfg $CFG --frac=$FRAC  &&

python train_rsnapneu.py --cfg $CFG --epochs=2 --frac=$FRAC &&

python train_siim.py --cfg $CFG --folds 0 --frac=$FRAC