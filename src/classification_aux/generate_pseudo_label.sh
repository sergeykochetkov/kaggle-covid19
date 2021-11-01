#!/bin/bash

set -x
'
cfg1=eb5_512_deeplabv3plus.yaml
cfg2=eb6_448_linknet.yaml
cfg3=eb7_512_unetplusplus.yaml
cfg4=seresnet152d_512_unet.yaml
cfgs=( $cfg1 $cfg2 $cfg3 $cfg4 )
folds=0
'

cfg1=configs/unittest_eb3_128_deeplabv3plus.yaml
cfgs=( $cfg1 $cfg1 )
folds=0

'
#predict softlabel for public test
for cfg in "${cfgs[@]}"
do
CUDA_VISIBLE_DEVICES=0 python predict_test.py --cfg $cfg --ckpt_dir $1 --folds $folds
done

#predict mask for public test using segmentation head
for cfg in "${cfgs[@]}"
do
CUDA_VISIBLE_DEVICES=0 python predict_test_seg.py --cfg $cfg --ckpt_dir $1 --folds $folds
done

#ensemble 4 models eb5, eb6, eb7, seresnet152
python ensemble_pseudo_test.py --cfgs ${cfgs[@]} --folds $folds
'

#predict softlabel for padchest, pneumothorax, vin
for cfg in "${cfgs[@]}"
do
CUDA_VISIBLE_DEVICES=0 python predict_ext.py --cfg $cfg --folds $folds  #--ckpt_dir $1
done

python ensemble_pseudo_ext.py --cfgs ${cfgs[@]} --folds $folds