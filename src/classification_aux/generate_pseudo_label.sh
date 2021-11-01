cfg1=eb5_512_deeplabv3plus.yaml
cfg2=eb6_448_linknet.yaml
cfg3=eb7_512_unetplusplus.yaml
cfg4=seresnet152d_512_unet.yaml
folds=0

#predict softlabel for public test
CUDA_VISIBLE_DEVICES=0 python predict_test.py --cfg configs/$cfg1 --ckpt_dir $1 --folds $folds
CUDA_VISIBLE_DEVICES=0 python predict_test.py --cfg configs/$cfg2 --ckpt_dir $1 --folds $folds
CUDA_VISIBLE_DEVICES=0 python predict_test.py --cfg configs/$cfg3 --ckpt_dir $1 --folds $folds
CUDA_VISIBLE_DEVICES=0 python predict_test.py --cfg configs/$cfg4 --ckpt_dir $1 --folds $folds
#predict mask for public test using segmentation head
CUDA_VISIBLE_DEVICES=0 python predict_test_seg.py --cfg configs/$cfg1 --ckpt_dir $1 --folds $folds
CUDA_VISIBLE_DEVICES=0 python predict_test_seg.py --cfg configs/$cfg2 --ckpt_dir $1 --folds $folds
CUDA_VISIBLE_DEVICES=0 python predict_test_seg.py --cfg configs/$cfg3 --ckpt_dir $1 --folds $folds
CUDA_VISIBLE_DEVICES=0 python predict_test_seg.py --cfg configs/$cfg4 --ckpt_dir $1 --folds $folds
#ensemble 4 models eb5, eb6, eb7, seresnet152
python ensemble_pseudo_test.py
#predict softlabel for padchest, pneumothorax, vin
CUDA_VISIBLE_DEVICES=0 python predict_ext.py --cfg configs/$cfg1 --ckpt_dir $1 --folds $folds
CUDA_VISIBLE_DEVICES=0 python predict_ext.py --cfg configs/$cfg2 --ckpt_dir $1 --folds $folds
CUDA_VISIBLE_DEVICES=0 python predict_ext.py --cfg configs/$cfg3 --ckpt_dir $1 --folds $folds
CUDA_VISIBLE_DEVICES=0 python predict_ext.py --cfg configs/$cfg4 --ckpt_dir $1 --folds $folds
python ensemble_pseudo_ext.py