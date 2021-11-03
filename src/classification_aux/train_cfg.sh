#!/bin/bash

CFG=$1

python train_chexpert_chest14.py --steps 0 1 --cfg $CFG  &&

python train_rsnapneu.py --cfg $CFG  &&

python train_siim.py --cfg $CFG --folds 0