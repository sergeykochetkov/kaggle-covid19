#!/bin/bash

conda create -n covid19 python=3.7.9
conda activate covid19
conda install -c conda-forge gdcm
pip install -r requirements.txt
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
pip install git+https://github.com/bes-dev/mean_average_precision.git@930df3618c924b694292cc125114bad7c7f3097e
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia