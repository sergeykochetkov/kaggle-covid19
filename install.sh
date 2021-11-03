#!/bin/bash

#

#sudo mkdir /root/.kaggle
#sudo cp kaggle.json /root/.kaggle
'''

apt-get update
apt-get install git -y
apt-get install wget -y
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh

git clone https://github.com/sergeykochetkov/kaggle-covid19.git
cd kaggle-covid19
'''

'''
SSH server: 193.106.172.198
Username: root
Password: 8kTiKM1K&xx55%*
'''

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install

conda create -n covid19 python=3.7.9 -y
conda activate covid19
conda install -c conda-forge gdcm -y
pip install -r requirements.txt
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
pip install git+https://github.com/bes-dev/mean_average_precision.git@930df3618c924b694292cc125114bad7c7f3097e
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia -y

./download_datasets.sh