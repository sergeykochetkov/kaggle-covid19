#!/bin/bash

BUCKET=s3://kochetkov-kaggle-covid19

cd dataset
aws s3 cp $BUCKET/siim-covid19-detection.zip .
unzip siim-covid19-detection.zip

cd external_dataset
for d in chest14.zip chexpert.zip pneumothorax.zip rsna-pneumonia-detection-challenge.zip vinbigdata.zip
do
  aws s3 cp $BUCKET/$d .
  unzip $d
done