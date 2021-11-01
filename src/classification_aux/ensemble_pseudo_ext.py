import numpy as np
import os
import pandas as pd
import torch
import yaml
import argparse

from utils import seed_everything
from dataset import classes
from predict_test import cfg_to_preds_path

import warnings

warnings.filterwarnings("ignore")

SEED = 123
seed_everything(SEED)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cfgs", default=[
        'configs/eb5_512_deeplabv3plus.yaml',
        'configs/eb6_448_linknet.yaml',
        'configs/eb7_512_unetplusplus.yaml',
        'configs/seresnet152d_512_unet.yaml'], nargs="+", type=str)
    parser.add_argument("--folds", default=[0, 1, 2, 3, 4], nargs="+", type=int)

    args = parser.parse_args()

    cfgs = []
    for cfg in args.cfgs:
        with open(cfg) as f:
            cfgs.append(yaml.load(f, Loader=yaml.FullLoader))

    os.makedirs('pseudo_csv', exist_ok=True)
    for source in ['pneumothorax', 'vin']:
        if source == 'pneumothorax':
            test_df = pd.read_csv('../../dataset/external_dataset/ext_csv/pneumothorax.csv')
        elif source == 'vin':
            test_df = pd.read_csv('../../dataset/external_dataset/ext_csv/vin.csv')

        study_pred_list = [torch.load(cfg_to_preds_path(cfg, args.folds, source))['pred_dict'] for cfg in cfgs]

        weights = [0.3, 0.2, 0.2, 0.3]
        weights = weights[:len(study_pred_list)]
        weights = np.array(weights)
        weights /= np.sum(weights)

        image_paths = []
        labels = []
        for _, row in test_df.iterrows():
            pred = 0
            for p, w in zip(study_pred_list, weights):
                pred += w * p[row['image_path']]

            image_path = row['image_path']
            assert os.path.isfile(image_path) == True
            image_paths.append(image_path)

            labels.append(pred)
        pseudo_test_df = pd.DataFrame()
        pseudo_test_df['image_path'] = np.array(image_paths)
        pseudo_test_df[classes] = np.array(labels, dtype=float)
        pseudo_test_df['pseudo'] = np.array([True] * len(test_df), dtype=bool)
        pseudo_test_df.to_csv('pseudo_csv/pseudo_{}.csv'.format(source), index=False)
