import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import yaml

import torch
from torch.utils.data import DataLoader

from models import SiimCovidAuxModel
from dataset import SiimCovidCLSTestDataset, classes
from utils import seed_everything

import warnings

warnings.filterwarnings("ignore")


def cfg_to_preds_path(cfg, folds, source='test'):
    return 'predictions/{}_{}_{}_aux_fold{}_{}_pred_8tta.pth'.format(cfg['encoder_name'], cfg['aux_image_size'],
                                                                     cfg['decoder'], '_'.join(str(x) for x in folds),
                                                                     source)


SEED = 123
seed_everything(SEED)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cfg", default='configs/seresnet152d_512_unet.yaml', type=str)
    parser.add_argument("--folds", default=[0, 1, 2, 3, 4], nargs="+", type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--workers", default=16, type=int)
    parser.add_argument("--ckpt_dir", default='checkpoints_v4', type=str)
    args = parser.parse_args()
    print(args)

    os.makedirs('predictions', exist_ok=True)

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    print(cfg)

    test_df = pd.read_csv('../../dataset/siim-covid19-detection/test_meta.csv')

    models = {}
    for fold in args.folds:
        CHECKPOINT = '{}/{}_{}_{}_aux_fold{}.pth'.format(args.ckpt_dir, cfg['encoder_name'], cfg['aux_image_size'],
                                                         cfg['decoder'], fold)
        models[fold] = SiimCovidAuxModel(
            encoder_name=cfg['encoder_name'],
            encoder_weights=None,
            decoder=cfg['decoder'],
            classes=len(classes),
            in_features=cfg['in_features'],
            decoder_channels=cfg['decoder_channels'],
            encoder_pretrained_path=None,
            encoder_pretrained_num_classes=None,
            model_pretrained_path=None,
            model_pretrained_num_classes=None,
            test_mode=True).cuda()
        models[fold].load_state_dict(torch.load(CHECKPOINT))
        models[fold].eval()

    test_dataset = SiimCovidCLSTestDataset(
        df=test_df,
        images_dir='../../dataset/siim-covid19-detection/images/test',
        image_size=cfg['aux_image_size'])

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    print('Test size: {}'.format(len(test_loader.dataset)))

    preds = []
    imageids = []
    for ids, images, images_center_crop in tqdm(test_loader):
        images = images.cuda()
        images_center_crop = images_center_crop.cuda()
        imageids.extend(ids)

        pred = []
        with torch.cuda.amp.autocast(), torch.no_grad():
            for fold in args.folds:
                pred.append(torch.sigmoid(models[fold](images)))
                pred.append(torch.sigmoid(models[fold](torch.flip(images, dims=(3,)).contiguous())))
                pred.append(torch.sigmoid(models[fold](torch.flip(images, dims=(2,)).contiguous())))
                pred.append(torch.sigmoid(models[fold](torch.flip(images, dims=(2, 3)).contiguous())))
                pred.append(torch.sigmoid(models[fold](images_center_crop)))
                pred.append(torch.sigmoid(models[fold](torch.flip(images_center_crop, dims=(3,)).contiguous())))
                pred.append(torch.sigmoid(models[fold](torch.flip(images_center_crop, dims=(2,)).contiguous())))
                pred.append(torch.sigmoid(models[fold](torch.flip(images_center_crop, dims=(2, 3)).contiguous())))

        pred = torch.mean(torch.stack(pred, -1), -1).data.cpu().numpy()
        preds.append(pred)
        del pred

    preds = np.concatenate(preds, axis=0)
    imageids = np.array(imageids)

    pred_dict = dict(zip(imageids, preds))
    pred_dict_path = cfg_to_preds_path(cfg, args.folds)
    torch.save({
        'pred_dict': pred_dict,
    }, pred_dict_path)
