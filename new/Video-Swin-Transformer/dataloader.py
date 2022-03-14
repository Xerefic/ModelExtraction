import argparse
import os
import warnings
import pickle as pkl

import mmcv
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model

def main():

    cfg = Config.fromfile('/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py')

    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        dist=False,
        shuffle=False)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    # model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    # load_checkpoint(model, '/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth', map_location='cpu')

    with open('/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/checkpoints/moments/logits.pkl', 'rb') as f:
        logits = pkl.load(f)

    for i, data in enumerate(data_loader):
        imgs = data['imgs']
        logit = logits[i]
        print(imgs.shape)
        quit()

if __name__ == '__main__':
    main()