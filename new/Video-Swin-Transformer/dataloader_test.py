import argparse
import os
import warnings
import pickle as pkl
import random

import numpy as np
import mmcv
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model


random.seed(17)
np.random.seed(17)
torch.manual_seed(17)
torch.cuda.manual_seed(17)
torch.cuda.manual_seed_all(17)

torch.set_printoptions(precision=8)

def get_trainloader():

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

    with open('/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/checkpoints/moments/logits.pkl', 'rb') as f:
        logits = pkl.load(f)
    # random.shuffle(logits)

    train_data, test_data = torch.utils.data.random_split(dataset, [len(dataset)-len(dataset)//10, len(dataset)//10])
    # train_logits, test_logits = torch.utils.data.random_split(logits, [len(dataset)-len(dataset)//10, len(dataset)//10])

    train_loader = build_dataloader(train_data, **dataloader_setting)
    test_loader = build_dataloader(test_data, **dataloader_setting)

    # return dataset
    return data_loader, logits
    # return train_loader, train_logits, test_loader, test_logits
    return train_loader, test_loader

def get_dataloader():

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

    with open('/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/checkpoints/moments/logits_new.pkl', 'rb') as f:
        logits = pkl.load(f)

    return data_loader, logits


if __name__ == '__main__':

    cfg = Config.fromfile('/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py')

    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, '/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth', map_location='cuda:0')
    # print(model)

    # dataloader, _ = get_dataloader()
    # print(next(iter(dataloader)))

    # del dataloader

    # print("\n\n")

    # dataloader, _ = get_dataloader()
    # print(next(iter(dataloader)))

    # print("Done")


    # quit()
    train_loader, train_logits = get_dataloader()
    # train_loader, train_logits, test_loader, test_logits = get_dataloader()

    for i, data in enumerate(train_loader):
        imgs = data['imgs']
        logit = train_logits[i]
        logit = torch.Tensor(logit)
        # print(torch.max(logit))
        print(logit)
        # quit()
        with torch.no_grad():
            out = model(return_loss=False, **data)
            out = torch.Tensor(out)
            print(out)
            # print(torch.max(out))
            quit()
        print(torch.Tensor(out-logit).sum())
        if i>=5:
            quit()