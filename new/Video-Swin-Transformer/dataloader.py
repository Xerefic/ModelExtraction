import argparse
import os
import warnings
import pickle as pkl
import random

import numpy as np
from torch.utils.data import Dataset, DataLoader
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

class data(Dataset):
    def __init__(self):
        cfg = Config.fromfile('/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py')
        self.vid_data = build_dataset(cfg.data.test, dict(test_mode=True))
        # with open('/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/checkpoints/moments/logits.pkl', 'rb') as f:
        #     self.logits = pkl.load(f)

    def __len__(self):
        # print(len(self.logits))
        return len(self.vid_data)

    def __getitem__(self, idx):
        # imgs = self.vid_data[idx]['imgs']
        # logits = self.logits[idx]

        return self.vid_data[idx]

def get_dataloader():

    dataset = data()
    print(len(dataset))
    cfg = Config.fromfile('/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py')
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        dist=False,
        shuffle=False)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
    # full_loader = build_dataloader(dataset, **dataloader_setting)

    train_data, test_data = torch.utils.data.random_split(dataset, [len(dataset)-len(dataset)//10, len(dataset)//10])
    
    # train_loader = DataLoader(train_data, shuffle = True, batch_size=1, num_workers = 4, pin_memory=True)
    # test_loader = DataLoader(test_data, shuffle=False, batch_size=1, num_workers = 2, pin_memory=True)
    # return full_loader

    train_loader = build_dataloader(train_data, **dataloader_setting)
    test_loader = build_dataloader(test_data, **dataloader_setting)

    return train_loader, test_loader


if __name__ == '__main__':

    # cfg = Config.fromfile('/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py')

    # model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    # load_checkpoint(model, '/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth', map_location='cuda:0')
    # print(model)
    # quit()
    train_loader, train_logits = get_dataloader()
    quit()
    # train_loader, train_logits, test_loader, test_logits = get_dataloader()

    for i, (imgs, logits) in enumerate(train_loader):
        #print(imgs.size())
        #print(logits.size())
        print(torch.argmax(logits))
        
        #with torch.no_grad():
            #out = model(return_loss=False, **data)
            #out = torch.Tensor(out)
            #print(out)
            # print(torch.max(out))
            #quit()
        #print(torch.Tensor(out-logit).sum())
        if i>=100:
            quit()
