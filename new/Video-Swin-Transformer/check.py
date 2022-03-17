import pickle as pkl
import torch
from collections import Counter
import argparse
import os
import warnings
import pickle as pkl
import random
import numpy as np
import torchvision.transforms as transforms
from glob import glob
from mmcv import Config
import yaml
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import time
from mmcv import Config, DictAction
from mmaction.models import build_model
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
import pyprind
import dataloader_new

random.seed(17)
np.random.seed(17)
torch.manual_seed(17)
torch.cuda.manual_seed(17)
torch.cuda.manual_seed_all(17)

if __name__ == '__main__':

    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    torch.cuda.manual_seed(17)
    torch.cuda.manual_seed_all(17)

    with open('train_config.yaml', 'r') as yml_file:
        config = yaml.full_load(yml_file)
        print("Config:", config)
        device = torch.device('cuda:3')
        lambd = config['train']['lambda']
        out_dir = config['out_dir']
        batch_size = config['train']['batch_size']
        num_workers = config['train']['num_workers']
    
    # trainset = dataloader_new.MIT(root_dir='/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/data/mit/videos')
    # print(trainset[0].shape)
    # quit()
    # cfg = Config.fromfile('/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py')

    # model =  build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    # load_checkpoint(model, '/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth', map_location=device)

    # config = '/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'
    # checkpoint = '/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth'

    # cfg = Config.fromfile(config)
    # model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    # load_checkpoint(model, checkpoint, map_location=device)
    # # quit()
    
    # bar = pyprind.ProgBar(len(trainset), bar_char='█')
    # results = []
    # model.eval()
    # ct = 0
    # for imgs in trainset:
    #     with torch.no_grad():
    #         imgs = imgs.permute((0, 2, 1, 3, 4))
    #         logits = model.cls_head(model.backbone(imgs)).squeeze(0)
    #         # logits = model(imgs = imgs, return_loss=False)
    #         # results.extend(logits)
    #         # bar.update()
    #         # ct = ct + 1
    #         # if ct%100 == 0:
    #         #     with open('/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/checkpoints/videoswin/logits.pkl', 'wb') as f:
    #         #         pkl.dump(results, f)
        

    with open('/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/checkpoints/videoswin/logits.pkl', 'rb') as f:
        data = pkl.load(f)

    # print(len(data))
    ct = 0
    print(len(data))
    # for i in range(len(data)):
    #     print(data[i].shape)
    quit()

    for logit in data:
        print(torch.argmax(torch.Tensor(logit).unsqueeze(0)))
        ct = ct + 1
        if ct%50 == 0:
            quit()