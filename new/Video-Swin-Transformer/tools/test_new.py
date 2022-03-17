import argparse
import os
import warnings
import pickle as pkl
import random
import numpy as np
import torchvision.transforms as transforms
from glob import glob
import yaml
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
from mmcv.runner import load_checkpoint
from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model

cfg = Config.fromfile('/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py')

teacher =  build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, '/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth', map_location='cuda:0')

for data in data_loader:
    with torch.no_grad():
        data['imgs'] = data['imgs'].to('cuda:1')
        result = model(return_loss=False, **data)
        break
    results.extend(result)

    # use the first key as main key to calculate the batch size
    batch_size = len(next(iter(data.values())))
    for _ in range(batch_size):
        prog_bar.update()
    # with open('/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/checkpoints/moments/logits_new.pkl', 'wb') as f: