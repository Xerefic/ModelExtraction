from PIL import Image
import numpy as np
import pandas as pd

import os
import cv2
import glob
import time
import random
import gc
import json
import pyprind
import tqdm
from dataclasses import dataclass, field

# from mmcv.parallel import collate
# from mmcv.runner import get_dist_info
# from mmcv.utils import Registry, build_from_cfg

import torch
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from config import TrainingArgs
class TrainingArgs():
    
    seed: int = 420
    lr: float = 1.e-3
    batch_size: int = 8
    num_workers: int = 2
    max_epochs: str = 1000

    image_size: int = 224
    temporal_history: int = 16

    train_file: str = '/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/data/mit/videos/training'
    valid_file: str = None
    test_file: str = None
    checkpoint: str = None

    project_name: str = None
    wandb_run_name: str = None

# from mmaction.datasets.pipelines import Compose

args = TrainingArgs()
# DATASETS = Registry('dataset')
# PIPELINES = Registry('pipeline')
# BLENDINGS = Registry('blending')
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

class CreateDataset(torch.utils.data.Dataset):
    def __init__(self, args, mode='Training'):
        self.args = args
        self.mode = mode
        self._init_video_dataset()
        self.transforms = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                # transforms.RandomResizedCrop(),
                # transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[123.675/255, 116.28/255, 103.53/255], 
                                     std=[58.395/255, 57.12/255, 57.375/255]),
            ]
        )
        # self.transform = Compose(self.args.transform)

        self.entry = []

    def _init_dataset(self):
        self.data = []
        if self.mode == 'Training':
            classes = sorted(os.listdir(self.args.train_file))
            for classname in classes:
                videos = sorted(os.listdir(os.path.join(self.args.train_file, classname)))
                for video in videos:
                    images = sorted(glob.glob(os.path.join(self.args.train_file, classname, video, '*.jpg')))
                    if len(images) <= 8:
                        continue
                    else:
                        for i in range(0, len(images)-8, 8):
                            self.data.append({'Class': classname, 'Video': video, 'Images': images[i:i+8]})

        elif self.mode == 'Validation':
            classes = sorted(os.listdir(self.args.train_file))
            for classname in classes:
                videos = sorted(os.listdir(os.path.join(self.args.train_file, classname)))
                for video in videos:
                    images = sorted(glob.glob(os.path.join(self.args.train_file, classname, video, '*.jpg')))
                    if len(images) <= 8:
                        continue
                    else:
                        for i in range(0, len(images)-8, 8):
                            self.data.append({'Class': classname, 'Video': video, 'Images': images[i:i+8]})

        else:
            print("No Such Dataset Mode")
            return None
        
    def _init_video_dataset(self):
        self.data = []
        
        classes = sorted(os.listdir(self.args.train_file))
        for classname in classes:
            videos = sorted(glob.glob(os.path.join(self.args.train_file, classname, '*.mp4')))
            for vidfile in videos:
                video = cv2.VideoCapture(vidfile)
                success, frame = video.read()
                buffer = []
                while success:
                    buffer.append(frame)
                    if len(buffer) == 8:
                        self.data.append({'Class': classname, 'Video': vidfile, 'Images': buffer})
                        buffer = []
                    success, frame = vid.read()

    # def __getitem__(self, index):

    #     img = torch.empty((0, 3, 224, 224))
    #     for i in range(len(self.data[index]['Images'])):
    #         image = Image.open(self.data[index]['Images'][i])
    #         image = self.transforms(image)
    #         image = image.unsqueeze(0)
    #         img = torch.cat((img, image), dim = 0)
    #     img = img.transpose(1, 0)
    #     images = self.data[index]['Images']
    #     label = self.data[index]['Class']
    #     videoname = self.data[index]['Video']

    #     return img, label, videoname, images
    
    def __getitem__(self, index):
        ''' To be used if init function is _init_video_dataset '''

        img = torch.empty((0, 3, 224, 224))
        for i in range(len(self.data[index]['Images'])):
            image = self.transforms(image)
            image = image.unsqueeze(0)
            img = torch.cat((img, image), dim = 0)
        img = img.transpose(1, 0)
        images = self.data[index]['Images']
        label = self.data[index]['Class']
        videoname = self.data[index]['Video']

        return img, label, videoname, images

    def __len__(self):
        return len(self.data)