from PIL import Image

import numpy as np
import pandas as pd
import pickle as pkl

import os
import glob

from mmcv.utils import Registry

import torch

from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import TrainingArgs

args = TrainingArgs()
DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')
BLENDINGS = Registry('blending')
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

class CreateDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.train_file = config['train_file']
        with open('./checkpoints/video_swin/logits.pkl', 'rb') as f:
            self.p_data = pkl.load(f)

        self._init_dataset()
        self.transforms = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                # transforms.RandomResizedCrop(),
                # transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),

            ]
        )


    def _init_dataset(self):
        self.data = []
        classes = sorted(os.listdir(self.train_file))
        for classname in classes:
            videos = sorted(os.listdir(os.path.join(self.train_file, classname)))
            for video in videos:
                images = sorted(glob.glob(os.path.join(self.train_file, classname, video, '*.jpg')))
                if len(images) <= 32:
                    continue
                else:
                    for i in range(0, len(images)-32, 32):
                        self.data.append({'Class': classname, 'Video': video, 'Images': images[i:i+32]})


    def __getitem__(self, index):

        img = torch.empty((0, 3, 224, 224))
        for i in range(len(self.data[index]['Images'])):
            image = Image.open(self.data[index]['Images'][i])
            image = self.transforms(image)
            image = image.unsqueeze(0)
            img = torch.cat((img, image), dim = 0)
        img = img.transpose(1, 0)
        images = self.data[index]['Images']
        label = self.data[index]['Class']
        videoname = self.data[index]['Video']
        # print(self.p_data.keys())
        logit_dict_list = self.p_data[tuple([label])][tuple([videoname])]
        for logit_dict in logit_dict_list:
            if logit_dict['Images'][0][0] == images[0]:
                logits = logit_dict['Logits']

        return img, logits, label, videoname, images



    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    train_dataset = CreateDataset(args)
    # print(train_dataset[0][0].shape)
    print(len(train_dataset))
    # print(len(val_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers = 2, drop_last=True)
    for i, (img, logit, label, vidid, imgid) in enumerate(train_dataloader):
        # img = img.to(device)
        # label = label.to(device)
        # vidid = vidid.to(device)
        print(img.shape)
        print(logit)
        print(label)
        print(vidid)
        break
        # print(imgid)
        # print(label)
        # print(vidid)
        quit()
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers = 2, drop_last=True)
