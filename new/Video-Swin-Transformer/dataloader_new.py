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

random.seed(17)
np.random.seed(17)
torch.manual_seed(17)
torch.cuda.manual_seed(17)
torch.cuda.manual_seed_all(17)

class MIT(Dataset):
    def __init__(self, root_dir):
        self.data_dir = root_dir 
        self._init_dataset()
        # if transform:
        self._init_transform()
    
    def _init_dataset(self):
        self.files = []
        dirs = sorted(os.listdir(os.path.join(self.data_dir, 'training')))
        for dir in range(len(dirs)):
            files = sorted(glob(os.path.join(self.data_dir, 'training', dirs[dir], '*.mp4')))         
            self.files += files
        
    def _init_transform(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
        ])  
    
    def __getitem__(self, index):
        
        vid = cv2.VideoCapture(self.files[index])
        # length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        # fps = vid.get(cv2.CAP_PROP_FPS)
        # chunksize = length // 32
        
        success, frame = vid.read()
        buffer = []
        # count = 0

        sec = 0 
        frameRate = 0.04
        success, frame = vid.read()
        while success: 
            sec = sec + frameRate 
            sec = round(sec, 2) 
            buffer.append(frame)
            vid.set(cv2.CAP_PROP_POS_MSEC,sec*1000) 
            success, frame = vid.read()
        
        frames = np.stack(buffer, 0)
        frames = torch.from_numpy(frames).float().permute(0, 3, 1, 2).contiguous()
        new_frames = torch.empty((0, 3, 224, 224))
        for i in range(0, frames.shape[0], 2):
            if i < 64:
                new_frames = torch.cat((new_frames, self.transform(frames[i]).unsqueeze(0)), dim = 0)
        return new_frames, self.files[index]

    def __len__(self):
        return len(self.files)

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
    
    trainset = MIT(root_dir='/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/data/mit/videos')
    train_dataloader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=4)
    # print('Started')
    # bar = pyprind.ProgBar(30500, bar_char='█')
    # for i in range(30500):
    #     print(trainset[i].shape[1])
    #     if trainset[i].shape[1] != 32:
    #         print(trainset[i].shape[1])
    #     bar.update()
    # print('Ended')
    # quit()
    # print(trainset[0].shape)
    # quit()
    # cfg = Config.fromfile('/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py')

    # model =  build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    # load_checkpoint(model, '/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth', map_location=device)

    config = '/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'
    checkpoint = '/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth'

    cfg = Config.fromfile(config)
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg')).to(device)
    load_checkpoint(model, checkpoint, map_location=device)
    bar = pyprind.ProgBar(len(trainset), bar_char='█')
    results = {}
    model.eval()
    for i, (imgs, vidid) in enumerate(train_dataloader):
        with torch.no_grad():
            imgs = imgs.to(device)
            imgs = imgs.permute((0, 2, 1, 3, 4))
            logits = model.cls_head(model.backbone(imgs))
            results[vidid] = logits
            bar.update()
            if (i+1)%50 == 0:
                with open('/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/checkpoints/videoswin/logits.pkl', 'wb') as f:
                    pkl.dump(results, f)