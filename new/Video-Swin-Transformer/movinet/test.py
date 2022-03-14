import argparse
from importlib import import_module
import os
import os.path as osp
import warnings
import pickle as pkl

import torch
import torchvision
from torchvision.utils import save_image
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms

num_frames = 16
clip_steps = 2
batch_size = 16



# dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)

# print(next(iter(dataloader)))