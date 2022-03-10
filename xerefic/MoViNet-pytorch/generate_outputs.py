import time
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import torch
# import torchtransforms as T
from movinets import MoViNet
from movinets.config import _C
# import dataloader
from dataloader import *
import pickle

args = TrainingArgs()
train_dataset = CreateDataset(args)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers = 32, drop_last=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

model = MoViNet(_C.MODEL.MoViNetA2, causal = True, pretrained = True ).to(device)
model.eval()

predicted = {}

with torch.no_grad():
    bar = pyprind.ProgBar(len(train_dataloader), bar_char='█')
    for i, (img, label, vidid, imgid) in enumerate(train_dataloader):
        torch.cuda.empty_cache()
        img = img.to(device)
        output = model(img).squeeze(0)

        if label in predicted.keys():
            if vidid in predicted[label].keys():
                predicted[label][vidid].append({'Images': imgid, 'Logits': output.cpu()})
            else:
                predicted[label][vidid] = [{'Images': imgid, 'Logits': output.cpu()}]
        else:
            predicted[label] = {vidid: [{'Images': imgid, 'Logits': output.cpu()}]}

        bar.update()

        torch.cuda.empty_cache()

with open('/home/ubuntu/ModelExtraction/xerefic/MoViNet-pytorch/checkpoints/movinet/logits.pkl', 'wb') as f:
        pickle.dump(predicted, f)


# model = MoViNet(_C.MODEL.MoViNetA0, causal = True, pretrained = True )

