import pickle

import pickle as pkl
import torch
from collections import Counter

with open('/home/ubuntu/ModelExtraction/xerefic/MoViNet-pytorch/checkpoints/kinetics400/logits.pkl', 'rb') as f:
    data = pkl.load(f)

l = []
for classes in data.keys():
    for video in data[classes].keys():
        for image in data[classes][video]:
            logits = image['Logits']
            out = torch.argmax(logits)
            l.append(out.item())

print(Counter(l))