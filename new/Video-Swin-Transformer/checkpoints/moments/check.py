import pickle as pkl
import torch
from collections import Counter

with open('/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/checkpoints/moments/logits.pkl', 'rb') as f:
    data = pkl.load(f)

for logit in data:
    print(torch.argmax(torch.Tensor(logit)))