import pickle as pkl
import torch
from collections import Counter

with open('/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/checkpoints/movinet/logits.pkl', 'rb') as f:
    data = pkl.load(f)

for logit in data:
    print(torch.mode(torch.argmax(torch.Tensor(logit), dim = 1), dim = 0)[0])
    # print(torch.argmax(torch.Tensor(logit), dim = 1))
