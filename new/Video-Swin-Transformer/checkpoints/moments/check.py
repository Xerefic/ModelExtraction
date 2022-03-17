import pickle as pkl
import torch
from collections import Counter


def get_dataloader():

    cfg = Config.fromfile('/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py')

    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        dist=False,
        shuffle=False)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    with open('/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/checkpoints/moments/logits_new.pkl', 'rb') as f:
        logits = pkl.load(f)

    # train_data, test_data = torch.utils.data.random_split(dataset, [len(dataset)-len(dataset)//10, len(dataset)//10])
    # train_logits, test_logits = torch.utils.data.random_split(logits, [len(dataset)-len(dataset)//10, len(dataset)//10])

    # train_loader = build_dataloader(train_data, **dataloader_setting)
    # test_loader = build_dataloader(test_data, **dataloader_setting)

    return data_loader, logits

with open('/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/checkpoints/moments/logits_new.pkl', 'rb') as f:
    data = pkl.load(f)


print(len(data))
for logit in data:
    print(torch.argmax(torch.Tensor(logit)))