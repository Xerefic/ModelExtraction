from mmcv import Config, DictAction
from mmaction.models import build_model
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
import torch
import torch.nn as nn

config = '/home/ubuntu/ModelExtraction/xerefic/VideoSwinTransformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'
checkpoint = '/home/ubuntu/ModelExtraction/xerefic/VideoSwinTransformer/checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth'



if __name__ == '__main__':

    cfg = Config.fromfile(config)
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint, map_location='cpu')
    # print(model)
    # [batch_size, channel, temporal_dim, height, width]
    dummy_x = torch.rand(1, 3, 32, 224, 224)

    # SwinTransformer3D without cls_head
    backbone = model.backbone
    head = model.cls_head
    # [batch_size, hidden_dim, temporal_dim/2, height/32, width/32]
    feat = head(backbone(dummy_x))
    print(backbone)
    print(feat.shape)