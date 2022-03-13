
import os
import yaml
import torch 
import wandb
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from movinets import MoViNet
from movinets.config import _C
from models.model import Model
from torchvision import models
from datetime import datetime as dt
from pytorch_pretrained_biggan import BigGAN
from mmcv import Config, DictAction
from mmaction.models import build_model
from mmcv.runner import get_dist_info, init_dist, load_checkpoint


COLORS = {
    "yellow": "\x1b[33m", 
    "blue": "\x1b[94m", 
    "green": "\x1b[32m",
    "red": "\x1b[33m", 
    "end": "\033[0m"
}

SWINT_CONFIG = '/home/ubuntu/ModelExtraction/xerefic/VideoSwinTransformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'
SWINT_CKPT = '/home/ubuntu/ModelExtraction/xerefic/VideoSwinTransformer/checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth'


class ModifiedBigGAN(nn.Module):
    
    def __init__(self, batch_size, num_classes, truncation, device):
        super(ModifiedBigGAN, self).__init__()
        self.truncation = truncation
        self.batch_size = batch_size 
        self.n_cls = num_classes
        self.device = device
        
        biggan = BigGAN.from_pretrained('biggan-deep-256')
        self.gan = biggan.generator 
        self.embedding = nn.Embedding(num_classes, 128)
        
    def _sample(self):
        noise = np.random.normal(0, 1, size=(self.batch_size, 128))
        noise = np.clip(noise, -self.truncation, self.truncation)
        noise = torch.from_numpy(noise).float().to(self.device)
        
        class_idx = np.random.randint(0, self.n_cls, size=(self.batch_size,))
        class_idx = torch.from_numpy(class_idx).long().to(self.device)
        return noise, class_idx
        
    def forward(self):
        noise, class_idx = self._sample()
        embeds = self.embedding(class_idx)
        inp = torch.cat([noise, embeds], -1)
        with torch.no_grad():
            output = self.gan(inp, 0.4)
        return output
    
    
class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {}

    def add(self, metrics: dict):
        if len(self.metrics) == 0:
            self.metrics = {key: [value] for key, value in metrics.items()}
        else:
            for key, value in metrics.items():
                if key in self.metrics.keys():
                    self.metrics[key].append(value)
                else:
                    self.metrics[key] = [value]
    
    def avg(self):
        return {key: np.mean(value) for key, value in self.metrics.items()}

    def msg(self):
        metrics = self.avg()
        msg = "".join(["[{}] {:.4f} ".format(key, value) for key, value in metrics.items()])
        return msg
    
    
def pbar(progress=0, desc="Progress", status="", barlen=20):
    status = status.ljust(30)
    if progress == 1:
        status = "{}".format(status.ljust(30))
    length = int(round(barlen * progress))
    text = "\r{}: [{}] {:.2f}% {}".format(
        desc, COLORS["green"] + "="*(length-1) + ">" + COLORS["end"] + " " * (barlen-length), progress * 100, status  
    ) 
    print(text, end="" if progress < 1 else "\n") 
    
    
if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--device_idx', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--num_classes', type=int, default=400)
    ap.add_argument('--frame_stack', type=int, default=8)
    ap.add_argument('--train_epochs', type=int, default=500)
    ap.add_argument('--steps_per_epoch', type=int, default=1000)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--weight_decay', type=float, default=1e-05)
    ap.add_argument('--kl_loss_weight', type=float, default=0.8)
    ap.add_argument('--log_wandb', action='store_true', default=False)
    ap.add_argument('--eval_interval', type=int, default=1)
    args = ap.parse_args()
    
    # Victim model
    device = torch.device("cuda:{}".format(args.device_idx) if torch.cuda.is_available() else "cpu")
    cfg = Config.fromfile(SWINT_CONFIG)
    victim = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg')).to(device)
    load_checkpoint(victim, SWINT_CKPT, map_location=device)
    next(victim.parameters()).device
    victim.eval()
    
    # Embedding model
    biggan = ModifiedBigGAN(args.batch_size, args.num_classes, 0.4, device).to(device)
    
    # Student model 
    with open('model_cfg.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    student = Model(cfg).to(device)
    
    # Optimizer and scheduler
    trainable_params = list(biggan.parameters()) + list(student.parameters())
    optimizer = optim.RMSprop(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epochs, eta_min=1e-06)
    
    outdir = os.path.join('biggan_movinet', dt.now().strftime('%d-%m-%Y_%H-%M'))
    os.makedirs(outdir, exist_ok=True)
    best_loss = float('inf')
    
    if args.log_wandb:
        wandb.init(project='biggan_movinet')
    
    print("\n[INFO] Beginning training...\n")
    for epoch in range(1, args.train_epochs+1):
        trainmeter = AverageMeter()
        student.train()
        biggan.train()
        
        for step in range(args.steps_per_epoch):
            imgs = biggan()
            bs, c, h, w = imgs.size()
            imgs = imgs.repeat(1, args.frame_stack, 1, 1).view(-1, args.frame_stack, c, h, w)
            
            with torch.no_grad():
                gt_logits = victim.cls_head(victim.backbone(imgs.transpose(1, 2).contiguous()))
                gt_labels = gt_logits.argmax(-1)
            
            pred_logits = student(imgs)
            pred_labels = pred_logits.argmax(-1)
            
            kl_loss = F.kl_div(F.log_softmax(pred_logits, 1), F.softmax(gt_logits, 1), reduction='batchmean')
            xent_loss = F.cross_entropy(pred_logits, gt_labels)
            loss = args.kl_loss_weight * kl_loss + (1-args.kl_loss_weight) * xent_loss 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = torch.eq(pred_labels, gt_labels.view_as(pred_labels)).sum().item() / gt_labels.numel()
            metrics = {'accuracy': acc, 'kl_loss': kl_loss.item(), 'xent_loss': xent_loss.item()}
            trainmeter.add(metrics)
            
            if args.log_wandb:
                wandb.log(metrics)
                
            pbar((step+1)/args.steps_per_epoch, desc='[Train] Epoch {:4d}'.format(epoch), status=trainmeter.msg())
            break

        print()
        scheduler.step()
            
        if epoch % args.eval_interval == 0:
            valmeter = AverageMeter()
            student.eval()
            biggan.eval()
            
            for step in range(args.steps_per_epoch):
                with torch.no_grad():
                    imgs = biggan()
                    bs, c, h, w = imgs.size()
                    imgs = imgs.repeat(1, args.frame_stack, 1, 1).view(-1, args.frame_stack, c, h, w)
                    
                    gt_logits = victim.cls_head(victim.backbone(imgs.transpose(1, 2).contiguous()))
                    gt_labels = gt_logits.argmax(-1)
                    
                    pred_logits = student(imgs)
                    pred_labels = pred_logits.argmax(-1)
                    
                    kl_loss = F.kl_div(F.log_softmax(pred_logits, 1), F.softmax(gt_logits, 1), reduction='batchmean')
                    xent_loss = F.cross_entropy(pred_logits, gt_labels)
                    loss = args.kl_loss_weight * kl_loss + (1-args.kl_loss_weight) * xent_loss 

                    acc = torch.eq(pred_labels, gt_labels.view_as(pred_labels)).sum().item() / gt_labels.numel()
                    metrics = {'accuracy': acc, 'kl_loss': kl_loss.item(), 'xent_loss': xent_loss.item()}
                    valmeter.add(metrics)
                    
                    pbar((step+1)/args.steps_per_epoch, desc='[ Val ] Epoch {:4d}'.format(epoch), status=valmeter.msg())
                    
            avg = valmeter.avg()
            if args.log_wandb:
                wandb.log({'epoch': epoch, 'val accuracy': avg['accuracy'], 
                           'val kl_loss': avg['kl_loss'], 'val xent_loss': avg['xent_loss']})
                
            total_loss = args.kl_loss_weight * avg['kl_loss'] + (1-args.kl_loss_weight) * avg['xent_loss']
            if total_loss < best_loss:
                best_loss = total_loss
                state = {
                    'epoch': epoch,
                    'student': student.state_dict(),
                    'gan': biggan.state_dict()
                }
                torch.save(state, os.path.join(outdir, 'checkpoint.pth.tar'))