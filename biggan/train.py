
import os
import cv2
import yaml
import json
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
from torchvision import models, transforms
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
CLS_MAP = '/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/data/kinetics400/kinetics400_val_list_videos.txt'

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
        return output, class_idx
    
    
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


def kinetics_name_to_idx():
    with open(CLS_MAP, 'r') as f:
        lines = f.read().split('\n')
    mapping = {}
    for line in lines:
        if len(line) > 0:
            fname, idx = line.split()
            classname = fname.split('/')[0]
            mapping[classname] = int(idx)
    return mapping


@torch.no_grad()
def eval_on_kinetics(data_root, student_model):

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
    ])

    cls_folders = sorted(os.listdir(data_root))        
    name_to_idx = kinetics_name_to_idx()
    clswise_correct = {name: 0 for name in enumerate(cls_folders)}
    clswise_count = {name: 0 for name in enumerate(cls_folders)}

    for folder in cls_folders:
        vidfiles = sorted([f for f in os.listdir(os.path.join(data_root, folder)) if f.endswith('mp4')])
        for i, vf in enumerate(vidfiles):
            path = os.path.join(data_root, folder, vf)
            vid = cv2.VideoCapture(path)
            length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            chunksize = length // 8
            
            success, frame = vid.read()
            buffer = []
            count = 0

            while success:
                count += 1
                if count % chunksize == 0 and len(buffer) < 8:
                    buffer.append(frame)

                success, frame = vid.read()
            
            frames = np.stack(buffer, 0)
            frames = torch.from_numpy(frames).float().permute(0, 3, 1, 2).contiguous()
            frames = transform(frames)
            frames = frames.unsqueeze(0).to(device)

            # Generate output
            pred = student_model(frames).argmax(-1).item()
            if pred == name_to_idx[folder]:
                clswise_correct[folder] += 1
            clswise_count[folder] += 1

            pbar((i+1)/len(vidfiles), desc=folder, status='')

    clswise_acc = {}
    for folder in clswise_correct.keys():
        clswise_acc[folder] = clswise_correct[folder] / clswise_count[folder]
    
    return clswise_acc

    
if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', type=str, default='train')
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
    

    if args.task == 'train':
        # Victim model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        embed_optim = optim.RMSprop(biggan.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        student_optim = optim.RMSprop(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        embed_sched = optim.lr_scheduler.CosineAnnealingLR(embed_optim, T_max=args.train_epochs, eta_min=1e-06)
        student_sched = optim.lr_scheduler.CosineAnnealingLR(student_optim, T_max=args.train_epochs, eta_min=1e-06)
        
        outdir = os.path.join('biggan', dt.now().strftime('%d-%m-%Y_%H-%M'))
        os.makedirs(outdir, exist_ok=True)
        best_acc = 0
        
        if args.log_wandb:
            wandb.init(project='biggan_movinet')
        
        print("\n[INFO] Beginning training...\n")
        for epoch in range(1, args.train_epochs+1):
            # embed_trainmeter = AverageMeter()
            # biggan.train()

            # for step in range(args.steps_per_epoch // 2):
            #     imgs, class_idx = biggan()
            #     bs, c, h, w = imgs.size()
            #     imgs = imgs.repeat(1, args.frame_stack, 1, 1).view(-1, args.frame_stack, c, h, w)

            #     gt_logits = victim.cls_head(victim.backbone(imgs.transpose(1, 2).contiguous()))
            #     loss = F.cross_entropy(gt_logits, class_idx)
            #     preds = gt_logits.argmax(-1)
            #     acc = torch.eq(preds, class_idx.view_as(preds)).sum().item() / preds.numel()

            #     embed_optim.zero_grad()
            #     loss.backward()
            #     embed_optim.step()

            #     metrics = {'embed_loss': loss.item(), 'embed_acc': acc}
            #     embed_trainmeter.add(metrics)

            #     if args.log_wandb:
            #         wandb.log(metrics)

            #     pbar(
            #         progress=(step+1)/(args.steps_per_epoch//2), 
            #         desc='[Embed train] Epoch {:4d}'.format(epoch),
            #         status=embed_trainmeter.msg()
            #     )

            student_trainmeter = AverageMeter()
            student.train()
            biggan.train()
            
            for step in range(args.steps_per_epoch):
                
                imgs, _ = biggan()
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
                
                student_optim.zero_grad()
                loss.backward()
                student_optim.step()
                
                acc = torch.eq(pred_labels, gt_labels.view_as(pred_labels)).sum().item() / gt_labels.numel()
                metrics = {'accuracy': acc, 'kl_loss': kl_loss.item(), 'xent_loss': xent_loss.item()}
                student_trainmeter.add(metrics)
                
                if args.log_wandb:
                    wandb.log(metrics)
                    
                pbar(
                    progress=(step+1)/args.steps_per_epoch, 
                    desc='[Stdnt train] Epoch {:4d}'.format(epoch), 
                    status=student_trainmeter.msg()
                )
                break
            print()

            # embed_sched.step()
            student_sched.step()
                
            # Evaluation
            if epoch % args.eval_interval == 0:

                with torch.no_grad():
                    clswise_acc = eval_on_kinetics(
                        data_root='../new/Video-Swin-Transformer/data/kinetics400/validation',
                        student_model=student
                    )

                with open(os.path.join(outdir, f'clswise_acc_epoch_{epoch}.json'), 'w') as f:
                    json.dump(clswise_acc, indent=4)

                # embed_valmeter = AverageMeter()
                # biggan.eval()
                # student.eval()

                # for step in range(args.steps_per_epoch // 2):
                #     with torch.no_grad():
                #         imgs, class_idx = biggan()
                #         bs, c, h, w = imgs.size()
                #         imgs = imgs.repeat(1, args.frame_stack, 1, 1).view(-1, args.frame_stack, c, h, w)
                #         gt_logits = victim.cls_head(victim.backbone(imgs.transpose(1, 2).contiguous()))

                #     loss = F.cross_entropy(gt_logits, class_idx)
                #     preds = gt_logits.argmax(-1)
                #     acc = torch.eq(preds, class_idx.view_as(preds)).sum().item() / preds.numel()
                    
                #     metrics = {'embed_loss': loss.item(), 'embed_acc': acc}
                #     embed_valmeter.add(metrics)

                #     pbar(
                #         progress=(step+1)/(args.steps_per_epoch // 2), 
                #         desc='[Embed val] Epoch {:4d}'.format(epoch),
                #         status=embed_valmeter.msg()
                #     )

                # avg = embed_valmeter.avg()
                # if args.log_wandb:
                #     wandb.log({
                #         'epoch': epoch, 
                #         'val embed_loss': avg['embed_loss'], 
                #         'val embed_acc': avg['embed_acc']
                #     })

                # student_valmeter = AverageMeter()
                # biggan.eval()
                # student.eval()

                # for step in range(args.steps_per_epoch):
                    
                #     with torch.no_grad():
                #         imgs, _ = biggan()
                #         bs, c, h, w = imgs.size()
                #         imgs = imgs.repeat(1, args.frame_stack, 1, 1).view(-1, args.frame_stack, c, h, w)
                        
                #         gt_logits = victim.cls_head(victim.backbone(imgs.transpose(1, 2).contiguous()))
                #         gt_labels = gt_logits.argmax(-1)
                #         pred_logits = student(imgs)
                #         pred_labels = pred_logits.argmax(-1)
                        
                #     kl_loss = F.kl_div(F.log_softmax(pred_logits, 1), F.softmax(gt_logits, 1), reduction='batchmean')
                #     xent_loss = F.cross_entropy(pred_logits, gt_labels)
                #     loss = args.kl_loss_weight * kl_loss + (1-args.kl_loss_weight) * xent_loss 
                    
                #     acc = torch.eq(pred_labels, gt_labels.view_as(pred_labels)).sum().item() / gt_labels.numel()
                #     metrics = {'accuracy': acc, 'kl_loss': kl_loss.item(), 'xent_loss': xent_loss.item()}
                #     student_valmeter.add(metrics)
                                            
                #     pbar(
                #         progress=(step+1)/args.steps_per_epoch, 
                #         desc='[Stdnt val] Epoch {:4d}'.format(epoch), 
                #         status=student_valmeter.msg()
                #     )
                        
                avg = 0
                for name, acc in clswise_acc.items():
                    avg += acc
                avg /= len(clswise_acc)

                if args.log_wandb:
                    wandb.log({'epoch': epoch, 'avg kinetics acc': avg})
                    
                if avg > best_acc:
                    best_acc = avg
                    state = {
                        'epoch': epoch,
                        'student': student.state_dict(),
                        'gan': biggan.state_dict()
                    }
                    torch.save(state, os.path.join(outdir, 'checkpoint_type1.pth.tar'))

    elif args.task == 'kinetics_eval':
        eval_on_kinetics(
            data_root='../new/Video-Swin-Transformer/data/kinetics400/validation',
            student_ckpt_dir='biggan/13-03-2022_18-55'
        )