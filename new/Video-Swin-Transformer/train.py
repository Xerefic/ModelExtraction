#from dataloader import CreateDataset
from models.model import Model

import yaml
import torch
import tqdm
import os
import shutil
import pickle
import logging
from mmaction.datasets import build_dataloader, build_dataset
from mmcv import Config
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import dataloader 
import wandb
import mmcv
from mmcv.runner import load_checkpoint
from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model


def train(epoch, train_dataloader, test_dataloader):
    model.train()
    tot = 0
    cor = 0
    l = 0
    dtl_len = len(train_dataloader)
    dtest_len = len(test_dataloader)
    eval_freq = int(dtl_len/4)
    print("Training epoch " + str(epoch+1))
    
    for i, data in enumerate(train_dataloader):
        
        teacher.eval() 
        with torch.no_grad():
           logits = teacher(return_loss=False, **data)
        print(f'Step {i}/{dtl_len}',end='\r')     
        #print(torch.sum(logits == logits2))  
        images = data['imgs']   
        images= images.to(device)    
        # logits = logits.to(device)
        logits = torch.Tensor(logits).to(device)
        # logits = logits.unsqueeze(0)
        b, seqs, c, seq_len, h, w = images.size()
        images = images.view(seqs, b, seq_len, c, h, w)  
        # x_out = torch.empty((0, 400)).to(device)
        x_out = torch.zeros((b, 400)).to(device)
        for j in range(seqs):
            pred = model(images[j])
            x_out = x_out + pred/seqs

        gt_class = torch.argmax(logits, dim = 1)
        pred_class = torch.argmax(x_out, dim = 1)
        # print(x_out.shape)
        # print(gt_class.shape)
        # print(logits.shape)
        loss = (1-lambd)*F.kl_div(F.log_softmax(x_out, dim=1), F.softmax(logits, dim=1), reduction='batchmean') + lambd*F.cross_entropy(x_out, gt_class)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tot+=loss.item()
        cor+=torch.sum(pred_class==gt_class).item()
        l+=images.size(0)
        acc = cor/l
        kl_loss = (1-lambd)*F.kl_div(F.log_softmax(x_out, dim=1), F.softmax(logits, dim=1), reduction='batchmean')
        xent_loss = lambd*F.cross_entropy(x_out, gt_class)
        
        if (i+1)%100==0:
            print(f'Step {i}/{dtl_len}' + "Train loss:" + str(tot/l) + "Train Acc:" + str(cor/l))
            logger.info(f'Step {i}/{dtl_len}' + "Train loss:" + str(tot/l) + "Train Acc:" + str(cor/l))
            metrics = {'accuracy': acc, 'kl_loss': kl_loss.item(), 'xent_loss': xent_loss.item()}
            wandb.log(metrics)
            torch.save(model.state_dict(), os.path.join('./exp04', f'./cpt_{epoch+1}_{i+1}.pth'))
        if (i+1)%eval_freq==0:
            eval_loss, eval_acc = eval(test_dataloader)
            logger.info(f'Step {i}/{dtest_len}' + " Eval loss:" + str(eval_loss) + "Eval Acc:" + str(eval_acc))
            print(f'Step {i}/{dtl_len}' + " Eval loss:" + str(eval_loss) + "Eval Acc:" + str(eval_acc))
            wandb.log({"epoch": epoch+1, "val_loss": eval_loss, "valid_acc": eval_acc}, step = i)

    return tot/l, cor/l

def eval(test_dataloader):
    model.eval()
    tot = 0
    cor = 0
    l = 0
    dtest_len = len(test_dataloader)
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            # images = data['imgs']
            # logits = test_logits[i]
            with torch.no_grad():
                logits = teacher(return_loss=False, **data)
                torch.cuda.empty_cache()
            images = data['imgs']
            print(f'Step {i}/{dtest_len}',end='\r')     
            images= images.to(device)  
            # logits = logits.to(device)  
            logits = torch.Tensor(logits).to(device)
            logits = logits.unsqueeze(0)
            b, seqs, c, seq_len, h, w = images.size()
            images = images.view(seqs, b, seq_len, c, h, w)  
            x_out = torch.empty((0, 400)).to(device)
            x_out = torch.zeros((b, 400)).to(device)
            for j in range(seqs):
                torch.cuda.empty_cache()
                pred = model(images[j])
                torch.cuda.empty_cache()
                x_out = x_out + pred/seqs

            gt_class = torch.argmax(logits, dim = 1)
            pred_class = torch.argmax(x_out, dim = 1)
            
            loss = (1-lambd)*F.kl_div(F.log_softmax(x_out, dim=1), F.softmax(logits, dim=1), reduction='batchmean') + lambd*F.cross_entropy(x_out, gt_class)
    
            tot+=loss.item()
            cor+=torch.sum(gt_class==pred_class).item()
            l+=images.size(0)
            

    return tot/l, cor/l


if __name__ == '__main__':
    wandb.login()

    with open('train_config.yaml', 'r') as yml_file:
        config = yaml.full_load(yml_file)
    print("Config:", config)
    device = config['device']
    lambd = config['train']['lambda']
    out_dir = config['out_dir']
    batch_size = config['train']['batch_size']
    num_workers = config['train']['num_workers']

    #try:

    os.mkdir('./exp04')

    shutil.copy('train_config.yaml', os.path.join(out_dir,'train_config.yaml'))
    logging.basicConfig(filename=os.path.join(out_dir, "log.txt"), format='%(asctime)s %(message)s', filemode='w') 
    logger=logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    train_loader, test_loader = dataloader.get_dataloader()
    model = Model(config)
    model = model.to(device)
    cfg = Config.fromfile('/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py')

    teacher =  build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(teacher, '/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth', map_location='cuda:0')
    
    optimizer = optim.Adam(model.parameters(), lr = config['train']['lr'], )

    # print(summary(model, (32, 3, 224, 224)))
    # quit()
    wandb.init(project='resnet18-videoswin-t')

    KL_loss = nn.KLDivLoss(reduction='batchmean')
    ent_loss = nn.CrossEntropyLoss()
    for epoch in range(config['train']['epochs']):
        train_loss, train_acc = train(epoch, train_loader, test_loader)
        # if epoch+1 % config['train']['eval_freq'] == 0:
        #     eval_loss, eval_acc = eval(test_loader)
            # logger.info("Epoch " + str(epoch+1) + " Eval loss:" + str(eval_loss) + "Eval Acc:" + str(eval_acc))
            # wandb.log({"epoch": epoch+1, "train_loss": train_loss, "train_acc": train_acc, "val_loss": eval_loss, "valid_acc": eval_acc}, step=1)
            # print("Epoch " + str(epoch+1) + " Eval loss:" + str(eval_loss) + "Eval Acc:" + str(eval_acc))
        
        # if epoch+1 % config['train']['save_freq'] == 0:
            # torch.save(model.state_dict(), os.path.join(out_dir, f'./cpt_{epoch+1}.pth'))
