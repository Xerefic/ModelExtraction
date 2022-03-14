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

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

def train(epoch, train_dataloader, test_dataloader):
    model.train()
    tot = 0
    cor = 0
    l = 0
    dtl_len = len(train_dataloader)
    eval_freq = int(dtl_len/4)
    print("Training epoch " + str(epoch+1))
    for i , (images, logits, _,_,_) in enumerate(train_dataloader):
        print(f'Step {i}/{dtl_len}',end='\r')     
        images, logits = images.to(device), logits.to(device)     
        pred = model(images)  
        gt_class = torch.argmax(logits, dim = 1)
        pred_class = torch.argmax(pred, dim = 1)

        loss = (1-lambd)*KL_loss(F.log_softmax(pred, dim=1), F.softmax(logits, dim=1)) + lambd*ent_loss(pred, gt_class)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tot+=loss.item()
        cor+=torch.sum(pred_class==gt_class).item()
        l+=images.size(0)
        if i%100==0:
            print(f'Step {i}/{dtl_len}' + "Train loss:" + str(tot/l) + "Train Acc:" + str(cor/l))
            logger.info(f'Step {i}/{dtl_len}' + "Train loss:" + str(tot/l) + "Train Acc:" + str(cor/l))
        if i%eval_freq==0:
            eval_loss, eval_acc = eval(test_loader)
            logger.info(f'Step {i}/{dtl_len}' + " Eval loss:" + str(eval_loss) + "Eval Acc:" + str(eval_acc))
            print(f'Step {i}/{dtl_len}' + " Eval loss:" + str(eval_loss) + "Eval Acc:" + str(eval_acc))


    return tot/l, cor/l

def eval(dataloader):
    model.eval()
    tot = 0
    cor = 0
    l = 0
    with torch.no_grad():
        for i , (images, logits, _,_,_) in enumerate(dataloader):
            images, logits = images.to(device), logits.to(device)
        
            pred = model(images)
            gt_class = torch.argmax(logits, dim = 1)
            pred_class = torch.argmax(pred, dim = 1)

            loss = (1-lambd)*KL_loss(F.log_softmax(pred,dim=1), F.softmax(logits, dim=1)) + lambd*ent_loss(pred, gt_class)

            tot+=loss.item()
            cor+=torch.sum(gt_class==pred_class).item()
            l+=images.size(0)
            

    return tot/l, cor/l


if __name__ == '__main__':
    with open('train_config.yaml', 'r') as yml_file:
        config = yaml.full_load(yml_file)
    print("Config:", config)
    device = config['device']
    lambd = config['train']['lambda']
    out_dir = config['out_dir']
    #try:
    os.mkdir(out_dir)
    #except FileExistsError:
    #    pass
    shutil.copy('train_config.yaml', os.path.join(out_dir,'train_config.yaml'))
    logging.basicConfig(filename=os.path.join(out_dir, "log.txt"), format='%(asctime)s %(message)s', filemode='w') 
    logger=logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # dataset = CreateDataset(config)
    # dataset_len = len(dataset)
    # test_len = int(config['train']['split_ratio']*dataset_len)
    # train_len = dataset_len - test_len
    cfg = Config.fromfile('/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py')
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))

    with open('/home/ubuntu/ModelExtraction/new/Video-Swin-Transformer/checkpoints/moments/logits.pkl', 'rb') as f:
        logits = pickle.load(f)
    #print(len(dataset))
    #print(dataset[0]['imgs'].size())

    dataset_len = len(dataset)
    test_len = int(config['train']['split_ratio']*dataset_len)
    train_len = dataset_len - test_len

    # dataloader_setting = dict(
    #     videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
    #     workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
    #     dist=False,
    #     shuffle=False)
    # dataloader_setting = dict(dataloader_setting,
    #                           **cfg.data.get('test_dataloader', {}))
    # data_loader = build_dataloader(dataset, **dataloader_setting)

    train_dataset, test_dataset = random_split(dataset, [train_len, test_len], generator = torch.Generator().manual_seed(42))
    #full_loader = DataLoader(dataset, batch_size = config['train']['batch_size'], shuffle=config['train']['shuffle'],
    #                          pin_memory=config['train']['pin_memory'], num_workers=config['train']['num_workers'])
    train_loader = DataLoader(train_dataset, batch_size = config['train']['batch_size'], shuffle=config['train']['shuffle'],
                              pin_memory=config['train']['pin_memory'], num_workers=config['train']['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size = 16, shuffle=config['train']['shuffle'],
                              pin_memory=config['train']['pin_memory'], num_workers=2)
    
    for i, x in enumerate(train_loader):
        vids = x['imgs']
        print(i)
        print(vids.size())
        quit()

    model = Model(config)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = config['train']['lr'], )

    
    KL_loss = nn.KLDivLoss(reduction='batchmean')
    ent_loss = nn.CrossEntropyLoss()
    for epoch in range(config['train']['epochs']):
        train_loss, train_acc = train(epoch, train_loader, test_loader)
        if epoch+1 % config['train']['eval_freq'] == 0:
            eval_loss, eval_acc = eval(test_loader)
            logger.info("Epoch " + str(epoch+1) + " Eval loss:" + str(eval_loss) + "Eval Acc:" + str(eval_acc))
            print("Epoch " + str(epoch+1) + " Eval loss:" + str(eval_loss) + "Eval Acc:" + str(eval_acc))
        
        if epoch+1 % config['train']['save_freq'] == 0:
            torch.save(model.state_dict(), os.path.join(out_dir, f'./cpt_{epoch+1}.pth'))
