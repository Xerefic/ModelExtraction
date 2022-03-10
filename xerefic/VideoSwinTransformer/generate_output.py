from inference import *
from dataloader import *

import pickle

args = TrainingArgs()
train_dataset = CreateDataset(args)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers = 32, drop_last=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

cfg = Config.fromfile(config)
model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg')).to(device)
load_checkpoint(model, checkpoint, map_location=device)
next(model.parameters()).device
model.eval()

predicted = {}

with torch.no_grad():
    bar = pyprind.ProgBar(len(train_dataloader), bar_char='█')
    for i, (img, label, vidid, imgid) in enumerate(train_dataloader):
        torch.cuda.empty_cache()
        img = img.to(device)
        output = model.cls_head(model.backbone(img)).squeeze(0)

        if label in predicted.keys():
            if vidid in predicted[label].keys():
                predicted[label][vidid].append({'Images': imgid, 'Logits': output.cpu()})
            else:
                predicted[label][vidid] = [{'Images': imgid, 'Logits': output.cpu()}]
        else:
            predicted[label] = {vidid: [{'Images': imgid, 'Logits': output.cpu()}]}

        bar.update()

        torch.cuda.empty_cache()

with open('/home/ubuntu/ModelExtraction/xerefic/VideoSwinTransformer/checkpoints/video_swin/logits.pkl', 'wb') as f:
        pickle.dump(predicted, f)

# train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers = 2, drop_last=True)
#     for i, (img, label, vidid) in enumerate(train_dataloader):
#         img = img.to(device)
#         # label = label.to(device)
#         # vidid = vidid.to(device)
#         print(img.shape)
#         # print(label)
#         # print(vidid)
#         quit()