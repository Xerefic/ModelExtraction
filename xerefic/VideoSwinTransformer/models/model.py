import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

def create_backbone(config):
    backbone_name = config['model']['backbone']['name']
    module = importlib.import_module(f'models.{backbone_name}')
    return module.make_model(config)

class Model(nn.Module):
    def __init__(self, config):
        self.agg = config['model']['agg']
        super().__init__()
        self.feature_extractor = create_backbone(config)
        n_channels = self.feature_extractor.c_features
        h = self.feature_extractor.h_features
        w = self.feature_extractor.w_features

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(n_channels, config['model']['classes'])
        if self.agg == 'weighted':
            self.weights = nn.Linear(n_channels, n_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is a tensor containing a number of tensors. 

        b, f, c, h,w = x.size()
        x = x.view(b*f,c,h,w)
        x = self.feature_extractor(x)
        _, c, h,w = x.size()

        if self.agg == 'weighted':
            x = x.view(b,f,h,w,c)
            wt = self.weights(x).view(b,f,c,h,w)
            wt = F.softmax(wt, dim=1)

            x = x.view(b,f,c,h,w)
            x = torch.mul(wt, x)
            x = torch.sum(x,dim=1)

        elif self.agg == 'maxpool':
            x = x.view(b,f,c,h,w)
            x = torch.amax(x, dim = 1)

        else:
            raise ValueError('Aggregate method not implemented')
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

    def enable_graybox_mode(self):
        self.feature_extractor.enable_graybox_mode()
    
    def enable_graybox_mode(self):
        self.feature_extractor.disable_graybox_mode()

if __name__ == "__main__":
    config = yaml.load(open('../config.yaml'), Loader=yaml.FullLoader)
    model = Model(config)
    inp = torch.rand([2,4,3,224,224])
    op = model(inp)
    print(op.size())
    #print(model)
    for name, param in model.named_parameters():
        break
        if 'layer1' in name:
            print(param)
