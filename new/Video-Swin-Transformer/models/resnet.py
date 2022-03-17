import torch
import torchvision
import yaml

class Resnet(torchvision.models.ResNet):
    def __init__(self, config):
        block_name = config['model']['backbone']['resnet_block']
        if block_name == 'basic':
            block = torchvision.models.resnet.BasicBlock
        elif block_name == 'bottleneck':
            block = torchvision.models.resnet.Bottleneck
        else:
            raise ValueError
        layers = config['model']['backbone']['resnet_layers'] + [1]
        super().__init__(block, layers)

        pretrained_name = config['model']['backbone']['pretrained']
        if pretrained_name:
            state_dict = torch.hub.load_state_dict_from_url(
                torchvision.models.resnet.model_urls[pretrained_name])
            self.load_state_dict(state_dict, strict=False)

            #Model trained on RGB, if images are BGR uncomment below
            # module = self.conv1
            # module.weight.data = module.weight.data[:, [2, 1, 0]]
        
        del self.avgpool
        del self.fc

        with torch.no_grad():
            size = config['transform']['input_size']
            data = torch.zeros((1, 3, size, size), dtype=torch.float32)
            features = self.forward(data)
            self.c_features = features.size(1)
            self.h_features = features.size(2)
            self.w_features = features.size(3)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.shape)
        # quit()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def enable_graybox_mode(self):
        for name, param in self.named_parameters():
            if 'layer1' in name or 'layer2' in name or 'layer3' in name:
                param.requires_grad = False
    
    def disable_graybox_mode(self):
        for param in self.parameters():
            param.requires_grad = True

def make_model(config):
    return Resnet(config)
    
if __name__=='__main__':   
    config = yaml.load(open('../config.yaml'), Loader=yaml.FullLoader)
    model = Resnet(config)
    for name, param in model.named_parameters():
        if 'layer1' in name:
            print(param)