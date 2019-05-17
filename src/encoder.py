import torch
import torch.nn as nn
import torchvision.models as models

class VGGNetEncoder(nn.Module):
    def __init__(self):
        super(VGGNetEncoder, self).__init__()
        # Loading pretrained VGG16 model
        self.net = models.vgg16(pretrained=True)
        
        # Removing the last max pool and fully connected layer used for classification
        # As the paper mentions the use of the output from a lower convolutional layer
        self.net = nn.Sequential(*list(self.net.features.children())[:-1])

    def forward(self, x):
        feature = self.net(x)
 
        return feature


class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        # Loading a pretrained ResNet model
        resnet = models.resnet152(pretrained=True)
        # Removing the last fc layer as per the paper
        self.extracted_net = nn.Sequential(*(list(resnet.children())[:-2]))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))

    def forward(self, x):
        feature = self.extracted_net(x)
        feature = self.adaptive_pool(feature)
        return feature
