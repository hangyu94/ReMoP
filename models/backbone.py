import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ResNet_18(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNet_18, self).__init__()

        ResNet18 = torchvision.models.resnet18(weights=None)
        
        checkpoint = torch.load('./models/resnet18_msceleb.pth')

        ResNet18.load_state_dict(checkpoint['state_dict'], strict=True)

        self.base = nn.Sequential(*list(ResNet18.children())[:-2])

        self.output = nn.Sequential(nn.Dropout(0.5), Flatten())
        self.classifier = nn.Linear(512, num_classes)


    def forward(self, image):
        feature_map = self.base(image)
        feature_map = F.avg_pool2d(feature_map, feature_map.size()[2:])
        feature_e = self.output(feature_map)
        out = self.classifier(feature_e)

        return out, feature_e
