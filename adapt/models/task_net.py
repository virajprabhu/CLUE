# -*- coding: utf-8 -*-
"""
Implements individual task networks
Adapted from https://github.com/jhoffman/cycada_release
"""

import sys
import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import torch.nn.functional as F
from torchvision import models

from .models import register_model
import numpy as np

sys.path.append('../../')
import utils

np.random.seed(1234)
torch.manual_seed(1234)

class TaskNet(nn.Module):

    num_channels = 3
    image_size = 32
    name = 'TaskNet'

    "Basic class which does classification."
    def __init__(self, num_cls=10, weights_init=None):
        super(TaskNet, self).__init__()
        self.num_cls = num_cls        
        self.setup_net()        
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, with_ft=False, with_emb=False, reverse_grad=False):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = x.clone()
        emb = self.fc_params(x)

        if isinstance(self.classifier, nn.Sequential): # LeNet
            emb = self.classifier[:-1](emb)
            if reverse_grad: emb = utils.ReverseLayerF.apply(emb)
            score = self.classifier[-1](emb)
        else:                                          # ResNet
            if reverse_grad: emb = utils.ReverseLayerF.apply(emb)
            score = self.classifier(emb)   
    
        if with_emb:
            return score, emb
        else:
            return score

    def setup_net(self):
        """Method to be implemented in each class."""
        pass

    def load(self, init_path):
        net_init_dict = torch.load(init_path)
        self.load_state_dict(net_init_dict, strict=False)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)

@register_model('LeNet')
class LeNet(TaskNet):
    "Network used for MNIST or USPS experiments."

    num_channels = 1
    image_size = 28
    name = 'LeNet'
    out_dim = 500 # dim of last feature layer

    def setup_net(self):

        self.conv_params = nn.Sequential(
                nn.Conv2d(self.num_channels, 20, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.Dropout2d(p=0.5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                )

        self.fc_params = nn.Linear(50*4*4, 500)
        self.classifier = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(500, self.num_cls)
                )

@register_model('ResNet34')
class ResNet34(TaskNet):
    num_channels = 3
    name = 'ResNet34'

    def setup_net(self):
        model = models.resnet34(pretrained=True)
        model.fc = nn.Identity()
        self.conv_params = model
        self.fc_params = nn.Identity()
        
        self.classifier = nn.Linear(512, self.num_cls)
        init.xavier_normal_(self.classifier.weight)
        self.classifier.bias.data.zero_()