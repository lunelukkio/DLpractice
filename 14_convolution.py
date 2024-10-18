# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 19:33:46 2024

@author: lunel
"""
from torch import utils
from torchvision import datasets
import torchvision.transforms as transforms

import torch
from torch import nn, optim
from torch.nn import functional as F

conv = nn.Conv2d(3, 5, 3)  # input=3, output=5  kernel size=3x3

x = torch.Tensor(torch.rand(1, 3, 28, 28))
x = conv(x)

print(x.shape)
