# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:52:56 2024

@author: lulul
"""

import torch
from torch import nn
from torch.nn import functional as F

from torch import optim
from torchvision.models import vgg11

x = torch.tensor(2.0, requires_grad = True)
a, b = 3, 5
y = a*x + x + b
print(y)
y.backward()
for i in range(4):
    print(x.grad)
    
    
print(torch.eye(10)[[3, 7]])