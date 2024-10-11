# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:31:06 2024

@author: lunel
"""

import torch
from torch import nn
from torch.nn import functional as F

from torch import optim
from torchvision.models import vgg11

criterion = nn.MSELoss()

x = torch.Tensor([0, 1, 2])
y = torch.Tensor([1, -1, 0])

loss = criterion(x, y)

print(loss)

model = vgg11()

optimizer = optim.Adam(model.parameters())
print(optimizer)