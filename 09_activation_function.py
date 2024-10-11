# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 20:27:02 2024

@author: lunel
"""

import torch
from torch import nn
from torch.nn import functional as F

fc = nn.Linear(4, 4)
x = torch.Tensor([-2.0, 1.0, 4, 0])

x = fc(x)

rl = F.relu(x)
soft = F.softmax(x, dim=0)
sig = torch.sigmoid(x)

print(x)
print(rl)
print(soft)
print(sig)


