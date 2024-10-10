# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 20:27:02 2024

@author: lunel
"""

import torch
from torch import nn
from torch.nn import functional as F

fc = nn.Linear(4, 2)
x = torch.Tensor([1, 2, 3, 4])

x = fc(x)

print(x)
