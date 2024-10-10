# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 11:56:50 2024

@author: lulul
"""

import torch
import numpy as np


v = torch.tensor(1.0, requires_grad = True)
w = torch.tensor(1.0, requires_grad = True)

out = 4*v + 6*w + 1

out.backward()
print(v.grad)
print(w.grad)