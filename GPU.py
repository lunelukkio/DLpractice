# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 19:06:54 2024

@author: lunel
"""

import torch
torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

val_cpu = torch.tensor([2, 3, 4])
val_gpu = torch.tensor([2, 3, 4], device='cuda')
print(val_cpu.device)
#print(val_gpu.device)
print(val_gpu)
val_cpu_gpu = val_cpu.to(device)
print(val_cpu_gpu, val_cpu_gpu.device)