# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:53:36 2024

@author: lunel
"""

import torch
from torchvision import datasets, transforms

# データ変換: テンソルに変換
transform = transforms.ToTensor()

# トレーニングデータセットのダウンロード
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# テストデータセットのダウンロード
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# データローダーの作成
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# データの例を表示
examples = iter(train_loader)
example_data, example_targets = examples.next()

print(f'Training data shape: {example_data.shape}')
