# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:42:35 2024

@author: lunel
"""

from torch import utils
from torchvision import datasets
import torchvision.transforms as transforms

import torch
from torch import nn, optim
from torch.nn import functional as F

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)


trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=2)


class MlpNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # forward function
        self.fc1 = nn.Linear(784, 256)   
        self.fc2 = nn.Linear(256, 10)
        
        # loss function
        self.criterion = nn.MSELoss()
        # optimize function
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)  # learing rate
        
    def forward(self, x):
        x = self.fc1(x)
        #print('passed fc1\n', x)
        x = F.relu(x)
        #print('passed reru()\n', x)
        x = self.fc2(x)
        
        return x
    
def train(model, train_loader):
    model.train()  # declear now is train mode
    
    ##########
    total_correct = 0
    total_loss = 0
    total_data_len = 0
    ##########
    
    for batch_imgs, batch_labels in train_loader:
        batch_imgs = batch_imgs.reshape(-1, 28*28*1)
        labels = torch.eye(10)[batch_labels]
        
        outputs = model(batch_imgs)
        model.optimizer.zero_grad()  # reset
        loss = model.criterion(outputs, labels)
        loss.backward()
        model.optimizer.step()
        
    ############
        _, pred_labels = torch.max(outputs, axis=1)
        batch_size = len(batch_labels)
        for i in range(batch_size):
            total_data_len += 1
            if pred_labels[i] == batch_labels[i]:
                total_correct += 1
        total_loss += loss.item()
        
    accuracy = total_correct/total_data_len*100
    loss = total_loss/total_data_len
    ############
    
    return accuracy, loss

def test(model, data_loader):
    model.eval()  # declear now is evaluation mode
    
    ##########
    total_correct = 0
    total_data_len = 0
    ##########
    
    for batch_imgs, batch_labels in data_loader:
        outputs = model(batch_imgs.reshape(-1, 28*28*1))
        
    ############
        _, pred_labels = torch.max(outputs, axis=1)
        batch_size = len(batch_labels)
        for i in range(batch_size):
            total_data_len += 1
            if pred_labels[i] == batch_labels[i]:
                total_correct += 1
        
    acc = 100.0 * total_correct/total_data_len
    ############
    
    return acc

if __name__ == '__main__':
    model = MlpNet()
    acc, loss = train(model, train_loader)
    print(f"accuracy： {acc}, loss: {loss}")
    
    test_acc = test(model, test_loader)
    print(test_acc)