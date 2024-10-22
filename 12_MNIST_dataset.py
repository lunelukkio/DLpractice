# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:42:35 2024

@author: lunel
"""

from torch import utils
from torchvision import datasets
from torchvision import models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


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
        self.fc1 = nn.Linear(784, 512)   
        self.fc2 = nn.Linear(512, 10)
        
        # loss function
        #self.criterion = nn.MSELoss()  # mean square error
        self.criterion = nn.CrossEntropyLoss()  # use softmax()
        # optimize function
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)  # learing rate
        #self.optimizer = optim.SGD(self.parameters(), lr=0.0001)
        #self.optimizer = optim.SGD(self.parameters(), lr=0.0001, momentum=0.9)
        
    def forward(self, x):
        x = self.fc1(x)
        #print('passed fc1\n', x)
        x = F.relu(x)
        #x = F.softmax(x)
        #x = torch.sigmoid(x)
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

# single image test
def classify_single_image(model, test_loader, image_num):
    model.eval()  # evaluation mode

    # load 100 images as images
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    # image number from images for test
    image = images[image_num].reshape(-1, 28*28*1)
    label = labels[image_num]

    # evaluation by deep learning
    with torch.no_grad():  # do not run grad calculatoin for faster culculation
        output = model(image)

    # get estimation
    _, predicted = torch.max(output, axis=1)

    # result
    print(f"Actual Label: {label.item()}, Predicted Label: {predicted.item()}")

    # show data
    plt.imshow(images[image_num].squeeze(), cmap="gray")
    plt.title(f"Predicted: {predicted.item()}, Actual: {label.item()}")
    plt.show()

if __name__ == '__main__':
    
    model = MlpNet()
    acc, loss = train(model, train_loader)
    print(f"accuracy： {acc}, loss: {loss}")
    test_acc = test(model, test_loader)
    print(test_acc)
    
    classify_single_image(model, test_loader, 20)