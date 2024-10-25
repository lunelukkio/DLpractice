

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:42:35 2024

@author: lunel
"""

from torch import utils
from torchvision import datasets
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
        self.criterion = nn.MSELoss()  # mean square error
        # optimize function
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)  # learing rate

        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
def train(model, train_loader):
    model.train()  # declear now is train mode
    
    
    for batch_imgs, batch_labels in train_loader:
        batch_imgs = batch_imgs.reshape(-1, 28*28*1)
        labels = torch.eye(10)[batch_labels]
        
        outputs = model(batch_imgs)
        model.optimizer.zero_grad()  # reset
        loss = model.criterion(outputs, labels)
        loss.backward()
        model.optimizer.step()



def test(model, data_loader):
    model.eval()  # declear now is evaluation mode
    
    for batch_imgs, batch_labels in data_loader:
        outputs = model(batch_imgs.reshape(-1, 28*28*1))
        

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
    
    print(output)

    # show data
    plt.imshow(images[image_num].squeeze(), cmap="gray")
    plt.title(f"Predicted: {predicted.item()}, Actual: {label.item()}")
    plt.show()

if __name__ == '__main__':
    
    model = MlpNet()
    train(model, train_loader)
    test(model, test_loader)

    classify_single_image(model, test_loader, 5)