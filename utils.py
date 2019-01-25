# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 22:02:16 2018

@author: vatsav & Mohit
"""
from torch import nn
import torch.nn.functional as F
#neural network architectures.

#network 1 architecture using torch.nn.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=5,stride=1,padding=2),
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.MaxPool2d(kernel_size=2, stride=2))#convolution layer 1
        self.conv2=nn.Sequential(
                        nn.Conv2d(32, 64, kernel_size=3,stride=1,padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.MaxPool2d(kernel_size=2, stride=2))#convolution layer 2
        self.fc1= nn.Sequential(
                        nn.Linear(64 * 8 * 8, 1024),
                        nn.ReLU())# fully connected layer 1
        self.fc2= nn.Sequential(
                        nn.Linear(1024, 2))# fully connected layer 2

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1,64*8*8)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
#network 2 architecture using torch.nn.functional.
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.convn1 = nn.Conv2d(3, 32, kernel_size = 5, stride = 1, padding = 2)#convolution layer 1
        self.pooling = nn.MaxPool2d(2,2)
        self.convn2 = nn.Conv2d(32,64, kernel_size = 3, stride=1,padding=1)#convolution layer 2
        self.fc1 = nn.Linear(64*8*8,1024)# fully connected layer 1
        self.fc2 = nn.Linear(1024,2)# fully connected layer 2
        #self.relu = nn.ReLU(inplace=True)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        
    def forward(self,x):
        x = self.batchnorm1(F.relu(self.convn1(x)))
        x = self.pooling(x)
        x = self.batchnorm2(F.relu(self.convn2(x)))
        x = self.pooling(x)
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#network 3 architecture.
#newly developed network.
class NewNet(nn.Module):
    def __init__(self):
        super(NewNet, self).__init__()
        self.conv1=nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=3,stride=1,padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(32))#convolution layer 1
        self.conv2=nn.Sequential(
                        nn.Conv2d(32, 64, kernel_size=3,stride=1,padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.MaxPool2d(kernel_size=2, stride=2))#convolution layer 2
        self.conv3=nn.Sequential(
                        nn.Conv2d(64, 128, kernel_size=3,stride=1,padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.MaxPool2d(kernel_size=2, stride=2))#convolution layer 3
        self.conv4=nn.Sequential(
                        nn.Conv2d(128, 256, kernel_size=3,stride=1,padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(256),
                        nn.MaxPool2d(kernel_size=2, stride=2))#convolution layer 4
        self.fc1= nn.Sequential(
                        nn.Linear(256 * 4 * 4, 1024),
                        nn.ReLU())# fully connected layer 1
        self.fc2= nn.Sequential(
                        nn.Linear(1024, 2))# fully connected layer 2

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1,256*4*4)
        x = self.fc1(x)
        x = self.fc2(x)
        return x