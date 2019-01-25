# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 15:02:55 2018

@author: vatsav & Mohit
"""

import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import pandas as pd
from skimage import io
import torch
from torchvision import utils, models
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
#from torch.utils.data import Dataset, DataLoader
from LungDataset import LungDataset
from torch.autograd import Variable
from utils import Net,Net1,NewNet
from torch import nn


TRAIN_DIR = './data/data'
TEST_DIR = './data/test_data'
csv_file_train= 'train.csv' #with salient and normal.
csv_file_test= 'test.csv' #with salient and normal.
#TRAIN_DIR = './data/data_fine'
#TEST_DIR = './data/test_data_fine'
#csv_file_train= 'train_finetune.csv' #with salient and normal.
#csv_file_test= 'test_finetune.csv' #with salient and normal.
IMG_SIZE = 60
LR = 1e-3
#trainsize=800
#validsize=10

trainsize=790
validsize=10 #with salient and normal.

epochs = 10
batchSize=2
#MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic') # just so we remember which saved model is which, sizes must match
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)


transform = transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

DogCat_dataset_train = LungDataset(csv_file_train, TRAIN_DIR, transform)
DogCat_dataset_test = LungDataset(csv_file_test, TEST_DIR, transform)

indices=torch.randperm(len(DogCat_dataset_train))
trainindices=indices[:len(indices)-validsize][:trainsize or None]
validindices=indices[len(indices)-validsize:] if validsize else None

trainloader = torch.utils.data.DataLoader(DogCat_dataset_train, batch_size=batchSize, sampler=torch.utils.data.SubsetRandomSampler(trainindices), num_workers=0)

if validsize:
    validloader=torch.utils.data.DataLoader(DogCat_dataset_train, batch_size=batchSize, sampler=torch.utils.data.SubsetRandomSampler(validindices), num_workers=0)
else:
    validloader= None

testloader = torch.utils.data.DataLoader(DogCat_dataset_test, batch_size=batchSize, shuffle=False, num_workers=0)

classes=('normal','TB')


net = Net() #network1 calling, comment this line to test other network.
#net = Net1() #network2 calling, un-comment this line to test network2.
#net = NewNet() #network3 calling, un-comment this line to test network3.
net.to(device)

#loss and optimiser defining
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LR) # changes in learing rate can be made here.

#validation function definition 
def val(epoch):
    net.eval()
    val_loss=0.0
    val_total = 0
    val_correct = 0
    val_acc = 0
    for i,(inputs, labels)  in enumerate(validloader,0):
        # get the validation inputs
        inputs = Variable(inputs)
        labels = Variable(labels).long()
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        val_total += labels.size(0)
        val_correct += (predicted == labels).sum().item()
        
        
        # print statistics
        val_loss += loss.item()
    val_acc = 100 * val_correct / val_total
    print('Accuracy of the network on the 1000 validation images: %d %%' % ( 100 * val_correct / val_total))
    return val_loss, val_acc
    


loss_append=[]
val_append=[]
val_acc_append=[]
train_acc_append=[]
#training
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    train_total = 0
    train_correct = 0
    train_acc = 0
    for i, (inputs, labels) in tqdm(enumerate(trainloader, 0)):
        # get the training inputs
#        print (type(inputs))
#        print (type(labels))
        inputs = Variable(inputs)
        labels = Variable(labels).long()
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        # print statistics
        if i == 389:    # print every 37 batch
            print('[epoch: %d,batch size: %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss/2))
            loss_append.append(running_loss/2)
            running_loss = 0.0
    train_acc = 100 * train_correct / train_total
    train_acc_append.append(train_acc)
    val_loss, val_acc=val(epoch)
    val_append.append(val_loss)
    val_acc_append.append(val_acc)
#    if val_acc>90:
#        break

print('Finished Training')

#saving model
torch.save(net,'checkpoint/modelbest1.pth')

#plotting accuracy and loss graphs for validation set per epoch. 
a=[]
for i in range(epochs):
    a.append(i+1)   
plt.plot(a,val_append)
plt.xlabel('epochs')
plt.ylabel('Validation Loss')
plt.show()

plt.plot(a,val_acc_append)
plt.xlabel('epochs')
plt.ylabel('Validation Accuracy')
plt.show()

#plotting accuracy and loss graphs for training set per epoch. 
plt.plot(a,loss_append)
plt.xlabel('epochs')
plt.ylabel('Training Loss')
plt.show()

plt.plot(a,train_acc_append)
plt.xlabel('epochs')
plt.ylabel('Training Accuracy')
plt.show()

net.eval()
#testing
correct = 0
total = 0
with torch.no_grad():
    for (inputs, labels) in testloader:
        inputs = Variable(inputs)
        labels = Variable(labels).long()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % ( 100 * correct / total))