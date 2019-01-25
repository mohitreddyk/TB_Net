# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 18:24:49 2018

@author: vatsa & Mohit
"""

#import numpy as np
import os
import matplotlib.pyplot as plt

TRAIN_DIR = './formatingData/Haar/ChinaSet_AllFiles'
TEST_DIR = './formatingData/Haar/MontgomerySet'
output_dir='./formatingData/Haar/data1'
train_set=os.listdir(TRAIN_DIR)
test_set=os.listdir(TEST_DIR)



i=0
for imgname in train_set:
#    print(imgname)
    newfolder = os.path.splitext(imgname)[0]
    imagesave=plt.imread(os.path.join(TRAIN_DIR , imgname))
    plt.imsave(os.path.join(output_dir , newfolder + '_'+ str(i) + '.png'), imagesave)
    i=i+1
i=0
for imgname in test_set:
    newfolder = os.path.splitext(imgname)[0]
#    print(imgname)
    imagesave=plt.imread(os.path.join(TEST_DIR , imgname))
    plt.imsave(os.path.join(output_dir , newfolder + '_'+ str(i) + '.png'), imagesave)
    i=i+1