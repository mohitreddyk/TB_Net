# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 17:46:04 2018

@author: vatsav
"""

import os

TRAIN_DIR = './data/data'
TEST_DIR = './data/test_data'
train_set=os.listdir(TRAIN_DIR)
test_set=os.listdir(TEST_DIR)
Csv_train = open('train.csv', 'w')
Csv_test = open('test.csv', 'w')
def label_img(img):
    newfolder = os.path.splitext(img)[0]
    data=newfolder.split('_')
    print(data[0])
    print(data[1])
    print(data[2])
#    animal=0
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if data[2] == '0':
        diag=0
        return diag
    #                             [no cat, very doggo]
    elif data[2] == '1': 
        diag=1
        return diag

for imgname in train_set:
#    print(imgname)
    diag = label_img(imgname)
    row = str(imgname) + ',' + str(diag) + '\n'
    Csv_train.write(row)

for imgname in test_set:
    print(imgname)
    diag = label_img(imgname)
    row = str(imgname) + ',' + str(diag) + '\n'
    Csv_test.write(row)