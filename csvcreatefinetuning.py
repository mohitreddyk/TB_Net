# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 17:46:04 2018

@author: vatsav
"""

import os



TRAIN_DIR = './formatingData/Haar/ChinaSet_AllFiles'
TEST_DIR = './formatingData/Haar/MontgomerySet'
train_set=os.listdir(TRAIN_DIR)
test_set=os.listdir(TEST_DIR)
Csv_train = open('finetuning_train_Haarmod.csv', 'w')
#Csv_test = open('finetuning_MontgomerySet_Haar.csv', 'w')
def label_img(img,i):
    newfolder = os.path.splitext(img)[0]
    data=newfolder.split('_')
    name = data[0]+'_'+data[1]+'_'+data[2]+'_'+str(i)+'.png'
#    print(data[0])
#    print(data[1])
#    print(data[2])
#    animal=0
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if data[2] == '0':
        diag=0
        return diag,name
    #                             [no cat, very doggo]
    elif data[2] == '1': 
        diag=1
        return diag,name
i=0
for imgname in train_set:
#    print(imgname)
    diag,name = label_img(imgname,i)
    row = str(name) + ',' + str(diag) + '\n'
    Csv_train.write(row)
    i=i+1
i=0
for imgname in test_set:
#    print(imgname)
    diag,name = label_img(imgname,i)
    row = str(name) + ',' + str(diag) + '\n'
    Csv_train.write(row)
    i=i+1