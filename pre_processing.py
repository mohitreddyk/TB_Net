# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 09:30:12 2018

@author: vatsav & Mohit
"""

import numpy as np
import cv2
import os
from tqdm import tqdm
resizeWidth=1400
resizeHeigth=700


#Csv_ChinaSet_AllFiles = open('formatingData/ChinaSet_AllFiles.csv', 'w')
#Csv_MontgomerySet = open('formatingData/MontgomerySet.csv', 'w')

data_dir1='data/pulmonary-chest-xray-abnormalities/ChinaSet_AllFiles/CXR_png'
data_dir2='data/pulmonary-chest-xray-abnormalities/MontgomerySet/CXR_png'
output_dir1='formatingData/ChinaSet_AllFiles'
output_dir2='formatingData/MontgomerySet'

def labelling(status):
    if status == '0': 
        normal=1
        abnormal=0
        return normal,abnormal
    elif status == '1': 
        normal=0
        abnormal=1
        return normal,abnormal
    
def find_max_rectangle(rectangles):
    max_rectangle = None
    max_area = 0
    for rectangle in rectangles:
        x, y, width, height = rectangle
        area = width * height

        if area > max_area:
            max_area = area
            max_rectangle = rectangle

    return max_rectangle


def detectLeftLung(im):
    cascadePath = "left_lung_haar.xml"
    imageGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    leftLungCascade = cv2.CascadeClassifier(cascadePath)
    leftLungRect = leftLungCascade.detectMultiScale(imageGray, scaleFactor=1.1, minNeighbors=1, minSize=(1,1))
    left_lung_rectangle = find_max_rectangle(leftLungRect)
    return left_lung_rectangle
def detectRightLung(im):
    cascadePath = "right_lung_haar.xml"
    imageGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    rightLungCascade = cv2.CascadeClassifier(cascadePath)
    rightLungRect = rightLungCascade.detectMultiScale(imageGray, scaleFactor=1.1, minNeighbors=1, minSize=(1,1))
    right_lung_rectangle = find_max_rectangle(rightLungRect)
    return right_lung_rectangle
def croppingLungs(im):
    left_lung_rectangle=detectLeftLung(im)
    right_lung_rectangle=detectRightLung(im)
    if right_lung_rectangle is not None:
        x, y, width, height = right_lung_rectangle
        right_image = im[y:y + height, x:x + width]
        right_image=cv2.resize(right_image,(resizeHeigth,resizeWidth))
    if left_lung_rectangle is not None:
        x, y, width, height = left_lung_rectangle
        left_image = im[y:y + height, x:x + width]
        left_image=cv2.resize(left_image,(resizeHeigth,resizeWidth))
    lungs = np.concatenate((right_image,left_image), axis=1)
    return lungs
    
for img in tqdm(os.listdir(data_dir1)):
    newfolder = os.path.splitext(img)[0]
    data=newfolder.split('_')
    
    
    in_im=data_dir1+'/'+img
    out_im=output_dir1+'/'+img
    im = cv2.imread(in_im)
    lungs = croppingLungs(im)
    cv2.imwrite(out_im, lungs)
    
#    normal,abnormal=labelling(data[2])
#    row = str(img) + ',' + str(normal) + ',' + str(abnormal) + '\n'
#    Csv_ChinaSet_AllFiles.write(row)
    
    
    
for img in tqdm(os.listdir(data_dir2)):
    newfolder = os.path.splitext(img)[0]
    data=newfolder.split('_')    
    
    
    in_im=os.path.join(data_dir2,img)
    out_im=os.path.join(output_dir2,img)
    im = cv2.imread(in_im)
    lungs = croppingLungs(im)
    cv2.imwrite(out_im, lungs)
    
#    normal,abnormal=labelling(data[2])
#    row = str(img) + ',' + str(normal) + ',' + str(abnormal) + '\n'
#    Csv_MontgomerySet.write(row)