# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 15:16:01 2018

@author: vatsav & Mohit
"""
import torch
import numpy as np         # dealing with arrays
import os                  # dealing with directories
#from skimage import io
from torch.utils.data.dataset import Dataset
import pandas as pd
from PIL import Image

class LungDataset(Dataset):
    def __init__(self,csv_file,root_dir, transform = None):
        self.root_dir= root_dir
        self.transform = transform
        self.data_info = pd.read_csv(csv_file)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        self.label = np.asarray(self.data_info.iloc[:, 1])
        self.data_len = len(self.data_info.index)
        
        
    def __len__(self):
        return self.data_len
  
    def __getitem__(self,idx):
        img_name = os.path.join(self.root_dir,
                                self.image_arr[idx])
        image = Image.open(img_name).convert('RGB')
        img_label = torch.from_numpy(np.asarray(self.label[idx], dtype='int32'))

        if self.transform:
            image = self.transform(image)

        return (image, img_label)