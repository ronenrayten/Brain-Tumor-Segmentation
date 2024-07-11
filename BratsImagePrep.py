# Import Packages
# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Machine Learning
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Deep Learning
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchinfo
from torchmetrics.classification import MulticlassAccuracy
from torchvision.transforms import v2 as TorchVisionTrns


# Miscellaneous
import math
import os
from dotenv import load_dotenv

from platform import python_version
from platform import python_version
import random
import time

# Typing
from typing import Callable, Dict, Generator, List, Optional, Self, Set, Tuple, Union

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


import pytest
import nibabel as nib
import os
from nibabel.testing import data_path
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from skimage.util import montage 
from skimage.transform import rotate
from skimage.transform import resize
from scipy.ndimage import zoom

from sklearn import preprocessing as pre
from loss import FocalLoss, DiceLoss, CombinedDiceFocalLoss
import torch.optim.lr_scheduler as lr_scheduler
import numpy.ma as ma

# Courses Packages

from DataVisualization import PlotLabelsHistogram, PlotMnistImages
from DeepLearningPyTorch import TrainModel
from unet3d import UNet3D
from torchmetrics.segmentation import MeanIoU
from torchmetrics.classification import Dice
import gzip



load_dotenv()

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='float')[y]

def str_to_bool(s):
    return s.lower() in ('true', '1')
print(os.getenv("NUMPY_ARCHIVE_PATH"))

# Parameters

# Data
numSamplesTrain = 0 #TBD after we know how many samples we have
numSamplesTest  = 0 #TBD after we know how many samples we have
trainTestPercentage = 0.2
# Model
dropP = 0.2 #<! Dropout Layer

# Training
batchSize   = 1
numWork     = 2 #<! Number of workers
nEpochs     = 5

ORG_IMG_SIZE = (240,240,155)

TENSOR_BOARD_BASE   = 'TB'

# Visualization
numImg = 3

# samples root directory
root_dir = os.getenv("ROOT_DIR") #The directeroy of the original Kaggel archive. 
numpy_archive_path = os.getenv("NUMPY_ARCHIVE_PATH") #the root directory of the processed numpy samples 
image_path = numpy_archive_path +'new_archive_cat/images/'  # path to the processed numpy images 
label_path = numpy_archive_path +'new_archive_cat/labels/'  # path to the processed numpy labeles 


# image types
img_types = ['t1','t1ce','t2','flair','seg']


if torch.cuda.is_available(): 
    dev = "cuda:0" 
else: 
    dev = "cpu" 
device = torch.device(dev)


# samples root directory
root_dir = os.getenv("ROOT_DIR") #The directeroy of the original Kaggel archive. 
numpy_archive_path = os.getenv("NUMPY_ARCHIVE_PATH") #the root directory of the processed numpy samples 
image_path = numpy_archive_path +'new_archive_cat/images/'  # path to the processed numpy images 
label_path = numpy_archive_path +'new_archive_cat/labels/'  # path to the processed numpy labeles 


# image types
img_types = ['t1','t1ce','t2','flair','seg']


#number of pixels in an image
p_image_size = ORG_IMG_SIZE[0]*ORG_IMG_SIZE[1]*ORG_IMG_SIZE[2]
crop_size = (160,160,128)
img_resize = (128,128,128)



#These functions to be run once to generate the numpy files containing the images
def get_file_lists():
   files = os.listdir(root_dir)
   
   file_list = []
   
   for file_name in files:
     if os.path.isdir(root_dir +file_name):
        #Eric - file structure
      #all_samples_dirs = os.listdir(root_dir + file_name +'/')
      #for nii_dir in all_samples_dirs:
        #Eric - file structure
        # if os.path.isdir(root_dir + file_name + '/'+nii_dir):
         nii_files = os.listdir(root_dir + file_name)
         
         local_dir = root_dir + file_name + '/' #+nii_dir+'/' #Eric -file structure
         sample_files=['','','','','']
         for file in nii_files:
            if '_t1.' in file: 
               sample_files[0] = local_dir+file
            elif '_t1ce.' in file: 
               sample_files[1] = local_dir+file
            elif '_t2.' in file: 
               sample_files[2] = local_dir+file
            elif '_flair.' in file: 
               sample_files[3] = local_dir+file
            elif '_seg.' in file: 
               sample_files[4] = local_dir+file
            else:
               print (file)
         file_list.append(sample_files)

   return np.array(file_list)

def create_repository():
    # Directory 
    archive_directory = "new_archive_cat"
    
    # Path 
    root_archive_path = os.path.join(numpy_archive_path, archive_directory)
    if (os.path.exists(root_archive_path) == False) :
      os.mkdir(root_archive_path) 
    
    # Directory 
    image_directory = "images"
   
    # Path 
    image_path = os.path.join(root_archive_path, image_directory) 
    if (os.path.exists(image_path) == False) :
      os.mkdir(image_path) 
    
    # Directory 
    label_directory = "labels"
   
    # Path 
    label_path = os.path.join(root_archive_path, label_directory) 
    if (os.path.exists(label_path) == False) :
      os.mkdir(label_path) 

#test function
def test_combined(sample_file):
      
#1. Normalize the images between 0 to 1
      #t1
      sample_files = [f'{root_dir}/{sample_file}/{sample_file}_t1.nii',
                      f'{root_dir}/{sample_file}/{sample_file}_t1ce.nii',
                      f'{root_dir}/{sample_file}/{sample_file}_t2.nii',
                      f'{root_dir}/{sample_file}/{sample_file}_flair.nii',
                      f'{root_dir}/{sample_file}/{sample_file}_seg.nii']
      t1 = nib.load(sample_files[0]).get_fdata()
      t1_norm = t1.reshape(-1,t1.shape[-1])
      t1_norm = pre.MinMaxScaler().fit_transform(t1_norm)
      t1_norm = t1_norm.reshape(*ORG_IMG_SIZE)
      
      t1ce = nib.load(sample_files[1]).get_fdata()
      t1ce_norm = t1ce.reshape(-1,t1ce.shape[-1])
      t1ce_norm = pre.MinMaxScaler().fit_transform(t1ce_norm)
      t1ce_norm = t1ce_norm.reshape(*ORG_IMG_SIZE)
      
      t2 = nib.load(sample_files[2]).get_fdata()
      t2_norm = t2.reshape(-1,t2.shape[-1])
      t2_norm = pre.MinMaxScaler().fit_transform(t2_norm)
      t2_norm = t2_norm.reshape(*ORG_IMG_SIZE)
      
      flair = nib.load(sample_files[3]).get_fdata()
      flair_norm = flair.reshape(-1,flair.shape[-1])
      flair_norm = pre.MinMaxScaler().fit_transform(flair_norm)
      flair_norm = flair_norm.reshape(*ORG_IMG_SIZE)
      
      #2. Get the mask file and change its labels to 0-3 
      seg = nib.load(sample_files[4]).get_fdata()
      seg = seg.astype(np.uint8)
      seg[seg==4] = 3
      
      # 3. combine the images into one with x channels
      combined_image = np.stack([t1_norm,t1ce_norm,t2_norm,flair_norm],axis=3)
      
      # 4. Crop the images to 125x125x125x4
      combined_image= combined_image[56:184, 43:203, 13:141]
      
      #5. crop the mask
      seg = seg[56:184, 43:203, 13:141]

      return combined_image,seg
   
      
def calc_mean_std(number_of_channels):
   file_list = os.listdir(image_path)
   avg_list= np.zeros(len(file_list)*4)
   avg_list = avg_list.reshape(len(file_list),4)
   std_list= np.zeros(len(file_list)*4)
   std_list = std_list.reshape(len(file_list),4)
   i=0
   for sample in file_list:
    image = np.load(image_path+sample)
    if(image.shape != (128,160,128,number_of_channels)):
       print(sample)
    avg_list[i] = np.mean(image,axis=(0,1,2))
    std_list[i] = np.std(image,axis=(0,1,2))
    i+=1
   
   return np.mean((avg_list),axis=0),np.mean((std_list),axis=0)


def CropandZoom(threedimage):
   threedimage = threedimage[np.floor((ORG_IMG_SIZE[0]-crop_size[0])/2).astype(int):
                                    np.floor((crop_size[0] - ORG_IMG_SIZE[0])/2).astype(int), 
                                    np.floor((ORG_IMG_SIZE[1]-crop_size[1])/2).astype(int):
                                    np.floor((crop_size[1] - ORG_IMG_SIZE[1])/2).astype(int),
                                    np.floor((ORG_IMG_SIZE[2]-crop_size[2])/2).astype(int):
                                    np.floor((crop_size[2] - ORG_IMG_SIZE[2])/2).astype(int)]
   
   threedimage = zoom(threedimage, (img_resize[0]/crop_size[0],img_resize[1]/crop_size[1],img_resize[2]/crop_size[2]))

   return threedimage




# Set the environment variables for enabling/disabling each channel
ENABLE_T1 = str_to_bool(os.getenv('ENABLE_T1'))
ENABLE_T1CE = str_to_bool(os.getenv('ENABLE_T1CE'))
ENABLE_T2 = str_to_bool(os.getenv('ENABLE_T2'))
ENABLE_FLAIR = str_to_bool(os.getenv('ENABLE_FLAIR'))
ENABLE_T1COMB = str_to_bool(os.getenv('ENABLE_T1COMB'))
ENABLE_T2COMB = str_to_bool(os.getenv('ENABLE_T2COMB'))
number_of_channels = sum([ENABLE_T1, ENABLE_T1CE, ENABLE_T2, ENABLE_FLAIR, ENABLE_T1COMB, ENABLE_T2COMB])
# print(f"ENABLE_T1: {ENABLE_T1} | ENABLE_T1CE: {ENABLE_T1CE} | ENABLE_T2: {ENABLE_T2} | ENABLE_FLAIR: {ENABLE_FLAIR}) 
# | ENABLE_T1COMB: {ENABLE_T1COMB} | ENABLE_T2COMB : {ENABLE_T2COMB})

IMG_SIZE = (240, 240, 155)  # Example size, replace with actual size

def generate_images(file_list):
    sample_number = 0
    start_time = time.time()
    for sample_files in file_list:
        combined_images_list = []

        if ENABLE_T1:
            t1 = nib.load(sample_files[0]).get_fdata()
            t1_norm = t1.reshape(-1, t1.shape[-1])
            t1_norm = pre.MinMaxScaler().fit_transform(t1_norm)
            t1_norm = t1_norm.reshape(*ORG_IMG_SIZE)
            t1_norm = CropandZoom(t1_norm)
            combined_images_list.append(t1_norm)

        if ENABLE_T1CE:
            t1ce = nib.load(sample_files[1]).get_fdata()
            t1ce_norm = t1ce.reshape(-1, t1ce.shape[-1])
            t1ce_norm = pre.MinMaxScaler().fit_transform(t1ce_norm)
            t1ce_norm = t1ce_norm.reshape(*ORG_IMG_SIZE)
            t1ce_norm = CropandZoom(t1ce_norm)
            combined_images_list.append(t1ce_norm)

        if ENABLE_T2:
            t2 = nib.load(sample_files[2]).get_fdata()
            t2_norm = t2.reshape(-1, t2.shape[-1])
            t2_norm = pre.MinMaxScaler().fit_transform(t2_norm)
            t2_norm = t2_norm.reshape(*ORG_IMG_SIZE)
            t2_norm = CropandZoom(t2_norm)
            combined_images_list.append(t2_norm)

        if ENABLE_FLAIR:
            flair = nib.load(sample_files[3]).get_fdata()
            flair_norm = flair.reshape(-1, flair.shape[-1])
            flair_norm = pre.MinMaxScaler().fit_transform(flair_norm)
            flair_norm = flair_norm.reshape(*ORG_IMG_SIZE)
            flair_norm = CropandZoom(flair_norm)
            combined_images_list.append(flair_norm)

        if ENABLE_T1COMB:
            t1 = nib.load(sample_files[0]).get_fdata()
            t1 = t1.reshape(-1, t1.shape[-1])
            t1 = pre.MinMaxScaler().fit_transform(t1)

            t1ce = nib.load(sample_files[1]).get_fdata()
            t1ce = t1ce.reshape(-1, t1ce.shape[-1])
            t1ce = pre.MinMaxScaler().fit_transform(t1ce)

            t1comb = t1 - t1ce
            t1comb[t1comb<0] = 0
            t1comb = t1comb.reshape(*ORG_IMG_SIZE)
            t1comb = CropandZoom(t1comb)
            combined_images_list.append(t1comb)

        if ENABLE_T2COMB:
            t2 = nib.load(sample_files[2]).get_fdata()
            t2 = t2.reshape(-1, t2.shape[-1])
            t2 = pre.MinMaxScaler().fit_transform(t2)

            t2flair = nib.load(sample_files[3]).get_fdata()
            t2flair = t2flair.reshape(-1, t2flair.shape[-1])
            t2flair = pre.MinMaxScaler().fit_transform(t2flair)

            t2comb = t2flair - t2
            t2comb[t2comb<0] = 0
            t2comb = t2comb.reshape(*ORG_IMG_SIZE)
            t2comb = CropandZoom(t2comb)
            combined_images_list.append(t2comb)

        # Combine the images into one with multiple channels
        combined_image = np.stack(combined_images_list, axis=3)

        # Get the mask file and change its labels to 0-3 
        seg = nib.load(sample_files[4]).get_fdata()
        seg = seg.astype(np.uint8)
        
        # Crop the mask
        seg = seg[np.floor((ORG_IMG_SIZE[0]-crop_size[0])/2).astype(int):
                    np.floor((crop_size[0] - ORG_IMG_SIZE[0])/2).astype(int), 
                    np.floor((ORG_IMG_SIZE[1]-crop_size[1])/2).astype(int):
                    np.floor((crop_size[1] - ORG_IMG_SIZE[1])/2).astype(int),
                    np.floor((ORG_IMG_SIZE[2]-crop_size[2])/2).astype(int):
                    np.floor((crop_size[2] - ORG_IMG_SIZE[2])/2).astype(int)]

        seg = zoom(seg, (img_resize[0]/crop_size[0],img_resize[1]/crop_size[1],img_resize[2]/crop_size[2]),order = 0, mode='nearest')
        seg = seg.astype(np.uint8)
        seg[seg == 4] = 3
        
        seg = to_categorical(seg, 4)

        # Save the files
        with gzip.open(image_path + 'sample_' + str(sample_number) + '.npy.gz', 'wb') as f:
          np.save(f, combined_image)
        with gzip.open(label_path + 'label_' + str(sample_number) + '.npy.gz', 'wb') as f:
          np.save(f, seg)
        print(f'Sample {sample_number + 1}/{len(file_list)} saved')
        print(f'Sample {sample_number + 1}/{len(file_list)} saved')
        sample_number += 1
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Elapsed time: {elapsed_time/60} minutes')


#create_repository and generate images
def main() :
   
    if str_to_bool(os.getenv('CREATE_REPOSITORY')):
        create_repository()
        files = get_file_lists()
        generate_images(files)


if __name__ =='__main__':
   main()