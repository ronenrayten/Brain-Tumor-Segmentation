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
import torchvision.transforms.functional as TF


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

# Jupyter
from IPython import get_ipython
from IPython.display import HTML, Image
from IPython.display import display
from ipywidgets import Dropdown, FloatSlider, interact, IntSlider, Layout, SelectionSlider
from ipywidgets import interact

import pytest
import nibabel as nib
import os
from nibabel.testing import data_path
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset
from skimage.util import montage 
from skimage.transform import rotate
from sklearn import preprocessing as pre
from loss import FocalLoss, DiceLoss, CombinedDiceFocalLoss
import torch.optim.lr_scheduler as lr_scheduler
import numpy.ma as ma

from DataVisualization import PlotLabelsHistogram, PlotMnistImages
from DeepLearningPyTorch import TrainModel
from unet3d import UNet3D
from torchmetrics.segmentation import MeanIoU
from torchmetrics.classification import Dice
import gzip


# Constants


IMG_SIZE = (240,240,155)

TENSOR_BOARD_BASE   = 'TB'

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
nEpochs     = 10

# Visualization
numImg = 3

load_dotenv()



def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='float')[y]

def str_to_bool(s):
    return s.lower() in ('true', '1')

print(os.getenv("NUMPY_ARCHIVE_PATH"))

# samples root directory
root_dir = os.getenv("ROOT_DIR") #The directeroy of the original Kaggel archive. 
numpy_archive_path = os.getenv("NUMPY_ARCHIVE_PATH") #the root directory of the processed numpy samples 
image_path = numpy_archive_path +'new_archive_cat/images/'  # path to the processed numpy images 
label_path = numpy_archive_path +'new_archive_cat/labels/'  # path to the processed numpy labeles 

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

# image types
img_types = ['t1','t1ce','t2','flair','seg']


#number of pixels in an image
p_image_size = IMG_SIZE[0]*IMG_SIZE[1]*IMG_SIZE[2]
crop_size = (128,160,128)

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
      t1_norm = t1_norm.reshape(*IMG_SIZE)
      
      t1ce = nib.load(sample_files[1]).get_fdata()
      t1ce_norm = t1ce.reshape(-1,t1ce.shape[-1])
      t1ce_norm = pre.MinMaxScaler().fit_transform(t1ce_norm)
      t1ce_norm = t1ce_norm.reshape(*IMG_SIZE)
      
      t2 = nib.load(sample_files[2]).get_fdata()
      t2_norm = t2.reshape(-1,t2.shape[-1])
      t2_norm = pre.MinMaxScaler().fit_transform(t2_norm)
      t2_norm = t2_norm.reshape(*IMG_SIZE)
      
      flair = nib.load(sample_files[3]).get_fdata()
      flair_norm = flair.reshape(-1,flair.shape[-1])
      flair_norm = pre.MinMaxScaler().fit_transform(flair_norm)
      flair_norm = flair_norm.reshape(*IMG_SIZE)
      
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




# create transform - convert image to tensor

class ToTensor(object):
    """Convert image in sample to Tensors."""

    def __call__(self, image):
         
        tImg = torch.from_numpy(image).permute(3,0,1,2)
        
        return tImg
    
class Normalize3D(object):
    
    def __init__(self,lMean,lStd):
        self.mean = lMean
        self.std = lStd
        
    """Normalize a 3d numpy image by channel"""

    def __call__(self,image):
         
        tImg = (image - self.mean)/self.std
        
        return tImg


###################################################################
#
# The format of the batch returned:
# Image: batch_size x number of images (4) x image_size (240x240x155)
# Label: batch_size x number of images (1) x image_size (240x240x155)
###################################################################

class ImageDatasetFromDisk(Dataset):
    def __init__(self, mX, vY, transform=None, target_transform=None):
        
        self.images = mX
        self.labels = vY
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        with gzip.open(image_path+self.images[idx], 'rb') as f:
        # Load the decompressed data into a NumPy array
         image = np.load(f)
         
        with gzip.open(label_path+self.labels[idx], 'rb') as f:
         label = np.load(f)
         label = np.moveaxis(label, -1, 0)
        
        ###Insert image augmentation here


        ###End of image augmentation

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
class ImageTrainDatasetFromDisk(Dataset):
    def __init__(self, mX, vY, transform=None, target_transform=None):
        
        self.images = mX
        self.labels = vY
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        with gzip.open(image_path+self.images[idx], 'rb') as f:
        # Load the decompressed data into a NumPy array
            image = np.load(f)
            image = np.moveaxis(image, -1, 0)

         
        with gzip.open(label_path+self.labels[idx], 'rb') as f:
            label = np.load(f)
            label = np.moveaxis(label, -1, 0)
        
        ###Insert image augmentation here
        ###Work with torchvision
        if str_to_bool(os.getenv('AUGMENT_DATA')):
            imaget = torch.from_numpy(image)
            labelt = torch.from_numpy(label)

            oElasTrans = TorchVisionTrns.ElasticTransform(sigma=2.)
            state=torch.get_rng_state() #get the random state
            imaget = oElasTrans(imaget)
            torch.set_rng_state(state)  #reset the random state so label is identical to image
            labelt = oElasTrans(labelt)

            # Random horizontal flipping
            if random.random() > 0.5:
                imaget = TF.hflip(imaget)
                labelt = TF.hflip(labelt)

            # Random vertical flipping
            if random.random() > 0.5:
                imaget = TF.vflip(imaget)
                labelt = TF.vflip(labelt)

            image=imaget.numpy()
            label=labelt.numpy()
        ###Back to Numpy

        image = image * np.random.uniform(low=0.8, high = 1.2)

        org_shape = np.shape(image)
        image = image.reshape(-1, image.shape[-1])
        image = pre.MinMaxScaler().fit_transform(image)
        image = image.reshape(*org_shape)
        image = np.moveaxis(image, 0, -1)

        ###End of image augmentation
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def main() :

    if torch.cuda.is_available(): 
        dev = "cuda:0" 
    else: 
        dev = "cpu" 
    device = torch.device(dev)


    mX = os.listdir(image_path)
    vY = os.listdir(label_path)

    numSamplesTest = np.int32(np.round(len(mX) * trainTestPercentage))
    numSamplesTrain = len(mX) - numSamplesTest
    #numSamplesTest = 200
    #numSamplesTrain = 1000

    print("Number of Train samples:",numSamplesTrain)
    print("Number of Test samples:",numSamplesTest)


    #===========================Fill This===========================#
    # 1. Split the data into train and test (Validation) data sets (NumPy arrays).
    # 2. Use stratified split.
    # !! The output should be: `mXTrain`, `mXTest`, `vYTrain`, `vYTest`.

    mXTrain, mXTest, vYTrain, vYTest = train_test_split(mX, vY, test_size = numSamplesTest, train_size = numSamplesTrain, shuffle = True)
    #===============================================================#

    print(f'The training features data shape: {len(mXTrain)}')
    print(f'The training labels data shape: {len(vYTrain)}')
    print(f'The test features data shape: {len(mXTest)}')
    print(f'The test labels data shape: {len(vYTest)}')
    #print(f'The unique values of the labels: {np.unique(vY)}')

    mXTrain.sort()
    vYTrain.sort()
    mXTest.sort()
    vYTest.sort()
    dsTrain = ImageTrainDatasetFromDisk(mXTrain,vYTrain)
    dsTest = ImageDatasetFromDisk(mXTest,vYTest)

    print(f'The training data set data len: {(len(dsTrain))}')
    print(f'The test data set data len: {(len(dsTest))}')

    # Update Transformer

    #===========================Fill This===========================#
    # 1. Define a transformer which normalizes the data.
    oDataTrns = TorchVisionTrns.Compose([  #<! Chaining transformations
    #    Normalize3D(µ, σ),   #<! Normalizes the Data (https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html)
        ToTensor(),      #<! Convert to Tensor (4,128,160,125)
        TorchVisionTrns.ToDtype(torch.float32, scale = True),
        ])

    oLblTrns = TorchVisionTrns.Compose([  #<! Chaining transformations
    #   Normalize3D(µ, σ),   #<! Normalizes the Data (https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html)
        ToTensor(),      #<! Convert to Tensor (4,128,160,125)
        TorchVisionTrns.ToDtype(torch.float32, scale = True),
        ])

    # Update the DS transformer
    dsTrain.transform = oDataTrns
    dsTest.transform = oDataTrns

    # Data Loader

    #put in if __name__ = 'main'
    dlTrain  = torch.utils.data.DataLoader(dsTrain, shuffle = True, batch_size = 1 * batchSize,drop_last=True, num_workers = 4, pin_memory=True,persistent_workers = True)
    dlTest   = torch.utils.data.DataLoader(dsTest, shuffle = False, batch_size = 1 * batchSize,drop_last=True, num_workers = 2, pin_memory = True,persistent_workers = True)


    # Iterate on the Loader
    # The first batch.
    tX, vY = next(iter(dlTrain)) #<! PyTorch Tensors

    print(f'The batch features dimensions: {tX.shape}')
    print(f'The batch labels dimensions: {vY.shape}')

    # Set the Loss & Score

    #===========================Fill This===========================#
    # 1. Define loss function
    # 2. Define score function.
    #hL = nn.CrossEntropyLoss()
    #hL = FocalLoss(alpha=0.5, gamma = 0.7, weight = class_weights, reduction = 'mean')
    hL = DiceLoss(num_classes=4)
    #hS = MulticlassAccuracy(num_classes = 4)
    #hS = MeanIoU(num_classes=4)
    hS = Dice(num_classes=4,ignore_index=0,average="macro")

    hL = hL.to(device) #<! Not required!
    hS = hS.to(device)
    #===============================================================#

# Train the Model

    #===========================Fill This===========================#
    # 1. Build a loop to evaluate all models.
    # 2. Define a TensorBoard Writer per model to keep its score.
    # !! You may use `TrainModel()`.


    oTBWriter = SummaryWriter(log_dir = TENSOR_BOARD_BASE)

    oModel = UNet3D(in_channels=number_of_channels, num_classes=4)

    pyfile=os.path.realpath(__file__)
    
    fullpath = os.path.split(pyfile)[0] + '\\BestModel.pt'
    checkpoint_filename = Path(fullpath)

    if checkpoint_filename.is_file() :
        dCheckpoint = torch.load(checkpoint_filename)
        oModel.load_state_dict(dCheckpoint['Model'])

    oModel = oModel.to(device) #<! Transfer model to device
    oOpt = torch.optim.AdamW(oModel.parameters(), lr = 1e-4, betas = (0.9, 0.99), weight_decay = 1e-3) #<! Define optimizer
    oSch = lr_scheduler.StepLR(oOpt, step_size=10, gamma=0.1)
    oRunModel, lTrainLoss, lTrainScore, lValLoss, lValScore = TrainModel(oModel, dlTrain, dlTest, oOpt, nEpochs, hL, hS, oSch=oSch, oTBWriter=oTBWriter)
    oTBWriter.close()

    #===============================================================#
    #===============================================================#
if __name__ =='__main__':
   main()
