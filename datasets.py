from torch.utils.data import Dataset
import gzip
import numpy as np
import os

numpy_archive_path = os.getenv("NUMPY_ARCHIVE_PATH") #the root directory of the processed numpy samples 
image_path = numpy_archive_path +'new_archive_cat/images/'  # path to the processed numpy images 
label_path = numpy_archive_path +'new_archive_cat/labels/'  # path to the processed numpy labeles 

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
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label