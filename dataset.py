import os.path
from PIL import Image
import random

import h5py
import io

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import numpy as np

import skimage
from skimage import transform
import warnings

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

VOL_EXTENSIONS = [
    '.raw', '.RAW',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_volume_file(filename):
    return any(filename.endswith(extension) for extension in VOL_EXTENSIONS)

def make_dataset(root, list_letter, max_dataset_size=float("inf"), vol=False):
    images = []
    list_path = os.path.join(root, (list_letter + ".list"))
    list_file = open(list_path, "r")
    print(list_path)
    assert os.path.isfile(str(list_path)), '%s is not a valid file' % dir
    lines = list_file.readlines()

    for line in lines:
        line = line.strip('\n')
        if vol:
            if is_volume_file(line):
                path = os.path.join(list_letter, "raw", line)
                images.append(path)
        else:
            if is_image_file(line):
                path = os.path.join(list_letter, "png", line)
                print(path)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

class GananaDataset(Dataset):
    """
    Dataset Structure    
    ├── dataroot/
    │   ├── trainA.list
    │   ├── trainA/
    |   |   ├── xxx.png
    |   |   ├── xxx.raw
    │   ├── trainB.list
    │   ├── trainB/
    |   |   ├── xxx.png
    │   ├── testA/
    |   |   ├── xxx.png
    """

    def __init__(self, dataroot, input_nc=3, output_nc=3, train=True):

        print("Starting Dataset with HDF5")

        self.train = train
        self.dataroot = dataroot
        self.A_paths = make_dataset(dataroot, "trainA")   # load images from '/path/to/data/trainA'
        self.B_paths = make_dataset(dataroot, "trainB")    # load images from '/path/to/data/trainB'
        self.V_paths = make_dataset(dataroot, "trainA", vol=True)
        self.A_paths.sort()
        self.B_paths.sort()
        self.V_paths.sort()
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.V_size = len(self.V_paths)
        print("\nNumber of Synthetic Images:\t" + str(self.A_size))
        print("\nNumber of Real Images:\t" + str(self.B_size))
        print("\nNumber of Volumes:\t" + str(self.V_size) + "\n")

        #self.Y = torch.zeros(3,256,256).float()
        #self.Z = torch.zeros(64,256,256).float()

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.train:
            V_path = self.V_paths[index % self.V_size]
            index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]
        else:
            V_path = self.V_paths[index % self.V_size]
            index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]
            
        hf = h5py.File(os.path.join(self.dataroot, "data.hdf5"), 'r')

        A = torch.tensor(hf[A_path])
        B = torch.tensor(hf[B_path])
        V = torch.tensor(hf[V_path])

        hf.close()

        A = A.float()
        B = B.float()
        V = V.float()
        
        A = A - torch.min(A)
        A = A / torch.max(A)
        B = B - torch.min(B)
        B = B / torch.max(B)
        V = (V > 0).float()
        
        if self.train:
            A, V = self.transform_AV(A, V)
            B = self.transform_img(B)
        else:
            A = self.transform_img(A)
            B = self.transform_img(B)

        if self.train:
            return {'A': A, 'B': B, 'V': V, 'A_paths': A_path, 'B_paths': B_path}
        else:
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size, self.V_size)

    def transform_AV(self, image, volume, grayscale=False, convert=True):
        if self.train:
            if random.random() > 0.5:
                image = torch.flip(image, [1])
                volume = torch.flip(volume, [1])                               

            if random.random() > 0.5:
                image = torch.flip(image, [2])
                volume= torch.flip(volume, [2])                               
                    
            if random.random() > 0.5:
                if random.random() > 0.5:
                    image = torch.rot90(image, 1, [1,2])
                    volume = torch.rot90(volume, 1, [1,2])
                else:
                    image = torch.rot90(image, 3, [1,2])
                    volume = torch.rot90(volume, 3, [1,2])
        return image, volume

    def transform_img(self, image, grayscale=False, convert=True, resize=False, size=(256,256)): 
        #Should probably finish adding the ability to resize
        if self.train:
            print("=============================================================================================")
            if random.random() > 0.5:
                image = torch.flip(image, [1])                             

            if random.random() > 0.5:
                image = torch.flip(image, [2])                              
                
            if random.random() > 0.5:
                if random.random() > 0.5:
                    image = torch.rot90(image, 1, [1,2])
                else:
                    image = torch.rot90(image, 3, [1,2])
        return image


if __name__ == '__main__':
    gd = GananaDataset("./")
    print(len(gd))
