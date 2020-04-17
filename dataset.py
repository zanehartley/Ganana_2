import os.path
from PIL import Image
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import numpy as np

import skimage
from skimage import io, transform
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

def make_dataset(dir, max_dataset_size=float("inf"), vol=False):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in os.walk(dir):
        for fname in fnames:
            if vol:
                if is_volume_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
            else:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    return images[:min(max_dataset_size, len(images))]

class GananaDataset(Dataset):
    """
    Dataset Structure    
    ├── dataroot/
    │   ├── trainA/
    |   |   ├── xxx.png
    |   |   ├── xxx.raw
    │   ├── trainB/
    |   |   ├── xxx.png
    │   ├── testA/
    |   |   ├── xxx.png
    """

    def __init__(self, dataroot, input_nc=3, output_nc=3, train=True):
        self.dir_A = os.path.join(dataroot, 'trainA')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(dataroot, 'trainB')  # create a path '/path/to/data/trainB'
        print(self.dir_A)
        print(self.dir_B)

        self.train = train

        self.A_paths = sorted(make_dataset(self.dir_A))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B))    # load images from '/path/to/data/trainB'
        self.V_paths = sorted(make_dataset(self.dir_A, vol=True))
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.V_size = len(self.V_paths)
        print("\nNumber of Synthetic Images:\t" + str(self.A_size))
        print("\nNumber of Real Images:\t" + str(self.B_size))
        print("\nNumber of Volumes:\t" + str(self.V_size) + "\n")

        #input_nc = channels     # get the number of channels of input image
        #output_nc = channels      # get the number of channels of output image
        self.transform_A = self.get_transform(grayscale=(input_nc == 1))
        self.transform_B = self.get_transform(grayscale=(output_nc == 1))

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        V_path = self.V_paths[index % self.V_size]
        if self.train:
            index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]
        else:
            B_path = self.B_paths[index % self.B_size]
        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')
        # apply image transformation
        if self.transform_A:
            A = self.transform_A(A)
        if self.transform_B:
            B = self.transform_B(B)

        V = np.memmap(V_path, dtype='uint8', mode='r').__array__()
        V = V.reshape(128, 256, 256)
        V = np.rot90(V, axes=(2,1))
        V= np.flip(V, 2)
        
        #V = V.reshape(256, 256, 128)
        #V = np.moveaxis(V, [0, 1, 2], [2, 0, 1]) #[-2, -1, -3])
        V = V / 255.0

        return {'A': A, 'B': B, 'V': V, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size, self.V_size)

    def get_transform(self, grayscale, convert=True):

        transform_list = []
        if grayscale:
            transform_list.append(transforms.Grayscale(1))
        if convert:
            transform_list += [transforms.ToTensor()]
            if grayscale:
                transform_list += [transforms.Normalize((0.5,), (0.5,))]
            else:
                transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)


