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


        data = []  
        group = []

        def func(name, obj):     # function to recursively store all the keys
            if isinstance(obj, h5py.Dataset):
                data.append(name)
            elif isinstance(obj, h5py.Group):
                group.append(name)

        #self.hf = h5py.File(os.path.join(dataroot, "data.hdf5"), 'r')
        #self.hf.visititems(func)  # this is the operation we are talking about.

        #input_nc = channels     # get the number of channels of input image
        #output_nc = channels      # get the number of channels of output image
        self.transform_A = self.get_transform_A(grayscale=(input_nc == 1))
        self.transform_B = self.get_transform_B(grayscale=(output_nc == 1))

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.train:
            V_path = self.V_paths[index % self.V_size]
            index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]
        else:
            B_path = self.B_paths[index % self.B_size]
            
        hf = h5py.File(os.path.join(self.dataroot, "data.hdf5"), 'r')

        A_hdf5 = np.array(hf[A_path])                
        B_hdf5 = np.array(hf[B_path])
        if self.train:
            V_hdf5 = np.array(hf[V_path])

        A = Image.open(io.BytesIO(A_hdf5)).convert('RGB')
        B = Image.open(io.BytesIO(B_hdf5)).convert('RGB')
        # apply image transformation

        if self.train:
            V = V_hdf5
            if self.transform_A:
                A, V = self.transform(A, V)
        else:
            if self.transform_A:
                A = self.transform_A(A)
        if self.transform_B:
            B = self.transform_B(B)

        if self.train:
            return {'A': A, 'B': B, 'V': V, 'A_paths': A_path, 'B_paths': B_path}
        else:
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

        hf.close()

    def __len__(self):
        return max(self.A_size, self.B_size, self.V_size)

    def transform(self, image, volume, grayscale=False, convert=True):
        
        volume = volume.reshape(128, 256, 256)
        image = transforms.functional.resize(image, (256,256))

        if convert:
            # Transform to tensor
            image = transforms.functional.to_tensor(image)
            volume = transforms.functional.to_tensor(volume)

        volume = torch.rot90(volume, 1, [0,1])
        volume = torch.flip(volume, [0])
        volume = volume / 255.0


        if grayscale:
            image = transforms.functional.Grayscale(1)

        # Random horizontal flipping
        if random.random() > 0.5:
            #image = transforms.functional.hflip(image)
            image = torch.flip(image, [1])
            volume = torch.flip(volume, [1])                               

        # Random vertical flipping
        if random.random() > 0.5:
            #image = transforms.functional.vflip(image)
            image = torch.flip(image, [2])
            volume= torch.flip(volume, [2])                               
        
        if random.random() > 0.5:
            if random.random() > 0.5:
                #image = transforms.functional.rotate(image, 90)
                image = torch.rot90(image, 1, [1,2])
                volume = torch.rot90(volume, 1, [1,2])
            else:
                #image = transforms.functional.rotate(image, 270)
                image = torch.rot90(image, 3, [1,2])
                volume = torch.rot90(volume, 3, [1,2])


        #Normalize
        if grayscale:
            image = transforms.functional.normalize(image, (0.5), (0.5))
        else:
            image = transforms.functional.normalize(image, (0.5,0.5,0.5), (0.5,0.5,0.5))

        return image, volume

    def get_transform_A(self, grayscale, convert=True):   
        transform_list = []

        transform_list += [transforms.Resize((256,256))]

        if grayscale:
            transform_list.append(transforms.Grayscale(1))

        if convert:
            transform_list += [transforms.ToTensor()]
            if grayscale:
                transform_list += [transforms.Normalize((0.5,), (0.5,))]
            else:
                transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)

    def get_transform_B(self, grayscale, convert=True):   
        transform_list = []

        # Random horizontal flipping
        transform_list += [transforms.RandomHorizontalFlip()]

        # Random vertical flipping
        transform_list += [transforms.RandomVerticalFlip()]

        transform_list += [transforms.Resize((256,256))]

        if grayscale:
            transform_list.append(transforms.Grayscale(1))

        if convert:
            transform_list += [transforms.ToTensor()]
            if grayscale:
                transform_list += [transforms.Normalize((0.5,), (0.5,))]
            else:
                transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)

if __name__ == '__main__':
    gd = GananaDataset("./")
    print(len(gd))
