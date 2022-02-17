import argparse
import logging
from logging import warning as warn
import os
import shutil  
import imageio

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from models.cyclegan_model import cycleGAN
from dataset import GananaDataset

import time
from datetime import datetime

timestr = time.strftime("%m%d-%H%M%S")
name = "less1and3"
#data_root = '/db/pszaj/proj-3d-plant/cyclegan-fayoum-wbkg/'
data_root = '/db/psyzh/Ganana_Datasets/2021-01-18_Test_Dataset'
dir_predictions = './predictions/' + name + "/"

lr = 0.0002
gpu_ids=[0]
n_epochs = 100
n_epochs_decay = 100
batch_size = 1

try:
    os.mkdir(dir_predictions)   
    logging.info('Created checkpoint directory')
except OSError:
    pass

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--cyclegan', '-c', metavar='CYCLE', default=False, nargs='+', type=bool,
                        help='Use Cyclegan before Unet: t/f')
                        
    parser.add_argument('--load_iter', '-l', metavar='LOAD_ITER', default=1, nargs='+', type=int,
                        help='Iteration to')

    parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')

    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of ouput images')

    return parser.parse_args()




if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    logging.info("Creating Cyclegan")
    model = cycleGAN(device, name=name, lr=lr, gpu_ids=gpu_ids, isTrain=False)
    model.setup(n_epochs, n_epochs_decay, load_iter=args.load_iter[0])

    logging.info("Cylegan loaded !")

    dataset = GananaDataset(data_root, train=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    n_test = len(dataset)

    for i, data in enumerate(dataloader):
        if i >= args.num_test:  # only apply our model to opt.num_test images.
            break
        #logging.info("Ganana-ising")
        model.set_input(data)  # unpack data from data loader

        now = datetime.now().time()
        print("Time before = ", now)
        for j in range(0, 1000):
            model.test()           # run inference
        now = datetime.now().time()
        print("Time after = ", now)

        img = model.fake_B
        img = img.squeeze(0)
        original_filename = data["A_paths"][0]
        original_filename = os.path.basename(original_filename)
        original_filename = os.path.splitext(original_filename)[0]
        logging.info(original_filename)

        mask = model.fake_V
        mask = mask.squeeze().cpu().numpy()
        
        logging.info("before: " + str(mask.max()))
        mask = mask * 255
        logging.info("after: " + str(mask.max()))
        out_fn = dir_predictions + original_filename + '.npy'
        out_gt_fn = dir_predictions + original_filename + '.png'   
        cg_fn = dir_predictions + original_filename + '_cycle.png'
        logging.info("\nMask Shape: " + str(mask.shape))
        np.save(out_fn, mask)

        #imageio.imwrite(cg_fn, np.rollaxis(img.cpu().detach().squeeze().numpy(), 0, 3))
        #imageio.imwrite(out_gt_fn, np.rollaxis(model.real_A.cpu().detach().squeeze().numpy(), 0, 3))

        #shutil.copy(dataset[i]['A_paths'], out_gt_fn)

        logging.info("Mask saved to {}".format(out_fn))
        logging.info("Image copied to {}".format(out_gt_fn))


'''
def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))
'''
