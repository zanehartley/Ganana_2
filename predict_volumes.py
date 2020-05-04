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

timestr = time.strftime("%m%d-%H%M%S")

data_root = '/db/pszaj/proj-3d-plant/volumetric-cyclegan/'
dir_predictions = './predictions/'

name = "Ganana_Test"
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

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    
    #array = np.asarray(full_img)
    #img = torch.from_numpy(array)
    #img = img.permute(2, 0, 1)

    img = full_img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        logging.info("Output Shape: " + str(output.shape))
        logging.info("Output Max: " + str(output.max()))

        '''
        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)
        '''
        probs = output.squeeze(0)
        #probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                #transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        #probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

        logging.info("Full mask shape: "  + str(full_mask.shape))

    return full_mask# > out_threshold


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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    n_test = len(dataset)

    for i, data in enumerate(dataloader):
        if i >= args.num_test:  # only apply our model to opt.num_test images.
            break
        #logging.info("Ganana-ising")
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        img = model.fake_B
        img = img.squeeze(0)

        mask = model.fake_V
        mask = mask.squeeze().cpu().numpy()
        
        logging.info("before: " + str(mask.max()))
        mask = mask * 255
        logging.info("after: " + str(mask.max()))
        out_fn = dir_predictions + str(i) + '.npy'
        out_gt_fn = dir_predictions + str(i) + '.png'   
        cg_fn = dir_predictions + str(i) + '_cycle.png'
        #logging.info("\nMask Shape: " + str(mask.shape))
        np.save(out_fn, mask)

        imageio.imwrite(cg_fn, np.rollaxis(img.cpu().detach().squeeze().numpy(), 0, 3))
        imageio.imwrite(out_gt_fn, np.rollaxis(model.real_A.cpu().detach().squeeze().numpy(), 0, 3))

        #shutil.copy(dataset[i]['A_paths'], out_gt_fn)

        logging.info("Mask saved to {}".format(out_fn))
        logging.info("Image copied to {}".format(out_gt_fn))


'''
def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))
'''