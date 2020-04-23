import argparse
import logging
from logging import warning as warn
import os
import shutil  

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2
from torchvision import transforms

from models.unet_model import UNet
from dataset import GananaDataset

import time

timestr = time.strftime("%m%d-%H%M%S")

data_root = '/db/pszaj/proj-3d-plant/volumetric-cyclegan/'
#data_root = '/db/psyzh/volumetric-cyclegan/'
dir_predictions = './predictions/'
try:
    os.mkdir(dir_predictions)   
    logging.info('Created checkpoint directory')
except OSError:
    pass

class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

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
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images')

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []
    timestr = time.strftime("%m%d-%H%M%S")

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("output/{}_{}_OUT{}".format(pathsplit[0], timestr, pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    #out_files = get_output_filenames(args)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    net = UNet(n_channels=3, n_classes=128)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net = MyDataParallel(net)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    dataset = GananaDataset(data_root, train=False)
    n_test = len(dataset)

    for i in range(0, n_test):
        img = dataset[i]['B']

        mask = predict_img(net=net,
                            full_img=img,
                            scale_factor=args.scale,
                            out_threshold=args.mask_threshold,
                            device=device)
        logging.info("before: " + str(mask.max()))
        mask = mask * 255
        logging.info("after: " + str(mask.max()))
        if not args.no_save:
            out_fn = dir_predictions + str(i) + '.npy'
            #logging.info("\nMask Shape: " + str(mask.shape))
            np.save(out_fn, mask)

            out_gt_fn = dir_predictions + str(i) + '.png'  
            shutil.copy(dataset[i]['B_paths'], out_gt_fn)

            logging.info("Mask saved to {}".format(out_fn))
            logging.info("Image copied to {}".format(out_gt_fn))
            

'''
    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_fn = out_files[i]
            logging.info("\nMask Shape: " + str(mask.shape))
            np.save(out_files[i], mask)

            logging.info("Mask saved to {}".format(out_files[i]))
'''
