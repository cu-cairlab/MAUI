# import packages
import argparse
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
import random
import cv2
from PIL import Image
from scipy.io import loadmat
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import segmentation_models_pytorch as smp
import csv

#import custom modules
from DM_new import DMData
from utils import AverageMeter, inter_and_union
from ImageMask2Tiles_uniform  import ImageMask2Tiles as ImageMask2Tiles

# change brightness of the output mask for improved visualization. 
# this does not affect the mask itself.
def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

# generate mask using deeplabv3 architecture
def inference(data_dir, output_dir, visualize_dir, weight_dir):

    """
    Perform semantic segmentation on images in the given directory.
    Args:
        data_dir: Directory containing input images.
        output_dir: Directory to save segmentation masks.
        visualize_dir: Directory to save visualization images.
        weight_dir: Path to the trained model weights.
    """
    # Define image transformations (normalization and tensor conversion)
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    nb_classes = 2 #number of classes in model
    ENCODER = 'resnet50'
    from deeplabv3_model import deeplabv3 # MUST HAVE the deeplabv3_model folder in the working directory
    model = deeplabv3.DeepLabV3(nb_classes,backbone=ENCODER,name_classifier='deeplab')
    model = nn.DataParallel(model).cuda()
    checkpoint = torch.load(weight_dir)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    cmap = loadmat('pascal_seg_colormap.mat')['colormap'] # MUST HAVE the pascal_seg_colormap.mat file in the working directory
    cmap = (cmap * 255).astype(np.uint8).flatten().tolist()

    for fn in os.listdir(data_dir):
        img_fp = os.path.join(data_dir, fn)
        img_shape = cv2.imread(img_fp).shape
        img_tiler = ImageMask2Tiles(img_shape,(150,150),isPadding=True, paddingColor=[255,255,255])
        
        img = cv2.imread(img_fp)
        tiles = img_tiler.processImg(img)
        tiles_keys = tiles.keys()
        inference_results = {}
        for key in tiles_keys:
            tile = tiles[key]
            input_data = data_transforms(tile).unsqueeze_(0).cuda()
            print(input_data.size())
            outputs = model(input_data)

            _, pred = torch.max(outputs, 1)
            pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
            if visualize_dir != '':
                mask_pred = Image.fromarray(pred)
                mask_pred.putpalette(cmap)
                mask_pred = mask_pred.convert('RGB')
                inference_results[key] = np.array(mask_pred)
            
        assembled_img = img_tiler.assemblyImg(inference_results).astype(np.uint8)
        output_fp = os.path.join(output_dir, fn[:-4]+'_mask.png')
        cv2.imwrite(output_fp, assembled_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        img = change_brightness(img, value=30) 
        cv2.addWeighted(img, 0.5, assembled_img, 0.5, 0, assembled_img)

        output_fp = os.path.join(visualize_dir, fn[:-4]+'_visual.png')
        cv2.imwrite(output_fp, assembled_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform inference with DeepLabV3 model.')
    parser.add_argument('data_dir', type=str, help='Directory containing input images')
    parser.add_argument('output_dir', type=str, help='Output directory for segmentation masks')
    parser.add_argument('visualize_dir', type=str, help='Output directory for visualization images')
    parser.add_argument('weight_dir', type=str, help='Path to the model weight file (.pth)')
    args = parser.parse_args()

    inference(args.data_dir, args.output_dir, args.visualize_dir, args.weight_dir)
