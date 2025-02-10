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

import matplotlib.pyplot as plt

from PIL import Image
from scipy.io import loadmat
from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


#import custom modules
from DM_new import DMData
from utils import AverageMeter, inter_and_union

# Set environment variable to avoid OpenMP initialization conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['GDAL_DATA'] = '/home/miniconda3/share'
os.environ['PROJ_LIB'] = '/home/miniconda3/share/proj'
os.environ['PROJ_DATA'] = '/home/miniconda3/share/proj'

# Assuming mean and std used for normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

#Argument parser for command-line options
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', default=True,
                    help='training mode')
parser.add_argument('--exp', type=str, required=True,
                    help='name of experiment')
parser.add_argument('--epochs', type=int, default=30,
                    help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch size')
parser.add_argument('--base_lr', type=float, default=0.00025,
                    help='base learning rate')
parser.add_argument('--crop_size', type=int, default=512,
                    help='image crop size')

parser.add_argument('--resume', type=str, default=None,
                    help='path to checkpoint to resume from')
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers')




parser.add_argument('--num_classes', type=int, default=3,
                    help='number of classes')


args = parser.parse_args()



def main():


  nb_classes = args.num_classes

  assert torch.cuda.is_available() # Ensures a CUDA-capable GPU is available. Raises an Assertion Error if otherwise.
  torch.backends.cudnn.benchmark = True #causes cuDNN to benchmark multiple convolution algorithms and select the fastest in order to decrease runtime

# points to a folder containing a training set and loads it. See other documentation for guidance on the creation of these directories
  dataset = DMData('train/labeled_data/',
        train=True, crop_size=args.crop_size) 
  
# points to a folder containing a validation set. See other documentation for guidance on the creation of these directories
  valDataset = DMData('train/labeled_data/',
        train=False, crop_size=args.crop_size)

  ENCODER = 'resnet50'
  
  #Model definition using DeeplabV3
  from deeplabv3_model import deeplabv3
  model = deeplabv3.DeepLabV3(nb_classes,backbone=ENCODER,name_classifier='deeplab')

      
      
  if args.train:

    best_score = 0
    best_dm_score = 0
    best_model_name = '-1'
    best_dm_model_name = '-1'
    last_model_name = '-1'
    now = datetime.now()
    logdir = './ignored/'+args.exp+'/'+now.strftime("%d%m%Y_%H%M%S")+'/'
    if os.path.exists(logdir):
        Question = input("log folder exists. Delete? (yes|no)")
        if Question == ("yes"):
            shutil.rmtree(logdir)
            print ("log deleted")
        elif Question == ("no"):
            print ("log kept")
    tensorboardWriter = SummaryWriter(logdir)

    #Define loss criterion and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    model = nn.DataParallel(model).cuda()
    model.train()



    optimizer = optim.Adam(model.parameters(),lr=args.base_lr, weight_decay=0.004)

    #DataLoader for training data
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=args.train,
        pin_memory=True, num_workers=args.workers,drop_last=True)
    max_iter = args.epochs * len(dataset_loader)
    
    start_epoch = 0
    
    # Resume training from checkpoint if specified
    if args.resume:
      if os.path.isfile(args.resume):
        print('=> loading checkpoint {0}'.format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('=> loaded checkpoint {0} (epoch {1})'.format(
          args.resume, checkpoint['epoch']))
      else:
        print('=> no checkpoint found at {0}'.format(args.resume))


    #Training loop
    for epoch in range(start_epoch, args.epochs):
      losses = AverageMeter()

      is_bestScore = False

      for i, (inputs, target,_) in enumerate(dataset_loader):
        cur_iter = epoch * len(dataset_loader) + i
        lr = args.base_lr * (1 - float(cur_iter) / max_iter) ** 0.9

        inputs = Variable(inputs.cuda())
        target = Variable(target.cuda())
        outputs = model(inputs)

        

        loss = criterion(outputs, target)


        if np.isnan(loss.item()) or np.isinf(loss.item()):
          pdb.set_trace()
        losses.update(loss.item(), args.batch_size)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print('epoch: {0}\t'
              'iter: {1}/{2}\t'
              'lr: {3:.6f}\t'
              'loss: {loss.val:.4f} ({loss.ema:.4f})'.format(
              epoch + 1, i + 1, len(dataset_loader), lr, loss=losses))

      tensorboardWriter.add_scalar('Loss/train', losses.avg,epoch)
      #after iterating dataset:
      #trainLogwritter.writerow([epoch+1,losses.avg]) 
      

      if epoch % 1 == 0:
        with torch.no_grad():
          score = validation(model,valDataset,epoch,folder=logdir+'/val/')
          score_train = validation(model,dataset,epoch,random_select=40)

        tensorboardWriter.add_scalar('iou/train', score_train.mean(),epoch)

        tensorboardWriter.add_scalar('iou/val', score.mean(),epoch)
        for i, val in enumerate(score):
          #print('IoU {0}: {1:.2f}'.format(dataset.CLASSES[i], val * 100))
          tensorboardWriter.add_scalar('iou/val/' + dataset.CLASSES[i], val,epoch)


        for i, val in enumerate(score_train):
          #print('IoU {0}: {1:.2f}'.format(dataset.CLASSES[i], val * 100))
          tensorboardWriter.add_scalar('iou/train/' + dataset.CLASSES[i], val,epoch)



        dm_score = score[1]
        dm_train_score = score_train[1]

        score = score.mean()
        score_train = score_train.mean()



        if score > best_score and score>0.5:
            best_score = score

            if best_model_name != '-1':
                os.remove(best_model_name)
            best_model_name= logdir+"best_model_epoch%d.pth" % (epoch + 1)
            torch.save({
              'epoch': epoch + 1,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              }, best_model_name)


        if dm_score > best_dm_score and dm_score>0.35:
            best_dm_score = dm_score

            if best_dm_model_name != '-1':
                os.remove(best_dm_model_name)
            best_dm_model_name= logdir+"best_dm_model_epoch%d.pth" % (epoch + 1)
            torch.save({
              'epoch': epoch + 1,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              }, best_dm_model_name)





 


 
      if epoch % 50 == 49:
        if last_model_name != '-1':
            os.remove(last_model_name)
        last_model_name = logdir+"epoch%d" %(epoch + 1)
        torch.save({
          'epoch': epoch + 1,
          'state_dict': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          }, last_model_name)
  



def validation(model,dataset,epoch,folder=None, random_select=0):
    model.eval()
    if folder!= None:
      if not os.path.exists(folder):
          os.mkdir(folder)
    #   cmap = loadmat('pascal_seg_colormap.mat')['colormap']
    #   cmap = (cmap * 255).astype(np.uint8).flatten().tolist()

    inter_meter = AverageMeter()
    union_meter = AverageMeter()

    if random_select == 0 or random_select > len(dataset):
        iterationList = range(len(dataset))
    else:
        iterationList = random.sample(range(0, len(dataset)), random_select)



    for num,i in enumerate(iterationList):
      inputs, target, _ = dataset[i]
      inputs = Variable(inputs.cuda())
      outputs = model(inputs.unsqueeze(0))

      _, pred = torch.max(outputs, 1)
      pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
      mask = target.numpy().astype(np.uint8)
      imname = dataset.masks[i].split('/')[-1]
      if folder!= None and num % 5 == 0:
        # mask_pred = Image.fromarray(pred)
        # mask_pred.putpalette(cmap)
        # mask_pred.save(os.path.join(folder, imname))

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        # Denormalize and convert to numpy array
        image = denormalize(inputs.cpu(), mean, std)
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        # Ground truth
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis('off')
        # Prediction
        axes[2].imshow(pred, cmap='gray')
        axes[2].set_title("Predicted Mask")
        axes[2].axis('off')

        plt.savefig(os.path.join(folder, imname), dpi=300)
        plt.close()

      print('eval: {0}/{1}'.format(num + 1, len(dataset)))

      inter, union = inter_and_union(pred, mask, len(dataset.CLASSES))
      inter_meter.update(inter)
      union_meter.update(union)

    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    for i, val in enumerate(iou):
      print('IoU {0}: {1:.2f}'.format(dataset.CLASSES[i], val * 100))
    #print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))
    #logwriter.writerow([epoch+1,iou[0]*100,iou[1]*100,iou.mean()*100])  
    model.train()
    return iou#.mean()

def denormalize(image, mean, std):
    image = image * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
    return image

if __name__ == "__main__":
  main()
