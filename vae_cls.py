# -*- coding: utf-8 -*-
import os, sys, glob, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

import time, datetime
import pdb, traceback

import cv2
from PIL import Image

import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

from vae import *
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import logging
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


logging.basicConfig(level=logging.WARNING 
                    ,filename="demo.log" 
                    ,format="%(asctime)s - %(loss)s" #日志输出的格式
                    # -8表示占位符，让输出左对齐，输出长度都为8位
                    ,datefmt="%Y-%m-%d %H:%M:%S" #时间输出的格式
                    )


CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 
            '0', '1', '2', '3', '4', '5', 
            '6', '7', '8', '9')




def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(float(correct_k.mul_(1.0 / batch_size).data.cpu()))
        return res






class ReadData(Dataset):
    def __init__(self, root='./', transform=None):
        super(ReadData, self).__init__()
        data_list = glob.iglob(os.path.join(root, "*/*.jpg"))
        self.data_list = []
        self.cls_list = CLASSES#os.listdir(root)
        for x in data_list:
            if str(x).split('/')[-2] in self.cls_list:
                self.data_list.append(x)
        self.transform = transform


    def __getitem__(self, idx):
        img = Image.open(self.data_list[idx])
        if self.transform:
            img = self.transform(img)
        label = self.cls_list.index(self.data_list[idx].split('/')[-2])
        return img, label
    
    def __len__(self):
        return len(self.data_list)




def validate(val_loader, model):

    # switch to evaluate mode
    model.eval()

    deloss_sum = 0
    acc = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = model.loss_function(target, output)
            deloss_sum += loss['loss'].detach()
            acc += accuracy(output[-1], target)[0] * target.shape[0]

        print(f'acc:{acc / len(val_loader)}')
        return deloss_sum / len(val_loader)


def train(train_loader, model, optimizer, epoch, scheduler):
    model.train()

    end = time.time()
    deloss_sum = 0
    loss = {}
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)


        output = model(input)
        loss_all = model.loss_function(target, output)

        loss.update(loss_all)
        loss_bp = loss_all['loss']
        logging.info(str(loss_bp.cpu().item()))
        # measure accuracy and record loss
        acc1 = accuracy(output[-1], target, topk=(1, ))

        optimizer.zero_grad()
        loss_bp.backward()
        optimizer.step()
        scheduler.step()
        print(loss)



train_loader = torch.utils.data.DataLoader(
    ReadData('/home/fht/data/VOCdevkit/VOC2007/classification/',
            transform = transforms.Compose([
                        # transforms.RandomGrayscale(),
                        transforms.Resize((64, 64)),
                        #transforms.RandomAffine(10),
                        # transforms.ColorJitter(hue=.05, saturation=.05),
                        # transforms.RandomCrop((450, 450)),
                        #transforms.RandomHorizontalFlip(),
                        #transforms.RandomVerticalFlip(),
                        transforms.ToTensor(),
                        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    ), batch_size=128, shuffle=True, num_workers=40, pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    ReadData('/home/fht/data/VOCdevkit/VOC2007/classification/',
            transform = transforms.Compose([
                        transforms.Resize((64, 64)),
                        # transforms.Resize((124, 124)),
                        # transforms.RandomCrop((450, 450)),
                        # transforms.RandomCrop((88, 88)),
                        transforms.ToTensor(),
                        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    ), batch_size=128, shuffle=False, num_workers=40, pin_memory=True
)



# ---------------------- load dataset ----------------------
train_loader = DataLoader(MNIST('./', train=True, download=True, transform=transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
        ])),
                          batch_size=128, shuffle=True, num_workers=40, pin_memory=True)

val_loader = DataLoader(MNIST('./', train=False, download=True, transform=transforms.Compose([
                        transforms.Resize((224, 224)),
                        # transforms.Resize((124, 124)),
                        # transforms.RandomCrop((450, 450)),
                        # transforms.RandomCrop((88, 88)),
                        transforms.ToTensor(),
                        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])),
                         batch_size=128, shuffle=False, num_workers=40, pin_memory=True )







model = VanillaVAE(1,30).cuda()
optimizer = torch.optim.SGD(model.parameters(), 0.005)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)


best_acc = 1000
train_flag = False
if train_flag:
    for epoch in range(40):
        
        print('Epoch: ', epoch)

        train(train_loader, model, optimizer, epoch, scheduler)
        val_loss = validate(val_loader, model)

        if val_loss < best_acc:
            best_acc = val_loss
            torch.save(model.state_dict(), './vae.pt')
else:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        ])
    model.load_state_dict(torch.load('./vae.pt'))
    model.eval()
    val_loader = DataLoader(MNIST('./', train=False, download=True, transform=transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
        ])),
                         batch_size=1, shuffle=True, num_workers=2, pin_memory=True )

    for i, (img, label) in enumerate(val_loader):
        img = img.cuda()
        out = model(img)
        out_img = out[0]

        out_img = transforms.ToPILImage()(out_img.cpu().squeeze(0))
        img = transforms.ToPILImage()(img.cpu().squeeze(0))
        img.save('./vae_img.jpg')
        #out_img.resize(img.size)
        out_img.save('./vae_out_img.jpg')
        break




    '''




    root = '/home/fht/data/VOCdevkit/VOC2007/classification/'
    data = glob.iglob(os.path.join(root, "*/*.jpg"))
    data_list = []
    cls_list = CLASSES#os.listdir(root)
    for x in data:
        if str(x).split('/')[-2] in cls_list:
            data_list.append(x)
    from random import shuffle
    shuffle(data_list)
    for img_path in data_list:
        #img_path = './vae_out_img.jpg'
        img = Image.open(img_path)
        img_cu = transform(img).unsqueeze(0).cuda()
        out = model(img_cu)
        

        out_img = out[0]


        out_img = transforms.ToPILImage()(out_img.cpu().squeeze(0))
        img.save('./vae_img.jpg')
        #out_img.resize(img.size)
        out_img.save('./vae_out_img.jpg')
        pdb.set_trace()



        break
    '''

