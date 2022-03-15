# -*- coding: utf-8 -*-
import os, sys, glob, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

import time, datetime
import pdb, traceback

import cv2
from PIL import Image

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from deResnet import deresnet18

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
            res.append(float(correct_k.mul_(100.0 / batch_size).data.cpu()))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, *meters):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = ""


    def pr2int(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class VisitNet(nn.Module):
    def __init__(self):
        super(VisitNet, self).__init__()

        model = models.resnet18(True)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(512, 20)
        self.resnet = model

    def forward(self, img):
        out = F.log_softmax(self.resnet(img))
        return out


class deVisitNet(nn.Module):
    def __init__(self):
        super(deVisitNet, self).__init__()

        model = deresnet18()
        model.fc = nn.Linear(20, 512)
        self.resnet = model

    def forward(self, img):
        out = self.resnet(img)
        return out






def validate(val_loader, model, demodel):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1)

    # switch to evaluate mode
    model.eval()
    demodel.eval()
    deloss_sum = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = F.cross_entropy(output, target)
            outimg = demodel(output.detach())
            deloss = F.l1_loss(outimg, input)
            deloss_sum += deloss.item()

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1, ))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # TODO: this should also be done with the ProgressMeter
        #print(' * Acc@1 {top1.avg:.3f}'
        #      .format(top1=top1))
        print(f'Avg de loss : {deloss_sum / len(val_loader)}')
        return top1, deloss_sum / len(val_loader)

def predict(test_loader, model, tta=10):
    # switch to evaluate mode
    model.eval()

    test_pred_tta = None
    test_pred = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(test_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input, path)
            output = output.data.cpu().numpy()

            test_pred.append(output)
    test_pred = np.vstack(test_pred)
    return test_pred

def train(train_loader, model, optimizer, epoch, scheduler, demodel, deoptimizer, descheduler):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, losses, top1)

    # switch to train mode
    #model.train()
    demodel.train()

    end = time.time()
    deloss_sum = 0
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        #loss = F.cross_entropy(output, target)

        outimg = demodel(output.detach())
        deloss = F.l1_loss(outimg, input)
        deloss_sum = deloss.item() / input.size(0)

        # measure accuracy and record loss
        #acc1 = accuracy(output, target, topk=(1, ))
        #losses.update(loss.item(), input.size(0))
        #top1.update(acc1[0], input.size(0))
        # top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()
        #scheduler.step()

        deoptimizer.zero_grad()
        deloss.backward()
        deoptimizer.step()
        descheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            #print(f'deloss_sum: {deloss_sum}')
            progress.pr2int(i)




model = VisitNet().cuda()
model = nn.DataParallel(model).cuda()
model.load_state_dict(torch.load('./model_res18.pt'))
model.eval()

demodel = deVisitNet().cuda()
demodel = nn.DataParallel(demodel).cuda()
demodel.load_state_dict(torch.load('./demodel_epoch15_res18.pt'))
demodel.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])



root = '/home/fht/data/VOCdevkit/VOC2007/classification/'
data = glob.iglob(os.path.join(root, "*/*.jpg"))
data_list = []
cls_list = CLASSES#os.listdir(root)
for x in data:
    if str(x).split('/')[-2] in cls_list:
        data_list.append(x)
from random import shuffle
shuffle(data_list)
debug = True
debug = False
for img_path in data_list:
    if debug:
        img_path = './out_img.jpg'
    img = Image.open(img_path)
    img_cu = transform(img).unsqueeze(0).cuda()


    out = model(img_cu)
    out_soft = F.softmax(out, dim=1)
    print(torch.argmax(out_soft))
    print(torch.max(out_soft))
    if debug:
        break
    if torch.max(out_soft).item() > 0.8:
        out_img = demodel(out).cpu().clone()
        img.save('./img.jpg')
        pdb.set_trace()
        out_img = transforms.Normalize([-0.485, -0.456, -0.406], [1/0.229, 1/0.224, 1/0.225])(out_img)
        out_img = transforms.ToPILImage()(out_img.squeeze(0))
        out_img.save('./out_img.jpg')
        break

