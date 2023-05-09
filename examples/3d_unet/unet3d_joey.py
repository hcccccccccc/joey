from itertools import product
import numpy as np
import cupy as cp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import joey
import time
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from apex import amp
import time
import multiprocessing
import gc

import matplotlib.pyplot as plt
import barts2019loader
import diceloss
import unet3d_test
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
def CB(input_size, kernel_size, stride=(1,1,1), padding=(1,1,1)):
    conv = joey.Conv3D(kernel_size=kernel_size, input_size=input_size, stride=stride, padding=padding,strict_stride_check=False)
    temp_size = (input_size[0], kernel_size[0], input_size[2],input_size[3], input_size[4])
    if stride[0] != 1:
        temp_size = (input_size[0], kernel_size[0], int(input_size[2]/2), int(input_size[3]/2), int(input_size[4]/2))
    inst = joey.InstanceNorm3D(input_size=temp_size, activation=joey.activation.LeakyReLU())
    CB = [conv, inst]
    return CB


def C3(input_size, kernel_size=(3,1,1,1),activation=None):
    conv = joey.Conv3D(kernel_size=kernel_size, input_size=input_size, stride=(1,1,1), padding=(1,1,1),activation=activation)
    return [conv]


def DB(input_size, kernel_size, stride=(1,1,1)):
    CB1 = CB(input_size, kernel_size, stride=stride)
    if stride[0] == 1: 
        temp_size = (input_size[0], kernel_size[0], input_size[2],input_size[3], input_size[4])
    else:
        temp_size = (input_size[0], kernel_size[0], int(input_size[2]/2), int(input_size[3]/2), int(input_size[4]/2))
    CB2 = CB(input_size=temp_size, kernel_size=kernel_size)
    CB3 = CB(input_size=temp_size, kernel_size=kernel_size)
    connect = joey.add(input_size=temp_size, layer=CB1[-1])
    DB = CB1 + CB2 + CB3 + [connect]
    return DB


def UB(input_size, kernel_size, cat_layer):
    U3 = joey.UpSample(input_size= (input_size),scale_factor=2)
    CB1 = CB(input_size=(input_size[0], input_size[1], input_size[2]*2, input_size[3]*2, input_size[4]*2), kernel_size=kernel_size)
    cat = joey.cat(input_size=(input_size[0], kernel_size[0], input_size[2]*2, input_size[3]*2, input_size[4]*2), layer=cat_layer)
    CB2 = CB(input_size=(input_size[0], input_size[1], input_size[2]*2, input_size[3]*2, input_size[4]*2), kernel_size=kernel_size)
    CB3 = CB(input_size=(input_size[0], kernel_size[0], input_size[2]*2, input_size[3]*2, input_size[4]*2), kernel_size=(kernel_size[0],1,1,1), padding=(0,0,0))
    return [U3] + CB1 + [cat] + CB2 + CB3


def unet_joey(batch_size, in_channel, depth, height, weight, filter):
    # Downward block
    DB1 = DB(input_size=(batch_size, in_channel, depth, height, weight), kernel_size=(filter,3,3,3))
    DB2 = DB(input_size=(batch_size, filter, depth, height, weight), kernel_size=(filter*2,3,3,3), stride=(2,2,2))
    DB3 = DB(input_size=(batch_size, filter*2, int(depth/2), int(height/2), int(weight/2)), kernel_size=(filter*4,3,3,3), stride=(2,2,2))
    # DB4 = DB(input_size=(batch_size, filter*4, int(depth/4), int(height/4), int(weight/4)), kernel_size=(filter*8,3,3,3), stride=(2,2,2))
    # DB5 = DB(input_size=(batch_size, filter*8, int(depth/8), int(height/8), int(weight/8)), kernel_size=(filter*16,3,3,3), stride=(2,2,2))

    DB_part = DB1 + DB2 + DB3

    # Upward Block
    # UB1 = UB(input_size=(batch_size, filter*16, int(depth/16), int(height/16), int(weight/16)), kernel_size=(filter*8,3,3,3), cat_layer=DB4[-1])
    # UB2 = UB(input_size=(batch_size, filter*8, int(depth/8), int(height/8), int(weight/8)), kernel_size=(filter*4,3,3,3), cat_layer=DB3[-1])
    UB3 = UB(input_size=(batch_size, filter*4, int(depth/4), int(height/4), int(weight/4)), kernel_size=(filter*2,3,3,3),cat_layer=DB2[-1])
    UB4 = UB(input_size=(batch_size, filter*2, int(depth/2), int(height/2), int(weight/2)), kernel_size=(filter,3,3,3),cat_layer=DB1[-1])
    C3_3 = C3(input_size=(batch_size, filter, depth, height, weight),kernel_size=(3,1,1,1),activation=joey.activation.Sigmoid())

    UB_part =  UB3 + UB4 + C3_3
    unet = DB_part + UB_part

    return(joey.Net(unet))



def train(net, inputs, targets, criterion, pytorch_optimizer, device):
    def dice_loss(layer, target):
        eps = 0.00001
        pred = layer.result.data

        target = target.numpy()
        # result = pred
        # for b in range(target.shape[0]):
        #     for c in range(target.shape[1]):
        #         intersection = (pred[b,c,:,:,:] * target[b,c,:,:,:]).sum()
        #         union =(pred[b,c,:,:,:] + target[b,c,:,:,:]).sum()
        #         result[b,c,:,:,:] = 2*(intersection-target[b,c,:,:,:]*union)/(union**2)
        result = 1- 2*((pred*target+eps)/(pred+target+eps))
        return result

    inputs = torch.stack(inputs, dim=1)
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = net.forward(inputs)
    net.backward(targets, dice_loss, pytorch_optimizer)
    outputs = torch.from_numpy(outputs)
    diceloss = criterion(outputs, targets)
    del inputs, targets, outputs
    return diceloss[0]


if __name__ == "__main__":
    # envs, args
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 10
    data_root = '/run/datasets/MICCAI_BraTS_2019_Data_Training' 
    batch_size = 2
    workers = 4
    image_size = 64
    channel_size = 4
    lr = 5e-4
    pool = multiprocessing.Pool(processes=workers)

    # dataset, dataloader
    barts2019 = barts2019loader.BratsDataset(data_root, image_size, 'joey')
    train_idx, val_idx = train_test_split(list(range(len(barts2019))), test_size=0.203, random_state=42)
    train_dataset = Subset(barts2019, train_idx)
    val_dataset = Subset(barts2019, val_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # model
    start = time.time()
    net = unet_joey(batch_size, 4, image_size, image_size, image_size, channel_size)
    finish = time.time()-start
    print("Time spend: {:.0f}m {:.0f}s".format(finish // 60, finish % 60))
    criterion=diceloss.WeightedMulticlassDiceLoss(num_classes=3, class_weights=[0.5,0.3,0.2])
    optimizer = optim.Adam(net.pytorch_parameters, lr=lr)
    # summary(net, input_size=(4,128,128,128))

    #train
    total_train_loss = []
    all_time = []
    total_time = time.time()
    for i in range(num_epochs):
        start = time.time()

        results = [pool.apply_async(train, args=(net, inputs, targets, criterion, optimizer, device)) for inputs,targets in train_loader]
        total_loss = [loss.get() for loss in results]
        gc.collect()

        finish = time.time()-start
        print("Time spend: {:.0f}m {:.0f}s".format(finish // 60, finish % 60))
        all_time.append(finish)
        print(f'Epoch: {i+1}, Training Loss: {sum(total_loss)/len(total_loss):.4f}')
        total_train_loss.append(sum(total_loss)/len(total_loss))
        plt.clf()
        plt.plot(range(0, i+1),total_train_loss, label='train loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('joey_loss.png')
        del results, total_loss

    pool.close()
    pool.join()

    total_time = time.time() - total_time
    print("Total time: {:.0f}m {:.0f}s | Average time cost: {:.0f}m {:.0f}s".format(total_time // 60, total_time % 60, (total_time // num_epochs) // 60, (total_time // num_epochs )% 60))
    print("{:.0f}m {:.0f}s".format(t // 60, t % 60) for t in all_time) 
