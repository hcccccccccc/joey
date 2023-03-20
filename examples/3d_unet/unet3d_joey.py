from itertools import product
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import joey
from torchsummary import summary

def CB(input_size, kernel_size, stride=1, padding=1):
    conv = joey.Conv3D(kernel_size=kernel_size, input_size=input_size, stride=stride, padding=padding, activation=joey.activation.LeakyReLU())
    inst = joey.InstanceNorm3D(input_size=input_size,generate_code=True)
    # lrelu = joey.activation.LeakyReLU()

    CB = [conv, inst]
    return (joey.Net(CB), CB)

def DB(in_channel, filter, stride):
    CB1 = CB(in_channel, filter, stride),
    # nn.Dropout3d(),
    CB2 = CB(filter, filter)
    DB = [CB1, CB2]
    return (joey.Net(DB), DB)

def create_unet3d(batch_size=2, filter=8):
    # DB1_CB = CB(input_size, kernel_size)
    DB1_conv1 = joey.Conv3D(input_size=(batch_size, 4,128,128,128), kernel_size=(filter,3,3,3), stride=1, padding=1, activation=joey.activation.LeakyReLU()),
    DB1_inst1 = joey.InstanceNorm3d(input_size=(batch_size,filter,128,128,128), generate_code=True),
    DB1_lrelu1 = joey.activation.LeakyReLU()

    # unet3d = [DB1_CB, DB1]
    # return(joey.Net(unet3d), unet3d)
        # DB1 = DB(filter, filter, stride=1)
    DB1_conv2 = joey.Conv3D(input_size=(batch_size, filter,128,128,128), kernel_size=(filter,3,3,3), stride=1, padding=1, activation=joey.activation.LeakyReLU()),
    DB1_inst2 = joey.InstanceNorm3d(input_size=(batch_size,filter,128,128,128), generate_code=True),
    DB1_lrelu2 = joey.activation.LeakyReLU()

    DB1_drop1 = nn.Dropout3d()

    DB1_conv3 = joey.Conv3D(input_size=(batch_size, filter,128,128,128), kernel_size=(filter,3,3,3), stride=1, padding=1, activation=joey.activation.LeakyReLU()),
    DB1_inst3 = joey.InstanceNorm3d(input_size=(batch_size, filter,128,128,128)),
    DB1_lrelu3 = joey.LeakyReLU()

        # DB2_CB = CB(filter, filter*2, stride=2, k_size=3)
    DB2_conv1 = joey.Conv3D(input_size=(batch_size, filter,128,128,128), kernel_size=(filter*2,3,3,3), stride=2, padding=1, activation=joey.activation.LeakyReLU()),
    DB2_inst1 = joey.InstanceNorm3d(input_size=(batch_size,filter*2,64,64,64), generate_code=True),
    DB2_lrelu1 = joey.LeakyReLU()
        # DB2 = DB(filter*2, filter*2, stride=1)
    DB2_conv2 = joey.Conv3D(input_size=(batch_size,filter*2,64,64,64), kernel_size=(filter*2,3,3,3), stride=1, padding=1, activation=joey.activation.LeakyReLU()),
    DB2_inst2 = joey.InstanceNorm3d(input_size=(batch_size,filter*2,64,64,64), generate_code=True),
    DB2_lrelu2 = joey.LeakyReLU()

    DB2_drop1 = nn.Dropout3d()

    DB2_conv3 = joey.Conv3D(input_size=(batch_size,filter*2,64,64,64), kernel_size=(filter*2,3,3,3), stride=1, padding=1, activation=joey.activation.LeakyReLU()),
    DB2_inst3 = joey.InstanceNorm3d(input_size=(batch_size,filter*2,64,64,64), generate_code=True),
    DB2_lrelu3 = joey.LeakyReLU()

        # DB3_CB = CB(filter*2, filter*4, stride=2, k_size=3)
    DB3_conv1 = joey.Conv3D(input_size=(batch_size,filter*2,64,64,64), kernel_size=(filter*4,3,3,3), stride=2, padding=1, activation=joey.activation.LeakyReLU()),
    DB3_inst1 = joey.InstanceNorm3d(input_size=(batch_size,filter*4,32,32,32), generate_code=True),
    DB3_lrelu1 = joey.LeakyReLU()
        # DB3 = DB(filter*4, filter*4, stride=1)
    DB3_conv2 = joey.Conv3D(input_size=(batch_size,filter*4,32,32,32), kernel_size=(filter*4,3,3,3), stride=1, padding=1, activation=joey.activation.LeakyReLU()),
    DB3_inst2 = joey.InstanceNorm3d(input_size=(batch_size,filter*4,32,32,32), generate_code=True),
    DB3_lrelu2 = joey.LeakyReLU()

    DB3_drop1 = nn.Dropout3d()

    DB3_conv3 = joey.Conv3D(input_size=(batch_size,filter*4,32,32,32), kernel_size=(filter*4,3,3,3), stride=1, padding=1, activation=joey.activation.LeakyReLU()),
    DB3_inst3 = joey.InstanceNorm3d(input_size=(batch_size,filter*4,32,32,32), generate_code=True),
    DB3_lrelu3 = joey.LeakyReLU()

        # DB4_CB = CB(filter*4, filter*8, stride=2, k_size=3)
    DB4_conv1 = joey.Conv3D(input_size=(batch_size,filter*4,32,32,32), kernel_size=(filter*8,3,3,3), stride=2, padding=1, activation=joey.activation.LeakyReLU()),
    DB4_inst1 = joey.InstanceNorm3d(input_size=(batch_size,filter*8,16,16,16), generate_code=True),
    DB4_lrelu1 = joey.LeakyReLU()
        # DB4 = DB(filter*8, filter*8, stride=1)
    DB4_conv2 = joey.Conv3D(input_size=(batch_size,filter*8,16,16,16), kernel_size=(filter*8,3,3,3), stride=1, padding=1, activation=joey.activation.LeakyReLU()),
    DB4_inst2 = joey.InstanceNorm3d(input_size=(batch_size,filter*8,16,16,16), generate_code=True),
    DB4_lrelu2 = joey.LeakyReLU()

    DB4_drop1 = nn.Dropout3d()

    DB4_conv3 = joey.Conv3D(input_size=(batch_size,filter*8,16,16,16), kernel_size=(filter*8,3,3,3), stride=1, padding=1, activation=joey.activation.LeakyReLU()),
    DB4_inst3 = joey.InstanceNorm3d(input_size=(batch_size,filter*8,16,16,16), generate_code=True),
    DB4_lrelu3 = joey.LeakyReLU()

        # DB5_CB = CB(filter*8, filter*16, stride=2, k_size=3)
    DB5_conv1 = joey.Conv3D(input_size=(batch_size,filter*8,16,16,16), kernel_size=(filter*16,3,3,3), stride=2, padding=1, activation=joey.activation.LeakyReLU()),
    DB5_inst1 = joey.InstanceNorm3d(input_size=(batch_size,filter*16,8,8,8), generate_code=True),
    DB5_lrelu1 = joey.LeakyReLU()
        # DB5 = DB(filter*16, filter*16, stride=1)
    DB5_conv2 = joey.Conv3D(input_size=(batch_size,filter*16,8,8,8), kernel_size=(filter*16,3,3,3), stride=1, padding=1, activation=joey.activation.LeakyReLU()),
    DB5_inst2 = joey.InstanceNorm3d(input_size=(batch_size,filter*16,8,8,8), generate_code=True),
    DB5_lrelu2 = joey.LeakyReLU()

    DB5_drop1 = nn.Dropout3d()

    DB5_conv3 = joey.Conv3D(input_size=(batch_size,filter*16,8,8,8), kernel_size=(filter*16,3,3,3), stride=1, padding=1, activation=joey.activation.LeakyReLU()),
    DB5_inst3 = joey.InstanceNorm3d(input_size=(batch_size,filter*16,8,8,8), generate_code=True),
    DB5_lrelu3 = joey.LeakyReLU()



        # Upward Block
        # UB1_U3_CB = UB_U3_CB(filter*16, filter*8)
    UB1_U3 = joey.UpSample(scale_factor=2, input_size=(batch_size, filter*16,8,8,8))

    UB1_conv1 = joey.Conv3D(input_size=(batch_size,filter*16,16,16,16), kernel_size=(filter*8,3,3,3), stride=1, padding=1, activation=joey.activation.LeakyReLU()),
    UB1_inst1 = joey.InstanceNorm3d(input_size=(batch_size,filter*8,16,16,16), generate_code=True),
    UB1_lrelu1 = joey.LeakyReLU()

        # UB1_CB_CB = UB_CB_CB(filter*16, filter*8)
    UB1_conv2 = joey.Conv3D(input_size=(batch_size,filter*16,16,16,16), kernel_size=(filter*8,3,3,3), stride=1, padding=1, activation=joey.activation.LeakyReLU()),
    UB1_inst2 = joey.InstanceNorm3d(input_size=(batch_size,filter*8,16,16,16), generate_code=True),
    UB1_lrelu2 = joey.LeakyReLU()

    UB1_conv3 = joey.Conv3D(input_size=(batch_size,filter*8,16,16,16), kernel_size=(filter*8,1,1,1), stride=1, padding=0, activation=joey.activation.LeakyReLU()),
    UB1_inst3 = joey.InstanceNorm3d(input_size=(batch_size,filter*8,16,16,16), generate_code=True),
    UB1_lrelu3 = joey.LeakyReLU()

        # UB2_U3_CB = UB_U3_CB(filter*8, filter*4)
    UB2_U3 = joey.UpSample(scale_factor=2, input_size=(batch_size, filter*8,16,16,16))

    UB2_conv1 = joey.Conv3D(input_size=(batch_size,filter*8,32,32,32), kernel_size=(filter*4,3,3,3), stride=1, padding=1, activation=joey.activation.LeakyReLU()),
    UB2_inst1 = joey.InstanceNorm3d(input_size=(batch_size,filter*4,32,32,32), generate_code=True),
    UB2_lrelu1 = joey.LeakyReLU()

        # UB2_CB_CB = UB_CB_CB(filter*8, filter*4)
    UB2_conv2 = joey.Conv3D(input_size=(batch_size,filter*8,32,32,32), kernel_size=(filter*4,3,3,3), stride=1, padding=1, activation=joey.activation.LeakyReLU()),
    UB2_inst2 = joey.InstanceNorm3d(input_size=(batch_size,filter*4,32,32,32), generate_code=True),
    UB2_lrelu2 = joey.LeakyReLU()

    UB2_conv3 = joey.Conv3D(input_size=(batch_size,filter*4,32,32,32), kernel_size=(filter*4,1,1,1), stride=1, padding=0, activation=joey.activation.LeakyReLU()),
    UB2_inst3 = joey.InstanceNorm3d(input_size=(batch_size,filter*8,16,16,16), generate_code=True),
    UB2_lrelu3 = joey.LeakyReLU()

    #     # UB3_U3_CB = UB_U3_CB(filter*4, filter*2)
    #     UB3_U3 = U3(filter*4)

    #     UB3_conv1 = joey.Conv3D(filter*4, filter*2, kernel_size=kernel_size, stride=1, padding=1, bias=False),
    #     UB3_inst1 = joey.InstanceNorm3d(filter*2),
    #     UB3_lrelu1 = joey.LeakyReLU()
    #     # UB3_CB_CB = UB_CB_CB(filter*4, filter*2)
    #     UB3_conv2 = joey.Conv3D(filter*4, filter*2, kernel_size=kernel_size, stride=1, padding=0, bias=False),
    #     UB3_inst2 = joey.InstanceNorm3d(filter*2),
    #     UB3_lrelu2 = joey.LeakyReLU()

    #     UB3_conv3 = joey.Conv3D(filter*2, filter*2, kernel_size=kernel_size, stride=1, padding=0, bias=False),
    #     UB3_inst3 = joey.InstanceNorm3d(filter*2),
    #     UB3_lrelu3 = joey.LeakyReLU()
    #     # UB4_U3_CB = UB_U3_CB(filter*2, filter)
    #     UB4_U3 = U3(filter*2)

    #     UB4_conv1 = joey.Conv3D(filter*2, filter, kernel_size=kernel_size, stride=1, padding=1, bias=False),
    #     UB4_inst1 = joey.InstanceNorm3d(filter),
    #     UB4_lrelu1 = joey.LeakyReLU()
    #     # UB4_CB_CB = UB_CB_CB(filter*2, filter)
    #     UB4_conv2 = joey.Conv3D(filter*2, filter, kernel_size=kernel_size, stride=1, padding=0, bias=False),
    #     UB4_inst2 = joey.InstanceNorm3d(filter),
    #     UB4_lrelu2 = joey.LeakyReLU()

    #     UB4_conv3 = joey.Conv3D(filter, filter, kernel_size=kernel_size, stride=1, padding=0, bias=False),
    #     UB4_inst3 = joey.InstanceNorm3d(filter),
    #     UB4_lrelu3 = joey.LeakyReLU()

    #     C3_1 = C3(filter*4)
    #     C3_2 = C3(filter*2)
    #     C3_3 = C3(filter)
    #     U3_1 = U3()
    #     U3_2 = U3()


size = (1,2,30,30,30)
net = joey.InstanceNorm3D(input_size=(size),generate_code=True)
a = torch.rand(size)
input_numpy = a.detach().numpy()
m = joey.InstanceNorm3D(input_size=(size),generate_code=True)
inst = nn.InstanceNorm3d(2)
out1 = m.execute(input_numpy)
out2 = inst(a)
# out = net(a)
# print(net)
# print(a)

# a = [1,2],[3,4],[5,6]
# from itertools import product
# print(a)
# b = product(*a)
# for i in b:
#     print(i)