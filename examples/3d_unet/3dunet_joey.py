from itertools import product
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import joey
from torchsummary import summary

# class Downward(nn.Module):


class test_joey(nn.Module):
    def __init__(self, in_channel = None, filter = 8) -> None:
        super(test_joey, self).__init__()
        self.in_channel = in_channel
        self.filter = filter

        # Downward block
        self.DB1_CB = self.CB(in_channel, self.filter, stride=1, k_size=3)
        self.DB1 = self.DB(self.filter, self.filter, stride=1)
        self.DB2_CB = self.CB(self.filter, self.filter*2, stride=2, k_size=3)
        self.DB2 = self.DB(self.filter*2, self.filter*2, stride=1)
        self.DB3_CB = self.CB(self.filter*2, self.filter*4, stride=2, k_size=3)
        self.DB3 = self.DB(self.filter*4, self.filter*4, stride=1)
        self.DB4_CB = self.CB(self.filter*4, self.filter*8, stride=2, k_size=3)
        self.DB4 = self.DB(self.filter*8, self.filter*8, stride=1)
        self.DB5_CB = self.CB(self.filter*8, self.filter*16, stride=2, k_size=3)
        self.DB5 = self.DB(self.filter*16, self.filter*16, stride=1)



        # Upward Block
        self.UB1_U3_CB = self.UB_U3_CB(self.filter*16, self.filter*8)
        self.UB1_CB_CB = self.UB_CB_CB(self.filter*16, self.filter*8)
        self.UB2_U3_CB = self.UB_U3_CB(self.filter*8, self.filter*4)
        self.UB2_CB_CB = self.UB_CB_CB(self.filter*8, self.filter*4)
        self.UB3_U3_CB = self.UB_U3_CB(self.filter*4, self.filter*2)
        self.UB3_CB_CB = self.UB_CB_CB(self.filter*4, self.filter*2)
        self.UB4_U3_CB = self.UB_U3_CB(self.filter*2, self.filter)
        self.UB4_CB_CB = self.UB_CB_CB(self.filter*2, self.filter)


    def CB(self, in_channel, filter, stride, k_size, padding=1):
        return nn.Sequential(joey.Conv3D(in_channel, filter, kernel_size=(3,3,3), stride=stride, padding=padding, bias=False),
                             joey.InstanceNorm3d(filter),
                             joey.LeakyReLU())

    def C3(self, in_channel):
        return joey.Conv3D(in_channel, 3, kernel_size=1, stride=1, padding=0, bias=False)
    
    def DB(self, in_channel, filter, stride):
        return nn.Sequential(self.CB(in_channel, filter,stride),
                             nn.Dropout3d(),
                             self.CB(filter, filter))
    
    def U3(self, in_channel):
        return joey.Upsample(in_channel, scale_factor=2, mode='nearest')


    
    def UB_U3_CB(self, in_channel, filter):
        return nn.Sequential(self.U3(in_channel),
                             self.CB(in_channel, filter))

    def UB_CB_CB(self, in_channel, filter):
        return nn.Sequential(self.CB(in_channel, filter),
                             self.CB(filter, filter, kernel_size=1,padding=0))

    def forward(self, input):

        #Level 1 DB(16,1)
        out = self.DB1_CB(input)
        tempout1 = out
        out = self.DB1(out)
        out += tempout1
        DB1 = out

        #Level 2 DB(32,2)
        out = self.DB2_CB(out)
        tempout2 = out
        out = self.DB2(out)
        out += tempout2
        DB2 = out

        #Level 3 DB(64,2)
        out = self.DB3_CB(out)
        tempout3 = out
        out = self.DB3(out)
        out += tempout3
        DB3 = out

        #Level 4 DB(128,2)
        out = self.DB4_CB(out)
        tempout4 = out
        out = self.DB4(out)
        out += tempout4
        DB4 = out
        
        #Level 5 DB(256,2)
        out = self.DB5_CB(out)
        tempout5 = out
        out = self.DB5(out)
        out += tempout5
        
        #Level 1 UP(128)
        out = self.UB1_U3_CB(out)
        out = torch.cat([out, DB4], dim=1)
        out = self.UB1_CB_CB(out)

        #Level 2 UP(64)
        out = self.UB2_U3_CB(out)
        out = torch.cat([out, DB3], dim=1)
        out = self.UB2_CB_CB(out)

        #upscale 1
        upscale1 = self.C3(out)
        upscale1 = self.U3(upscale1)

        #Level 3 UP(32)
        out = self.UB3_U3_CB(out)
        out = torch.cat([out, DB2], dim=1)
        out = self.UB3_CB_CB(out)

        #upscale 2
        upscale2 = self.C3(out)
        upscale2 += upscale1
        upscale2 = self.U3(upscale2) 

        #Level 4 UP(16)
        out = self.UB4_U3_CB(out)
        out = torch.cat([out, DB1], dim=1)
        out = self.UB4_CB_CB(out)
        
        #upscale 3
        out = self.C3(out)
        out += upscale2

        #activation
        out = nn.Sigmoid(out)

        return out


# input = torch.rand(5,4)

# m = test_joey(8)
# print(m)