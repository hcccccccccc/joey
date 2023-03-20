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
    def __init__(self, in_channel = 4, filter = 16, kernel_size = (3,3,3)):
        super(test_joey, self).__init__()
        self.in_channel = in_channel
        self.filter = filter
        self.kernel_size = kernel_size

        # Downward block
        # self.DB1_CB = self.CB(in_channel, self.filter, stride=1, k_size=3)
        self.DB1_conv1 = joey.Conv3D(in_channel, filter, kernel_size=(3,3,3), stride=1, padding=1, bias=False),
        self.DB1_inst1 = joey.InstanceNorm3d(filter),
        self.DB1_lrelu1 = joey.LeakyReLU()

        # self.DB1 = self.DB(self.filter, self.filter, stride=1)
        self.DB1_conv2 = joey.Conv3D(filter, filter, kernel_size=self.kernel_size, stride=1, padding=1, bias=False),
        self.DB1_inst2 = joey.InstanceNorm3d(filter),
        self.DB1_lrelu2 = joey.LeakyReLU()

        self.DB1_drop1 = nn.Dropout3d()

        self.DB1_conv3 = joey.Conv3D(filter, filter, kernel_size=self.kernel_size, stride=1, padding=1, bias=False),
        self.DB1_inst3 = joey.InstanceNorm3d(filter),
        self.DB1_lrelu3 = joey.LeakyReLU()

        # self.DB2_CB = self.CB(self.filter, self.filter*2, stride=2, k_size=3)
        self.DB2_conv1 = joey.Conv3D(filter, filter*2, kernel_size=self.kernel_size, stride=2, padding=1, bias=False),
        self.DB2_inst1 = joey.InstanceNorm3d(filter*2),
        self.DB2_lrelu1 = joey.LeakyReLU()
        # self.DB2 = self.DB(self.filter*2, self.filter*2, stride=1)
        self.DB2_conv2 = joey.Conv3D(filter*2, filter*2, kernel_size=self.kernel_size, stride=1, padding=1, bias=False),
        self.DB2_inst2 = joey.InstanceNorm3d(filter*2),
        self.DB2_lrelu2 = joey.LeakyReLU()

        self.DB2_drop1 = nn.Dropout3d()

        self.DB2_conv3 = joey.Conv3D(filter*2, filter*2, kernel_size=self.kernel_size, stride=1, padding=1, bias=False),
        self.DB2_inst3 = joey.InstanceNorm3d(filter*2),
        self.DB2_lrelu3 = joey.LeakyReLU()

        # self.DB3_CB = self.CB(self.filter*2, self.filter*4, stride=2, k_size=3)
        self.DB3_conv1 = joey.Conv3D(filter*2, filter*4, kernel_size=self.kernel_size, stride=2, padding=1, bias=False),
        self.DB3_inst1 = joey.InstanceNorm3d(filter*4),
        self.DB3_lrelu1 = joey.LeakyReLU()
        # self.DB3 = self.DB(self.filter*4, self.filter*4, stride=1)
        self.DB3_conv2 = joey.Conv3D(filter*4, filter*4, kernel_size=self.kernel_size, stride=1, padding=1, bias=False),
        self.DB3_inst2 = joey.InstanceNorm3d(filter*4),
        self.DB3_lrelu2 = joey.LeakyReLU()

        self.DB3_drop1 = nn.Dropout3d()

        self.DB3_conv3 = joey.Conv3D(filter*4, filter*4, kernel_size=self.kernel_size, stride=1, padding=1, bias=False),
        self.DB3_inst3 = joey.InstanceNorm3d(filter*4),
        self.DB3_lrelu3 = joey.LeakyReLU()
        # self.DB4_CB = self.CB(self.filter*4, self.filter*8, stride=2, k_size=3)
        self.DB4_conv1 = joey.Conv3D(filter*4, filter*8, kernel_size=self.kernel_size, stride=2, padding=1, bias=False),
        self.DB4_inst1 = joey.InstanceNorm3d(filter*8),
        self.DB4_lrelu1 = joey.LeakyReLU()
        # self.DB4 = self.DB(self.filter*8, self.filter*8, stride=1)
        self.DB4_conv2 = joey.Conv3D(filter*8, filter*8, kernel_size=self.kernel_size, stride=1, padding=1, bias=False),
        self.DB4_inst2 = joey.InstanceNorm3d(filter*8),
        self.DB4_lrelu2 = joey.LeakyReLU()

        self.DB4_drop1 = nn.Dropout3d()

        self.DB4_conv3 = joey.Conv3D(filter*8, filter*8, kernel_size=self.kernel_size, stride=1, padding=1, bias=False),
        self.DB4_inst3 = joey.InstanceNorm3d(filter*8),
        self.DB4_lrelu3 = joey.LeakyReLU()

        # self.DB5_CB = self.CB(self.filter*8, self.filter*16, stride=2, k_size=3)
        self.DB5_conv1 = joey.Conv3D(filter*8, filter*16, kernel_size=self.kernel_size, stride=2, padding=1, bias=False),
        self.DB5_inst1 = joey.InstanceNorm3d(filter*16),
        self.DB5_lrelu1 = joey.LeakyReLU()
        # self.DB5 = self.DB(self.filter*16, self.filter*16, stride=1)
        self.DB5_conv2 = joey.Conv3D(filter*16, filter*16, kernel_size=self.kernel_size, stride=1, padding=1, bias=False),
        self.DB5_inst2 = joey.InstanceNorm3d(filter*16),
        self.DB5_lrelu2 = joey.LeakyReLU()

        self.DB5_drop1 = nn.Dropout3d()

        self.DB5_conv3 = joey.Conv3D(filter*16, filter*16, kernel_size=self.kernel_size, stride=1, padding=1, bias=False),
        self.DB5_inst3 = joey.InstanceNorm3d(filter*16),
        self.DB5_lrelu3 = joey.LeakyReLU()



        # Upward Block
        # self.UB1_U3_CB = self.UB_U3_CB(self.filter*16, self.filter*8)
        self.UB1_U3 = self.U3(self.filter*16)

        self.UB1_conv1 = joey.Conv3D(filter*16, filter*8, kernel_size=self.kernel_size, stride=1, padding=1, bias=False),
        self.UB1_inst1 = joey.InstanceNorm3d(filter*8),
        self.UB1_lrelu1 = joey.LeakyReLU()

        # self.UB1_CB_CB = self.UB_CB_CB(self.filter*16, self.filter*8)
        self.UB1_conv2 = joey.Conv3D(filter*16, filter*8, kernel_size=self.kernel_size, stride=1, padding=0, bias=False),
        self.UB1_inst2 = joey.InstanceNorm3d(filter*8),
        self.UB1_lrelu2 = joey.LeakyReLU()

        self.UB1_conv3 = joey.Conv3D(filter*8, filter*8, kernel_size=self.kernel_size, stride=1, padding=0, bias=False),
        self.UB1_inst3 = joey.InstanceNorm3d(filter*8),
        self.UB1_lrelu3 = joey.LeakyReLU()

        # self.UB2_U3_CB = self.UB_U3_CB(self.filter*8, self.filter*4)
        self.UB2_U3 = self.U3(self.filter*8)

        self.UB2_conv1 = joey.Conv3D(filter*8, filter*4, kernel_size=self.kernel_size, stride=1, padding=1, bias=False),
        self.UB2_inst1 = joey.InstanceNorm3d(filter*4),
        self.UB2_lrelu1 = joey.LeakyReLU()

        # self.UB2_CB_CB = self.UB_CB_CB(self.filter*8, self.filter*4)
        self.UB2_conv2 = joey.Conv3D(filter*8, filter*4, kernel_size=self.kernel_size, stride=1, padding=0, bias=False),
        self.UB2_inst2 = joey.InstanceNorm3d(filter*4),
        self.UB2_lrelu2 = joey.LeakyReLU()

        self.UB2_conv3 = joey.Conv3D(filter*4, filter*4, kernel_size=self.kernel_size, stride=1, padding=0, bias=False),
        self.UB2_inst3 = joey.InstanceNorm3d(filter*4),
        self.UB2_lrelu3 = joey.LeakyReLU()

        # self.UB3_U3_CB = self.UB_U3_CB(self.filter*4, self.filter*2)
        self.UB3_U3 = self.U3(self.filter*4)

        self.UB3_conv1 = joey.Conv3D(filter*4, filter*2, kernel_size=self.kernel_size, stride=1, padding=1, bias=False),
        self.UB3_inst1 = joey.InstanceNorm3d(filter*2),
        self.UB3_lrelu1 = joey.LeakyReLU()
        # self.UB3_CB_CB = self.UB_CB_CB(self.filter*4, self.filter*2)
        self.UB3_conv2 = joey.Conv3D(filter*4, filter*2, kernel_size=self.kernel_size, stride=1, padding=0, bias=False),
        self.UB3_inst2 = joey.InstanceNorm3d(filter*2),
        self.UB3_lrelu2 = joey.LeakyReLU()

        self.UB3_conv3 = joey.Conv3D(filter*2, filter*2, kernel_size=self.kernel_size, stride=1, padding=0, bias=False),
        self.UB3_inst3 = joey.InstanceNorm3d(filter*2),
        self.UB3_lrelu3 = joey.LeakyReLU()
        # self.UB4_U3_CB = self.UB_U3_CB(self.filter*2, self.filter)
        self.UB4_U3 = self.U3(self.filter*2)

        self.UB4_conv1 = joey.Conv3D(filter*2, filter, kernel_size=self.kernel_size, stride=1, padding=1, bias=False),
        self.UB4_inst1 = joey.InstanceNorm3d(filter),
        self.UB4_lrelu1 = joey.LeakyReLU()
        # self.UB4_CB_CB = self.UB_CB_CB(self.filter*2, self.filter)
        self.UB4_conv2 = joey.Conv3D(filter*2, filter, kernel_size=self.kernel_size, stride=1, padding=0, bias=False),
        self.UB4_inst2 = joey.InstanceNorm3d(filter),
        self.UB4_lrelu2 = joey.LeakyReLU()

        self.UB4_conv3 = joey.Conv3D(filter, filter, kernel_size=self.kernel_size, stride=1, padding=0, bias=False),
        self.UB4_inst3 = joey.InstanceNorm3d(filter),
        self.UB4_lrelu3 = joey.LeakyReLU()

        self.C3_1 = self.C3(self.filter*4)
        self.C3_2 = self.C3(self.filter*2)
        self.C3_3 = self.C3(self.filter)
        self.U3_1 = self.U3()
        self.U3_2 = self.U3()


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
        out = self.DB1_conv1(input)
        out = self.DB1_inst1(out)
        out = self.DB1_lrelu1(out)
        tempout1 = out
        out = self.DB1_conv2(out)
        out = self.DB1_inst2(out)
        out = self.DB1_lrelu2(out)
        out = self.DB1_drop1(out)
        out = self.DB1_conv3(out)
        out = self.DB1_inst3(out)
        out = self.DB1_lrelu3(out)
        out += tempout1
        DB1 = out

        #Level 2 DB(32,2)
        out = self.DB2_conv1(out)
        out = self.DB2_inst1(out)
        out = self.DB2_lrelu1(out)
        tempout2 = out
        out = self.DB2_conv2(out)
        out = self.DB2_inst2(out)
        out = self.DB2_lrelu2(out)
        out = self.DB2_drop1(out)
        out = self.DB2_conv3(out)
        out = self.DB2_inst3(out)
        out = self.DB2_lrelu3(out)
        out += tempout2
        DB2 = out

        #Level 3 DB(64,2)
        out = self.DB3_conv1(out)
        out = self.DB3_inst1(out)
        out = self.DB3_lrelu1(out)
        tempout3 = out
        out = self.DB3_conv2(out)
        out = self.DB3_inst2(out)
        out = self.DB3_lrelu2(out)
        out = self.DB3_drop1(out)
        out = self.DB3_conv3(out)
        out = self.DB3_inst3(out)
        out = self.DB3_lrelu3(out)
        out += tempout3
        DB3 = out

        # #Level 4 DB(128,2)
        # out = self.DB4_CB(out)
        # tempout4 = out
        # out = self.DB4(out)
        # out += tempout4
        # DB4 = out
        
        # #Level 5 DB(256,2)
        # out = self.DB5_CB(out)
        # tempout5 = out
        # out = self.DB5(out)
        # out += tempout5
        
        # #Level 1 UP(128)
        # out = self.UB1_U3_CB(out)
        # out = torch.cat([out, DB4], dim=1)
        # out = self.UB1_CB_CB(out)

        # #Level 2 UP(64)
        # out = self.UB2_U3_CB(out)
        # out = torch.cat([out, DB3], dim=1)
        # out = self.UB2_CB_CB(out)

        # #upscale 1
        # upscale1 = self.C3(out)
        # upscale1 = self.U3(upscale1)

        # #Level 3 UP(32)
        # out = self.UB3_U3_CB(out)
        # out = torch.cat([out, DB2], dim=1)
        # out = self.UB3_CB_CB(out)

        # #upscale 2
        # upscale2 = self.C3(out)
        # upscale2 += upscale1
        # upscale2 = self.U3(upscale2) 

        # #Level 4 UP(16)
        # out = self.UB4_U3_CB(out)
        # out = torch.cat([out, DB1], dim=1)
        # out = self.UB4_CB_CB(out)
        
        # #upscale 3
        # out = self.C3(out)
        # out += upscale2

        # #activation
        # out = nn.Sigmoid(out)

        return out


input = torch.rand(5,4)

m = test_joey(8)
print(m)