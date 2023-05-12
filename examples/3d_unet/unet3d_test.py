import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split


class test_pytorch(nn.Module):
    def __init__(self, in_channel = 4, filter = 16):
        super(test_pytorch, self).__init__()
        self.in_channel = in_channel
        self.filter = filter

        # Downward block
        self.DB1_CB = self.CB(in_channel=in_channel, filter=self.filter, stride=(1,1,1),kernel_size=(3,3,3))
        self.DB1 = self.DB(in_channel=self.filter, filter=self.filter)
        self.DB2_CB = self.CB(in_channel=self.filter, filter=self.filter*2, stride=(2,2,2), kernel_size=(3,3,3))
        self.DB2 = self.DB(in_channel=self.filter*2, filter=self.filter*2)
        self.DB3_CB = self.CB(in_channel=self.filter*2, filter=self.filter*4, stride=(2,2,2), kernel_size=(3,3,3))
        self.DB3 = self.DB(in_channel=self.filter*4, filter=self.filter*4)
        self.DB4_CB = self.CB(in_channel=self.filter*4, filter=self.filter*8, stride=(2,2,2), kernel_size=(3,3,3))
        self.DB4 = self.DB(in_channel=self.filter*8, filter=self.filter*8)
        self.DB5_CB = self.CB(in_channel=self.filter*8, filter=self.filter*16, stride=(2,2,2), kernel_size=(3,3,3))
        self.DB5 = self.DB(in_channel=self.filter*16, filter=self.filter*16)



        # Upward Block
        self.UB1_U3_CB = self.UB_U3_CB(self.filter*16, self.filter*8)
        self.UB1_CB_CB = self.UB_CB_CB(self.filter*16, self.filter*8)
        self.UB2_U3_CB = self.UB_U3_CB(self.filter*8, self.filter*4)
        self.UB2_CB_CB = self.UB_CB_CB(self.filter*8, self.filter*4)
        self.UB3_U3_CB = self.UB_U3_CB(self.filter*4, self.filter*2)
        self.UB3_CB_CB = self.UB_CB_CB(self.filter*4, self.filter*2)
        self.UB4_U3_CB = self.UB_U3_CB(self.filter*2, self.filter)
        self.UB4_CB_CB = self.UB_CB_CB(self.filter*2, self.filter)
        self.C3_1 = self.C3(in_channel=self.filter*4)
        self.C3_2 = self.C3(in_channel=self.filter*2)
        self.C3_3 = self.C3(in_channel=self.filter)
        self.U3_1 = self.U3()
        self.U3_2 = self.U3()
        self.sigmoid = nn.Sigmoid()

    def CB(self, in_channel, filter, stride=(1,1,1), kernel_size=(3,3,3),padding=1):
        return nn.Sequential(nn.Conv3d(in_channel, filter, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                             nn.InstanceNorm3d(filter),
                             nn.LeakyReLU())

    def C3(self, in_channel,filter=3):
        return nn.Conv3d(in_channel, filter, kernel_size=(1,1,1), stride=(1,1,1), padding=0, bias=False)
    
    def DB(self, in_channel, filter, stride=(1,1,1)):
        return nn.Sequential(self.CB(in_channel, filter, stride),
                             self.CB(filter, filter))
    
    def U3(self):
        return nn.Upsample(scale_factor=2, mode='nearest')


    
    def UB_U3_CB(self, in_channel, filter):
        return nn.Sequential(self.U3(),
                             self.CB(in_channel, filter))

    def UB_CB_CB(self, in_channel, filter):
        return nn.Sequential(self.CB(in_channel, filter),
                             self.CB(filter, filter, kernel_size=(1,1,1),padding=0))

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

        # #upscale 1
        # upscale1 = self.C3_1(out)
        # upscale1 = self.U3_1(upscale1)

        #Level 3 UP(32)
        out = self.UB3_U3_CB(out)
        out = torch.cat([out, DB2], dim=1)
        out = self.UB3_CB_CB(out)

        # #upscale 2
        # upscale2 = self.C3_2(out)
        # upscale2 += upscale1
        # upscale2 = self.U3_2(upscale2) 

        #Level 4 UP(16)
        out = self.UB4_U3_CB(out)
        out = torch.cat([out, DB1], dim=1)
        out = self.UB4_CB_CB(out)
        
        #upscale 3
        out = self.C3_3(out)
        # out += upscale2

        #activation
        out = self.sigmoid(out)

        return out
