import joey
import torch
from torch import nn
from devito import Function, Grid, SpaceDimension
import numpy as np

def DB(input_size, kernel_size, stride=(1,1,1)):
    CB1 = joey.Conv3D(kernel_size=kernel_size, input_size=input_size, stride=stride, padding=(1,1,1), activation=joey.activation.LeakyReLU(),strict_stride_check=False)
    inst = joey.InstanceNorm3D(input_size=input_size)
    connect = joey.add(input_size=input_size, layer=CB1)
    DB = [CB1, inst, connect]
    return DB

def unet_joey(batch_size, in_channel, depth, height, weight, filter):
    DB1 = DB(input_size=(batch_size, in_channel, depth, height, weight), kernel_size=(filter,3,3,3))
    DB2 = DB(input_size=(batch_size, filter, depth, height, weight), kernel_size=(filter*2,3,3,3))
    DB3 = DB(input_size=(batch_size, filter*2, depth, height, weight), kernel_size=(filter*4,3,3,3))
    DB1 = DB1 + DB2 + DB3
    return(joey.Net(DB1), DB1)

# net, layer = unet_joey(2,4,4,4,4, 8)
# print(net)
# print(layer)


size = (4,8,64,64,64)
input = torch.rand(size)
# print(input)
inst = joey.InstanceNorm3D(input_size=(size), generate_code=True)
cat = joey.cat(input_size=(size), layer=inst, generate_code=True)
out1 = inst.execute(input)
# print(out1)
out2 = cat.execute(input)
# print(out2)
out3 = torch.cat([input, torch.tensor(out1)], dim=1)
print(out3-out2)

