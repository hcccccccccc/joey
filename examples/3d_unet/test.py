import joey
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
from devito import Function, Grid, SpaceDimension
import numpy as np
from devito import logger
# def DB(input_size, kernel_size, stride=(1,1,1)):
#     CB1 = joey.Conv3D(kernel_size=kernel_size, input_size=input_size, stride=stride, padding=(1,1,1), activation=joey.activation.LeakyReLU(),strict_stride_check=False)
#     inst = joey.InstanceNorm3D(input_size=input_size)
#     connect = joey.add(input_size=input_size, layer=CB1)
#     DB = [CB1, inst, connect]
#     return DB

# def unet_joey(batch_size, in_channel, depth, height, weight, filter):
#     DB1 = DB(input_size=(batch_size, in_channel, depth, height, weight), kernel_size=(filter,3,3,3))
#     DB2 = DB(input_size=(batch_size, filter, depth, height, weight), kernel_size=(filter*2,3,3,3))
#     DB3 = DB(input_size=(batch_size, filter*2, depth, height, weight), kernel_size=(filter*4,3,3,3))
#     DB1 = DB1 + DB2 + DB3
#     return(joey.Net(DB1), DB1)
def dice_loss(layer, target):
    eps = 0.00001
    pred = layer.result.data
    target = target.numpy()
        # pred = nn.functional.sigmoid(pred)
        # dice_loss = criterion(pred, expected_results)
        # pred = 1/(1+(np.exp(-pred)))



        # result = (2*target[:,:,:,:,:]*pred[:,:,:,:,:]- 2*target[:,:,:,:,:]*(pred[:,:,:,:,:] + target[:,:,:,:,:] + eps) + eps)
        # result = result/((target[:,:,:,:,:]+pred[:,:,:,:,:]+eps)**2)
    result = ((2*target[:,:,:,:,:]*pred[:,:,:,:,:]+eps)/(target[:,:,:,:,:]+pred[:,:,:,:,:]+ eps))
        # result *= pred[:,:,:,:,:]*(1-pred[:,:,:,:,:])

        # print(batch.sum(dims=(2,3,4)))
    return result

# size = (1,1,1,32,32)
# start = time.time()
# print(start)
# model, net = unet_joey(2,4,128,128,128, 16)
# finish = time.time()-start
# print("Time spend: {:.0f}m {:.0f}s".format(finish // 60, finish % 60))
# print(model)

size = (1,2,1,3,3)
expected_size = (1,2,2,6,6)
input = torch.rand(size)
expected = torch.rand(expected_size)
print(input)
inst = joey.InstanceNorm3D(input_size=(size))
inst2 = nn.InstanceNorm3d(2)
add = joey.UpSample(input_size= size, scale_factor=2)
net = joey.Net([inst, add])

net.forward(input)
net.backward(expected, dice_loss)
print(net)

# input = torch.rand(size)
# print(input)
# input_numpy = a.detach().numpy()
# inst = joey.InstanceNorm3D(input_size=(size), generate_code=True)
# conv = joey.Conv3D(kernel_size=(4,3,3,3), input_size=size, stride=(1,1,1), padding=(0,0,0), activation=joey.activation.ReLU(),strict_stride_check=False, generate_code=True)
# add = joey.add(input_size=size, layer=inst, generate_code=True)
# out1 = inst.execute(input)
# print(out1)
# out2 = add.execute(input)
# print(out2)
# inst2 = nn.InstanceNorm3d(2)
# out2 = inst2(input)
# print(out2)

# transform = transforms.Compose(
#     [transforms.Resize((32, 32)),
#      transforms.ToTensor(),
#      transforms.Normalize(0.5, 0.5)])
# trainset = torchvision.datasets.MNIST(root='./mnist', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False, num_workers=2)

# classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# batch_size=4
# logger.set_log_noperf()
# def create_lenet():
#     # Six 3x3 filters, activation RELU
#     layer1 = ml.Conv(kernel_size=(6, 3, 3),
#                      input_size=(batch_size, 1, 32, 32),
#                      activation=ml.activation.ReLU())
#     # Max 2x2 subsampling
#     layer2 = ml.MaxPooling(kernel_size=(2, 2),
#                            input_size=(batch_size, 6, 30, 30),
#                            stride=(2, 2))
#     # Sixteen 3x3 filters, activation RELU
#     layer3 = ml.Conv(kernel_size=(16, 3, 3),
#                      input_size=(batch_size, 6, 15, 15),
#                      activation=ml.activation.ReLU())
#     # Max 2x2 subsampling
#     layer4 = ml.MaxPooling(kernel_size=(2, 2),
#                            input_size=(batch_size, 16, 13, 13),
#                            stride=(2, 2),
#                            strict_stride_check=False)
#     # Full connection (16 * 6 * 6 -> 120), activation RELU
#     layer5 = ml.FullyConnected(weight_size=(120, 576),
#                                input_size=(576, batch_size),
#                                activation=ml.activation.ReLU())
#     # Full connection (120 -> 84), activation RELU
#     layer6 = ml.FullyConnected(weight_size=(84, 120),
#                                input_size=(120, batch_size),
#                                activation=ml.activation.ReLU())
#     # Full connection (84 -> 10), output layer
#     layer7 = ml.FullyConnected(weight_size=(10, 84),
#                                       input_size=(84, batch_size))
#     # Flattening layer necessary between layer 4 and 5
#     layer_flat = ml.Flat(input_size=(batch_size, 16, 6, 6))
    
#     layers = [layer1, layer2, layer3, layer4,
#               layer_flat, layer5, layer6, layer7]
    
#     return (ml.Net(layers), layers)


# def train(net, input_data, expected_results, pytorch_optimizer):
#     outputs = net.forward(input_data)
    
#     def loss_grad(layer, expected):
#         gradients = []
        
#         for b in range(len(expected)):
#             row = []
            
#             for i in range(10):
#                 result = layer.result.data[i, b]
#                 if i == expected[b]:
#                     result -= 1
#                 row.append(result)
            
#             gradients.append(row)
        
#         return gradients
    
#     net.backward(expected_results, loss_grad, pytorch_optimizer)

# def dice_loss(prediction, target):
#         intersection = (prediction * target)
#         union = (prediction + target)
#         dice = 2 * (intersection ) / (union )
#         loss_gradient = 1 - dice.mean()
#         return loss_gradient

# devito_net, devito_layers = create_lenet()
# optimizer = optim.SGD(devito_net.pytorch_parameters, lr=0.001, momentum=0.9)

# for i, data in enumerate(trainloader, 0):
#     images, labels = data
#     images.double()
    
#     train(devito_net, images, labels, optimizer)
