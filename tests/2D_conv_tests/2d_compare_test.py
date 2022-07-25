from asyncio import FastChildWatcher
import pytest
from pyTorch_2Dconv import pyTorch_conv
from Devito_Conv_2d import conv_devito_2D
import numpy as np
import joey

custom_kernel = [[0,0,0] ,[1,0,-1],[0,0,0]]

@pytest.mark.parametrize("input_size, kernel, padding", [(5, custom_kernel, 2),(50, custom_kernel, 6) ])
def test_Devito_pyTorch_padding(input_size, kernel, padding, print_results = False ):
    c=0;
    input =[]
    for i in range(0,input_size):
        temp =[]
        for j in range(0,input_size):
            temp.append(c)
            c = c+1;
        input.append(temp)

 #to simulate kernel, image with channel 1
    kernel = [[kernel]]
    input =[input]

    result_devito = conv_devito_2D(input, kernel, padding,1, print_code = print_results)
    result_torch = pyTorch_conv(input, kernel, padding,1)

    if print_results:
        print ("torch",result_torch[0])

        print ("devito",result_devito)


    assert(np.allclose(result_devito, result_torch))




@pytest.mark.parametrize("input_size, kernel, stride", [(5, custom_kernel, 2),(50, custom_kernel, 5) ])
def test_Devito_pyTorch_stride(input_size, kernel, stride, print_results = False ):
    c=0;
    input =[]
    for i in range(0,input_size):
        temp =[]
        for j in range(0,input_size):
            temp.append(c)
            c = c+1;
        input.append(temp)

    #to simulate kernel, image with channel 1
    kernel = [[kernel]]
    input =[input]

    result_devito = conv_devito_2D(input, kernel, 1, stride, print_code = print_results)

    result_torch = pyTorch_conv(input, kernel, 1, stride)

    if print_results:
        print ("torch",result_torch[0])

        print ("devito",result_devito)

    assert(np.allclose(result_devito, result_torch))




custom_kernel1 = [[0.5,0.5,0.5,0.5,0.5] ,[1,0,-1,0.5,0.5],[0.5,0.5,0.5,0.5,0.5],[1,0,-1,0.5,0.5],[0.5,0.5,0.5,0.5,0.5]]

@pytest.mark.parametrize("input_size, kernel, channels, no_kernels, padding, stride", [(5, custom_kernel, 3, 3, 1, 1),(50, custom_kernel1, 2,2, 6, 4) ])
def test_Devito_pyTorch(input_size,kernel,channels, no_kernels, padding, stride, print_results = False):
    
    c=1;
    input =[]
    for i in range(0,input_size):
        temp =[]
        for j in range(0,input_size):
            temp.append(c)
            c = c+1;
        input.append(temp)   
    #to simulate kernel, image with channel as specified
    channel_kernel = []
    channel_input =[]
    for i in range(0,channels):
        channel_kernel.append(kernel)
        channel_input.append(input)

    kernel = channel_kernel
    channel_kernel = []

    for i in range(0,no_kernels):
        channel_kernel.append(kernel)

    channel_input = [channel_input, channel_input, channel_input]

    result_torch = pyTorch_conv(channel_input, channel_kernel, padding, stride)

    result_devito = conv_devito_2D(channel_input, channel_kernel, padding, stride, print_code = print_results)
    if print_results:
        print ("torch",result_torch)

        print ("devito",result_devito)


    print("Do they match", np.allclose(result_devito, result_torch))

    #assert(np.allclose(result_devito, result_torch))


@pytest.mark.parametrize("input_size, kernel, channels, no_kernels, padding, stride", 
[((3,5), custom_kernel, 3, 3, 1, 1),((5,50), custom_kernel1, 2,2, 6, 3), ((5,35), custom_kernel1,5,3, 10, 2) ])
def test_Joey_pyTorch_Conv(input_size,kernel,channels, no_kernels, padding, stride, print_results = False):
    
    c=1;
    input =[]
    for i in range(0,input_size[1]):
        temp =[]
        for j in range(0,input_size[1]):
            temp.append(c)
            c = c+1;
        input.append(temp)   
    #to simulate kernel, image with channel as specified
    channel_kernel = []
    channel_input =[]
    for i in range(0,channels):
        channel_kernel.append(kernel)
        channel_input.append(input)

    channel_kernel = [channel_kernel for x in range(0,no_kernels)]

    channel_input = [channel_input for x in range(0,input_size[0])]

    result_torch = pyTorch_conv(channel_input, channel_kernel, padding, stride)

    layer = joey.Conv_v2(kernel_size=(no_kernels, len(channel_kernel[0][0]), len(channel_kernel[0][0])), 
            input_size=(input_size[0], len(channel_kernel[0]), input_size[1], input_size[1]), padding= (padding,padding)
            ,stride=(stride,stride),generate_code=True)

    result_joey = layer.execute(channel_input,[0]*len(channel_kernel), channel_kernel)
    if print_results:
        print ("torch",result_torch)

        print ("joey",result_joey)


    print("Do they match", np.allclose(result_joey, result_torch))

    #assert(np.allclose(result_devito, result_torch))


#test_Joey_pyTorch_Conv((2,5), custom_kernel,6,5, 4, 2, True)