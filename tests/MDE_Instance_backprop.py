import pytest
import numpy as np
from devito import configuration
import joey
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import grad



def generate_random_input(input_size) -> tuple:
    '''generate random data for test'''

    input_data = \
        torch.randn(input_size, dtype=torch.double, requires_grad=True)

    return input_data

'''
grad_res is the incoming DL/DY
'''
def np_backprop_eq(input_numpy, grad_res, outputs):
    # computing var & mean using input data : simulating forward_pass
    N = np.prod(input_numpy.shape)
    mean = np.sum(input_numpy)/N
    input_mean = input_numpy - mean
    var = np.sum(input_mean*input_mean)/N
    var= var+0.00001
    var_sqrt = np.sqrt(var)

    # backprop equations
    inv_no_ofelements = 1.0 /N
    grad_sigma   = np.sum( grad_res *  (input_numpy-mean)   ) * -0.5 * (var) ** -1.5
    grad_mean    = np.sum( grad_res *  (-1./var_sqrt)) + grad_sigma * inv_no_ofelements * 2.0 * np.sum((input_numpy-mean)) * -1
    grad_x       = grad_res * 1/(var_sqrt) + grad_sigma * inv_no_ofelements * 2.0 * (input_numpy-mean) + grad_mean * inv_no_ofelements
    return grad_x

input_data = generate_random_input((1,1,5,5))

criterion = nn.MSELoss()

#  torch instance norm
torch_instance_op = torch.nn.InstanceNorm2d(input_data.shape[1])
outputs = torch_instance_op(input_data)
exp_res = torch.randn(outputs.shape, dtype=torch.double)
# any loss function for autograd
loss = criterion(outputs,exp_res)
# DL/DY
res_grad = grad(outputs=loss, inputs=outputs, allow_unused=True,
                        retain_graph=True)[0].detach().numpy()

#DL/DX
result_torch = grad(outputs=loss, inputs=input_data, allow_unused=True,
                        retain_graph=True)
   
result_np = np_backprop_eq(input_data.detach().numpy(), res_grad, outputs.detach().numpy())

print("Do they match" ,np.allclose(result_torch[0].detach().numpy(), result_np))
