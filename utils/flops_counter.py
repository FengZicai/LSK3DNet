import numpy as np
import os

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from utils.sparse_core import Masking, CosineDecay #, LinearDecay
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import spconv.pytorch as spconv



# def print_model_param_nums(model=None):
#     if model == None:
#         model = torchvision.models.alexnet()
#     total = sum([(param!=0).sum() if len(param.size()) == 4 or len(param.size()) == 2 else 0 for name,param in model.named_parameters()])
#     print('  + Number of params: %.2f' % (total))

def print_model_param_nums(model):
    total = sum([(param!=0).sum() for _,param in model.named_parameters()]) / 1e6
    print('  + Number of params: %.2f M' % (total))

def count_model_param_flops(model=None, input=None, multiply_adds=True):

    list_conv=[]
    def conv_hook(self, input, output):
        batch_size = input[0].batch_size
        output_channels = output.features.shape[1]
        output_number = output.features.shape[0]

        # output_height, output_width, output_height = output.spatial_shape

        bias_ops = 1 if self.bias is not None else 0

        try:
            kernel_ops = self.weight.data.shape[1] * self.weight.data.shape[2] * self.weight.data.shape[3] * (self.in_channels / self.groups)

        # params = output_channels * (kernel_ops + bias_ops)

            num_weight_params = (self.weight.data != 0).float().sum()
            assert self.weight.numel() == kernel_ops * output_channels, "Not match"
        except: 
            kernel_ops = self.weight.data.shape[0] * self.weight.data.shape[1] * self.weight.data.shape[2] * (self.in_channels / self.groups)
            num_weight_params = (self.weight.data != 0).float().sum()            

        flops = (num_weight_params * (2 if multiply_adds else 1) + bias_ops * output_channels) * output_number * batch_size

        list_conv.append(flops)

    list_linear=[]
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn=[]
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu=[]
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    def foo(net):
        childrens = list(net.children())

        if not childrens:
            if isinstance(net, spconv.SubMConv3d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm1d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.LeakyReLU) or isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)

            return
        for c in childrens:
            foo(c)

    foo(model)
    out = model(input)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu))

    print('  + Number of FLOPs: %.2fG' % (total_flops / 1e9))

    return total_flops

if __name__ == '__main__':
   
    model = torchvision.models.alexnet().to(torch.device('cuda:0')) #resnet.build_resnet('resnet50', 'fanin')
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
    decay = CosineDecay(0.5, 1000 * (10))
    mask = Masking(optimizer, prune_mode='magnitude', prune_rate_decay=decay, growth_mode='random',
                   redistribution_mode='none', device=torch.device('cuda:0'), sparse_init='ERK', sparsity=0.4)
    mask.add_module(model)

    total_flops = count_model_param_flops(model=model)

