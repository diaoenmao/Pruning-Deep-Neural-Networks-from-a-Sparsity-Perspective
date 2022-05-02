import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from utils import load


def init_param(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        if m.bias is not None:
            m.bias.data.zero_()
    return m


def normalize(input):
    if cfg['data_name'] in cfg['stats']:
        broadcast_size = [1] * input.dim()
        broadcast_size[1] = input.size(1)
        m, s = cfg['stats'][cfg['data_name']]
        m, s = torch.tensor(m, dtype=input.dtype).view(broadcast_size).to(input.device), \
               torch.tensor(s, dtype=input.dtype).view(broadcast_size).to(input.device)
        input = input.sub(m).div(s)
    return input


def denormalize(input):
    if cfg['data_name'] in cfg['stats']:
        broadcast_size = [1] * input.dim()
        broadcast_size[1] = input.size(1)
        m, s = cfg['stats'][cfg['data_name']]
        m, s = torch.tensor(m, dtype=input.dtype).view(broadcast_size).to(input.device), \
               torch.tensor(s, dtype=input.dtype).view(broadcast_size).to(input.device)
        input = input.mul(s).add(m)
    return input


def loss_fn(output, target):
    if target.dtype == torch.int64:
        loss = F.cross_entropy(output, target)
    else:
        loss = F.mse_loss(output, target)
    return loss


def load_init_state_dict(seed):
    pivot_data_name_dict = {'MNIST': 'MNIST', 'FashionMNIST': 'MNIST', 'CIFAR10': 'CIFAR10', 'SVHN': 'CIFAR10'}
    pivot_data_name = pivot_data_name_dict[cfg['data_name']]
    control_name = [pivot_data_name, cfg['control']['model_name']]
    model_tag_list = [str(seed), *control_name]
    model_tag = '_'.join([x for x in model_tag_list if x])
    init_state_dict = load(os.path.join('output', 'init', '{}.pt'.format(model_tag)))['model_state_dict']
    return init_state_dict
