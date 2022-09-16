import torch
import torch.nn as nn
import math
from config import cfg
from .utils import init_param, loss_fn, make_loss


class Linear(nn.Module):
    def __init__(self, data_shape, target_size):
        super().__init__()
        input_size = math.prod(data_shape)
        self.linear = nn.Linear(input_size, target_size)

    def feature(self, x):
        x = x.reshape(x.size(0), -1)
        return x

    def classify(self, x):
        x = self.linear(x)
        return x

    def f(self, x):
        x = self.feature(x)
        x = self.classify(x)
        return x

    def forward(self, input):
        output = {}
        x = input['data']
        x = self.f(x)
        output['target'] = x
        output['loss'] = make_loss(output, input)
        return output


def linear():
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    model = Linear(data_shape, target_size)
    model.apply(init_param)
    return model
