import torch
import torch.nn as nn
from config import cfg
from .utils import init_param_classifier, loss_fn


class Conv(nn.Module):
    def __init__(self, data_shape, hidden_size, target_size):
        super().__init__()
        blocks = [nn.Conv2d(data_shape[0], hidden_size[0], 5, 1, 2),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(2),
                  nn.Conv2d(hidden_size[0], hidden_size[1], 5, 1, 2),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(2),
                  nn.Flatten(),
                  nn.Linear(64 * (data_shape[1] // 4) * (data_shape[2] // 4), 512),
                  nn.ReLU(inplace=True),
                  nn.Linear(512, target_size)]
        self.blocks = nn.Sequential(*blocks)

    def f(self, x):
        x = self.blocks(x)
        return x

    def forward(self, input):
        output = {}
        x = self.f(input['data'])
        output['target'] = x
        if 'target' in input:
            output['loss'] = loss_fn(output['target'], input['target'])
        return output


def conv():
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    hidden_size = cfg['conv']['hidden_size']
    model = Conv(data_shape, hidden_size, target_size)
    model.apply(init_param_classifier)
    return model