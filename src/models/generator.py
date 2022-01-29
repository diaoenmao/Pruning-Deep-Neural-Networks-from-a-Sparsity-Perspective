import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import init_param_generator
from config import cfg


class Generator(nn.Module):
    def __init__(self, data_shape, init_shape, latent_size, hidden_size):
        super().__init__()
        self.init_shape = init_shape
        self.linear = nn.Sequential(nn.Linear(latent_size, 128 * init_shape[0] * init_shape[1]))

        self.blocks = nn.Sequential(
            nn.BatchNorm2d(hidden_size[0]),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(hidden_size[0], hidden_size[0], 3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_size[0], 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(hidden_size[0], hidden_size[1], 3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_size[1], 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_size[1], data_shape[0], 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(data_shape[0], affine=False)
        )

    def forward(self, input):
        output = {}
        x = input['z']
        x = self.linear(x)
        x = x.view(x.size(0), -1, self.init_shape[0], self.init_shape[1])
        x = self.blocks(x)
        output['x'] = x
        return output


def generator():
    data_shape = cfg['data_shape']
    init_shape = cfg['init_shape']
    latent_size = cfg['generator']['latent_size']
    hidden_size = cfg['generator']['hidden_size']
    model = Generator(data_shape, init_shape, latent_size, hidden_size)
    model.apply(init_param_generator)
    return model