import copy
import datetime
import time
import sys
import torch
import models
from collections import OrderedDict
from config import cfg
from utils import to_device, collate, make_optimizer, make_scheduler


def make_sparsity_index(model, q):
    sparsity_index = OrderedDict()
    for k, v in model.state_dict().items():
        if 'weight' in k:
            sparsity_index[k] = torch.linalg.norm(v, 1, dim=-1) / torch.linalg.norm(v, q, dim=-1)
    return sparsity_index


class Compression:
    def __init__(self, model):
        self.init_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        self.mask = [make_mask(init_model_state_dict)]
        self.prune_percent = cfg['prune_percent']

    def make_mask(self, model_state_dict):
        mask = OrderedDict()
        for name, param in model_state_dict.items():
            parameter_type = name.split('.')[-1]
            if 'weight' in parameter_type:
                mask[name] = param.new_ones(param.size())
        return mask

    def prune(self, model):
        new_mask = OrderedDict()
        for name, param in model.named_parameters():
            parameter_type = name.split('.')[-1]
            if 'weight' in parameter_type:
                masked_param = param[mask]
                pivot_param = masked_param.abs()
                percentile_value = torch.quantile(pivot_param, percent)
                new_mask[name] = torch.where(param.data.abs() < percentile_value, 0, mask)
                param.data = copy.deepcopy((param.data * new_mask[name]).detach())
        self.mask.append(new_mask)
        return

    def init(self, model):
        for name, param in model.named_parameters():
            parameter_type = name.split('.')[-1]
            if 'weight' in parameter_type:
                mask = self.mask[-1][name]
                param.data = copy.deepcopy((mask * self.init_model_state_dict[name]).detach())
            if "bias" in parameter_type:
                param.data = copy.deepcopy(self.init_model_state_dict[name])
        return
