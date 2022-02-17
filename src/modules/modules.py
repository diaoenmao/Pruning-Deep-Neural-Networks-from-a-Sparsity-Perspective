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
        sparsity_index[k] = torch.linalg.norm(v, 1, dim=-1) / torch.linalg.norm(v, q, dim=-1)
    return sparsity_index
