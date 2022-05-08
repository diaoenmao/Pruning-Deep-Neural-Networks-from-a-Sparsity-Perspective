import copy
import datetime
import time
import sys
import torch
import models
from collections import OrderedDict
from config import cfg
from utils import to_device, collate, make_optimizer, make_scheduler


class SparsityIndex:
    def __init__(self, q):
        self.q = q
        self.si = {'neuron': [], 'layer': [], 'global': []}
        self.sie = {'neuron': [], 'layer': [], 'global': []}

    def sparsity_index(self, x, q, dim):
        d = float(x.size(dim))
        si = (torch.linalg.norm(x, 1, dim=dim).pow(1) / d).pow(1) / \
             (torch.linalg.norm(x, q, dim=dim).pow(q) / d).pow(1 / q)
        ub = min(torch.finfo(x.dtype).max, d ** (1 / q - 1))
        si[si.isnan()] = ub
        return si

    def sparsity_index_excluded(self, x, mask, q, dim):
        d = mask.to(x.device).float().sum(dim=dim)
        si = (torch.linalg.norm(x, 1, dim=dim).pow(1) / d).pow(1) / \
             (torch.linalg.norm(x, q, dim=dim).pow(q) / d).pow(1 / q)
        return si

    def make_sparsity_index(self, model, mask=None):
        self.si['neuron'].append(self.make_sparsity_index_(model, 'neuron'))
        self.si['layer'].append(self.make_sparsity_index_(model, 'layer'))
        self.si['global'].append(self.make_sparsity_index_(model, 'global'))
        self.sie['neuron'].append(self.make_sparsity_index_excluded_(model, mask, 'neuron'))
        self.sie['layer'].append(self.make_sparsity_index_excluded_(model, mask, 'layer'))
        self.sie['global'].append(self.make_sparsity_index_excluded_(model, mask, 'global'))
        return

    def make_sparsity_index_(self, model, mode):
        if mode == 'neuron':
            si = []
            for i in range(len(self.q)):
                si_i = OrderedDict()
                for name, param in model.state_dict().items():
                    parameter_type = name.split('.')[-1]
                    if 'weight' in parameter_type and param.dim() > 1:
                        param = param.view(param.size(0), -1)
                        si_i[name] = self.sparsity_index(param, self.q[i], -1)
                si.append(si_i)
        elif mode == 'layer':
            si = []
            for i in range(len(self.q)):
                si_i = OrderedDict()
                for name, param in model.state_dict().items():
                    parameter_type = name.split('.')[-1]
                    if 'weight' in parameter_type and param.dim() > 1:
                        si_i[name] = self.sparsity_index(param.view(-1), self.q[i], -1)
                si.append(si_i)
        elif mode == 'global':
            si = []
            for i in range(len(self.q)):
                param_all = []
                for name, param in model.state_dict().items():
                    parameter_type = name.split('.')[-1]
                    if 'weight' in parameter_type and param.dim() > 1:
                        param_all.append(param.view(-1))
                param_all = torch.cat(param_all, dim=0)
                si_i = self.sparsity_index(param_all, self.q[i], -1)
                si.append(si_i)
        else:
            raise ValueError('Not valid mode')
        return si

    def make_sparsity_index_excluded_(self, model, mask, mode):
        if mask is None:
            return self.make_sparsity_index_(model, mode)
        if mode == 'neuron':
            si = []
            for i in range(len(self.q)):
                si_i = OrderedDict()
                for name, param in model.state_dict().items():
                    parameter_type = name.split('.')[-1]
                    if 'weight' in parameter_type and param.dim() > 1:
                        param = param.view(param.size(0), -1)
                        si_i[name] = self.sparsity_index_excluded(param, mask[name], self.q[i], -1)
                si.append(si_i)
        elif mode == 'layer':
            si = []
            for i in range(len(self.q)):
                si_i = OrderedDict()
                for name, param in model.state_dict().items():
                    parameter_type = name.split('.')[-1]
                    if 'weight' in parameter_type and param.dim() > 1:
                        si_i[name] = self.sparsity_index_excluded(param.view(-1), mask[name].view(-1), self.q[i], -1)
                si.append(si_i)
        elif mode == 'global':
            si = []
            for i in range(len(self.q)):
                param_all = []
                mask_all = []
                for name, param in model.state_dict().items():
                    parameter_type = name.split('.')[-1]
                    if 'weight' in parameter_type and param.dim() > 1:
                        param_all.append(param.view(-1))
                        mask_all.append(mask[name].view(-1))
                param_all = torch.cat(param_all, dim=0)
                mask_all = torch.cat(mask_all, dim=0)
                si_i = self.sparsity_index_excluded(param_all, mask_all, self.q[i], -1)
                si.append(si_i)
        else:
            raise ValueError('Not valid mode')
        return si


class Norm:
    def __init__(self, q):
        self.q = q
        self.norm = {'neuron': [], 'layer': [], 'global': []}

    def make_norm(self, model):
        self.norm['neuron'].append(self.make_norm_(model, 'neuron'))
        self.norm['layer'].append(self.make_norm_(model, 'layer'))
        self.norm['global'].append(self.make_norm_(model, 'global'))
        return

    def make_norm_(self, model, mode):
        if mode == 'neuron':
            norm = []
            for i in range(len(self.q)):
                norm_i = OrderedDict()
                for name, param in model.state_dict().items():
                    parameter_type = name.split('.')[-1]
                    if 'weight' in parameter_type and param.dim() > 1:
                        param = param.view(param.size(0), -1)
                        norm_i[name] = torch.linalg.norm(param, self.q[i], dim=-1)
                norm.append(norm_i)
        elif mode == 'layer':
            norm = []
            for i in range(len(self.q)):
                norm_i = OrderedDict()
                for name, param in model.state_dict().items():
                    parameter_type = name.split('.')[-1]
                    if 'weight' in parameter_type and param.dim() > 1:
                        norm_i[name] = torch.linalg.norm(param.view(-1), self.q[i], dim=-1)
                norm.append(norm_i)
        elif mode == 'global':
            norm = []
            for i in range(len(self.q)):
                param_all = []
                for name, param in model.state_dict().items():
                    parameter_type = name.split('.')[-1]
                    if 'weight' in parameter_type and param.dim() > 1:
                        param_all.append(param.view(-1))
                param_all = torch.cat(param_all, dim=0)
                norm_i = torch.linalg.norm(param_all, self.q[i], dim=-1)
                norm.append(norm_i)
        else:
            raise ValueError('Not valid mode')
        return norm


class Compression:
    def __init__(self, prune_ratio, prune_mode):
        self.init_model_state_dict = models.load_init_state_dict(cfg['seed'])
        self.mask = [self.make_mask(self.init_model_state_dict)]
        self.prune_ratio = prune_ratio
        self.prune_mode = prune_mode

    def make_mask(self, model_state_dict):
        mask = OrderedDict()
        for name, param in model_state_dict.items():
            parameter_type = name.split('.')[-1]
            if 'weight' in parameter_type and param.dim() > 1:
                mask[name] = param.new_ones(param.size(), dtype=torch.bool)
        return mask

    def make_bound(self, sie, d, q, eta_m):
        m = d * sie ** (q / (q - 1)) * (1 + eta_m) ** (1 / (q - 1))
        m = torch.ceil(m).long()
        return m

    def prune(self, model, sparsity_index=None):
        if self.prune_mode[1] == 'neuron':
            new_mask = OrderedDict()
            for name, param in model.named_parameters():
                parameter_type = name.split('.')[-1]
                if 'weight' in parameter_type and param.dim() > 1:
                    if 'si' in self.prune_ratio:
                        prune_ratio_list = self.prune_ratio.split('-')
                        q, eta_m = float(prune_ratio_list[1]), float(prune_ratio_list[2])
                        q_idx = cfg['si_q'].index(q)
                        mask = self.mask[-1][name]
                        sie_i = sparsity_index.sie[self.prune_mode[1]][-1][q_idx][name]
                        d = mask.float().sum(dim=list(range(1, param.dim()))).to(sie_i.device)
                        m = self.make_bound(sie_i, d, q, eta_m)
                        d_m = d.long() - m
                        pivot_value = torch.sort(param.data.abs().view(param.size(0), -1), dim=1)[0][
                            torch.arange(param.size(0)), d_m]
                        pivot_value = pivot_value.view(-1, *[1 for _ in range(param.dim() - 1)])
                    else:
                        mask = self.mask[-1][name]
                        masked_param = param.clone().abs()
                        prune_ratio = float(self.prune_ratio)
                        masked_param[~mask] = float('nan')
                        pivot_value = torch.nanquantile(masked_param, prune_ratio, dim=1, keepdim=True)
                    pivot_mask = (param.data.abs() <= pivot_value).to('cpu')
                    new_mask[name] = torch.where(pivot_mask, False, mask)
                    param.data = torch.where(new_mask[name].to(param.device), param.data,
                                             torch.tensor(0, dtype=torch.float, device=param.device))
        elif self.prune_mode[1] == 'layer':
            new_mask = OrderedDict()
            for name, param in model.named_parameters():
                parameter_type = name.split('.')[-1]
                if 'weight' in parameter_type and param.dim() > 1:
                    if 'si' in self.prune_ratio:
                        prune_ratio_list = self.prune_ratio.split('-')
                        q, eta_m = float(prune_ratio_list[1]), float(prune_ratio_list[2])
                        q_idx = cfg['si_q'].index(q)
                        mask = self.mask[-1][name]
                        sie_i = sparsity_index.sie[self.prune_mode[1]][-1][q_idx][name]
                        d = mask.float().sum().to(sie_i.device)
                        m = self.make_bound(sie_i, d, q, eta_m)
                        d_m = d.long() - m
                        pivot_value = torch.sort(param.data.abs().view(-1))[0][d_m]
                    else:
                        mask = self.mask[-1][name]
                        masked_param = param[mask]
                        prune_ratio = float(self.prune_ratio)
                        pivot_param_i = masked_param.abs()
                        pivot_value = torch.quantile(pivot_param_i, prune_ratio)
                    pivot_mask = (param.data.abs() <= pivot_value).to('cpu')
                    new_mask[name] = torch.where(pivot_mask, False, mask)
                    param.data = torch.where(new_mask[name].to(param.device), param.data,
                                             torch.tensor(0, dtype=torch.float, device=param.device))
        elif self.prune_mode[1] == 'global':
            pivot_param = []
            pivot_mask = []
            for name, param in model.named_parameters():
                parameter_type = name.split('.')[-1]
                if 'weight' in parameter_type and param.dim() > 1:
                    mask = self.mask[-1][name]
                    masked_param = param[mask]
                    pivot_param_i = masked_param.abs()
                    pivot_param.append(pivot_param_i.view(-1))
                    pivot_mask.append(mask.view(-1))
            pivot_param = torch.cat(pivot_param, dim=0)
            pivot_mask = torch.cat(pivot_mask, dim=0)
            if 'si' in self.prune_ratio:
                prune_ratio_list = self.prune_ratio.split('-')
                q, eta_m = float(prune_ratio_list[1]), float(prune_ratio_list[2])
                q_idx = cfg['si_q'].index(q)
                mask = pivot_mask
                sie_i = sparsity_index.sie[self.prune_mode[1]][-1][q_idx]
                d = mask.float().sum().to(sie_i.device)
                # m = self.make_bound(sie_i, d, q, eta_m)
                m = self.make_bound(sie_i, d, q, 0)
                c = eta_m
                prune_ratio = min(c*(1 - m / d), 0.9)
                d_m = torch.floor(d * prune_ratio).long()
                # d_m = d.long() - m
                print('c: {}'.format(c), 'sie: {}'.format(sie_i), 'd: {}'.format(d), 'd_m: {}'.format(d_m), 'prune_ratio: {}'.format(prune_ratio))
                pivot_value = torch.sort(pivot_param.data.abs().view(-1))[0][d_m]
                # pivot_value = torch.sort(pivot_param.data.abs().view(-1))[0][m]
            else:
                prune_ratio = float(self.prune_ratio)
                pivot_value = torch.quantile(pivot_param.data.abs(), prune_ratio)
            new_mask = OrderedDict()
            for name, param in model.named_parameters():
                parameter_type = name.split('.')[-1]
                if 'weight' in parameter_type and param.dim() > 1:
                    mask = self.mask[-1][name]
                    pivot_mask = (param.data.abs() < pivot_value).to('cpu')
                    new_mask[name] = torch.where(pivot_mask, False, mask)
                    param.data = torch.where(new_mask[name].to(param.device), param.data,
                                             torch.tensor(0, dtype=torch.float, device=param.device))
        else:
            raise ValueError('Not valid prune mode')
        self.mask.append(new_mask)
        return

    def init(self, model):
        for name, param in model.named_parameters():
            parameter_type = name.split('.')[-1]
            if 'weight' in parameter_type and param.dim() > 1:
                mask = self.mask[-1][name]
                param.data = torch.where(mask, self.init_model_state_dict[name],
                                         torch.tensor(0, dtype=torch.float)).to(param.device)
            if "bias" in parameter_type:
                param.data = self.init_model_state_dict[name].to(param.device)
        return

    def freeze_grad(self, model):
        for name, param in model.named_parameters():
            parameter_type = name.split('.')[-1]
            if 'weight' in parameter_type and param.dim() > 1:
                mask = self.mask[-1][name]
                param.grad.data = torch.where(mask.to(param.device), param.grad.data,
                                              torch.tensor(0, dtype=torch.float, device=param.device))
        return
