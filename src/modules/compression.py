import numpy as np
import torch
from collections import OrderedDict
from config import cfg


def make_bound_si(si, d, p, q, eta_m):
    m = d * (1 + eta_m) ** (q / (p - q)) * (1 - si) ** ((q * p) / (q - p))
    m = torch.ceil(m).long()
    return m


class Compression:
    def __init__(self, prune_scope, prune_mode):
        super().__init__()
        self.prune_scope = prune_scope
        self.prune_mode = prune_mode

    def init(self, model, mask, init_model_state_dict):
        for name, param in model.named_parameters():
            parameter_type = name.split('.')[-1]
            if 'weight' in parameter_type and param.dim() > 1:
                mask_i = mask.state_dict()[name]
                param.data = torch.where(mask_i, init_model_state_dict[name],
                                         torch.tensor(0, dtype=torch.float)).to(param.device)
            else:
                param.data = init_model_state_dict[name].to(param.device)
        return

    def compress(self, model, mask, sparsity_index):
        if self.prune_scope == 'neuron':
            new_mask = OrderedDict()
            for name, param in model.named_parameters():
                parameter_type = name.split('.')[-1]
                if 'weight' in parameter_type and param.dim() > 1:
                    mask_i = mask.state_dict()[name]
                    pivot_param = param.data.abs()
                    pivot_param[~mask_i] = float('nan')
                    if self.prune_mode[0] == 'si':
                        p, q, eta_m, gamma = self.prune_mode[1:]
                        p, q, eta_m, gamma = float(p), float(q), float(eta_m), float(gamma)
                        p_idx = (sparsity_index.p == p).nonzero().item()
                        q_idx = (sparsity_index.q == q).nonzero().item()
                        si_i = sparsity_index.si[self.prune_scope][-1][name][p_idx, q_idx]
                        d = mask_i.float().sum(dim=list(range(1, param.dim()))).to(si_i.device)
                        m = make_bound_si(si_i, d, p, q, eta_m)
                        retain_ratio = m / d
                        prune_ratio = torch.clamp(gamma * (1 - retain_ratio), 0, cfg['beta'])
                        num_prune = torch.floor(d * prune_ratio).long()
                        pivot_value = torch.sort(pivot_param.view(pivot_param.size(0), -1), dim=1)[0][
                            torch.arange(pivot_param.size(0)), num_prune]
                        pivot_value = pivot_value.view(-1, *[1 for _ in range(pivot_param.dim() - 1)])
                    elif self.prune_mode[0] in ['os', 'lt']:
                        prune_ratio = float(self.prune_mode[1])
                        pivot_value = torch.nanquantile(pivot_param, prune_ratio, dim=1, keepdim=True)
                    else:
                        raise ValueError('Not valid prune mode')
                    pivot_mask = (param.data.abs() < pivot_value).to('cpu')
                    new_mask[name] = torch.where(pivot_mask, False, mask_i)
                    param.data = torch.where(new_mask[name].to(param.device), param.data,
                                             torch.tensor(0, dtype=torch.float, device=param.device))
        elif self.prune_scope == 'layer':
            new_mask = OrderedDict()
            for name, param in model.named_parameters():
                parameter_type = name.split('.')[-1]
                if 'weight' in parameter_type and param.dim() > 1:
                    mask_i = mask.state_dict()[name]
                    pivot_param = param[mask_i].data.abs()
                    if self.prune_mode[0] == 'si':
                        p, q, eta_m, gamma = self.prune_mode[1:]
                        p, q, eta_m, gamma = float(p), float(q), float(eta_m), float(gamma)
                        p_idx = (sparsity_index.p == p).nonzero().item()
                        q_idx = (sparsity_index.q == q).nonzero().item()
                        si_i = sparsity_index.si[self.prune_scope][-1][name][p_idx, q_idx]
                        d = mask_i.float().sum().to(si_i.device)
                        m = make_bound_si(si_i, d, p, q, eta_m)
                        retain_ratio = m / d
                        prune_ratio = torch.clamp(gamma * (1 - retain_ratio), 0, cfg['beta'])
                        num_prune = torch.floor(d * prune_ratio).long()
                        pivot_value = torch.sort(pivot_param.view(-1))[0][num_prune]
                    elif self.prune_mode[0] in ['os', 'lt']:
                        prune_ratio = float(self.prune_mode[1])
                        pivot_value = torch.quantile(pivot_param, prune_ratio)
                    else:
                        raise ValueError('Not valid prune mode')
                    pivot_mask = (param.data.abs() < pivot_value).to('cpu')
                    new_mask[name] = torch.where(pivot_mask, False, mask_i)
                    param.data = torch.where(new_mask[name].to(param.device), param.data,
                                             torch.tensor(0, dtype=torch.float, device=param.device))
        elif self.prune_scope == 'global':
            pivot_param = []
            pivot_mask = []
            for name, param in model.named_parameters():
                parameter_type = name.split('.')[-1]
                if 'weight' in parameter_type and param.dim() > 1:
                    mask_i = mask.state_dict()[name]
                    pivot_param_i = param[mask_i].abs()
                    pivot_param.append(pivot_param_i.view(-1))
                    pivot_mask.append(mask_i.view(-1))
            pivot_param = torch.cat(pivot_param, dim=0).data.abs()
            pivot_mask = torch.cat(pivot_mask, dim=0)
            if self.prune_mode[0] == 'si':
                p, q, eta_m, gamma = self.prune_mode[1:]
                p, q, eta_m, gamma = float(p), float(q), float(eta_m), float(gamma)
                p_idx = (sparsity_index.p == p).nonzero().item()
                q_idx = (sparsity_index.q == q).nonzero().item()
                mask_i = pivot_mask
                si_i = sparsity_index.si[self.prune_scope][-1]['global'][p_idx, q_idx]
                d = mask_i.float().sum().to(si_i.device)
                m = make_bound_si(si_i, d, p, q, eta_m)
                retain_ratio = m / d
                prune_ratio = torch.clamp(gamma * (1 - retain_ratio), 0, cfg['beta'])
                num_prune = torch.floor(d * prune_ratio).long()
                pivot_value = torch.sort(pivot_param.view(-1))[0][num_prune]
            else:
                prune_ratio = float(self.prune_mode[1])
                pivot_value = np.quantile(pivot_param.data.abs().cpu().numpy(), prune_ratio)
            new_mask = OrderedDict()
            for name, param in model.named_parameters():
                parameter_type = name.split('.')[-1]
                if 'weight' in parameter_type and param.dim() > 1:
                    mask_i = mask.state_dict()[name]
                    pivot_mask = (param.data.abs() < pivot_value).to('cpu')
                    new_mask[name] = torch.where(pivot_mask, False, mask_i)
                    param.data = torch.where(new_mask[name].to(param.device), param.data,
                                             torch.tensor(0, dtype=torch.float, device=param.device))
        else:
            raise ValueError('Not valid prune mode')
        mask.load_state_dict(new_mask)
        return
