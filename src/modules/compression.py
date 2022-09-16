import torch
from collections import OrderedDict


def make_bound_si(si, d, p, q, gamma, eta_m):
    m = d * si ** (q / (q - 1)) * (1 + eta_m) ** (1 / (q - 1))
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
                        p, q, gamma, eta_m = self.prune_mode[1:]
                        p_idx = sparsity_index.p.index(p)
                        q_idx = sparsity_index.q.index(q)
                        si_i = sparsity_index.si[self.prune_scope][-1][name][p_idx, q_idx]
                        d = mask_i.float().sum(dim=list(range(1, param.dim()))).to(si_i.device)
                        m = gamma ** (1 / (1 - q)) * self.make_bound_si(si_i, d, p, q, gamma, eta_m) / d
                        prune_ratio = torch.minimum(eta_m * (1 - m), d.new_tensor([0.9]))
                        prune_ratio[m > 1] = 0.
                        d_m = torch.floor(d * prune_ratio).long()
                        pivot_value = torch.sort(pivot_param.view(pivot_param.size(0), -1), dim=1)[0][
                            torch.arange(pivot_param.size(0)), d_m]
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
        elif self.prune_mode[1] == 'layer':
            new_mask = OrderedDict()
            for name, param in model.named_parameters():
                parameter_type = name.split('.')[-1]
                if 'weight' in parameter_type and param.dim() > 1:
                    mask_i = mask.state_dict()[name]
                    pivot_param = param[mask_i].data.abs()
                    if self.prune_mode[0] == 'si':
                        p, q, gamma, eta_m = self.prune_mode[1:]
                        p_idx = sparsity_index.p.index(p)
                        q_idx = sparsity_index.q.index(q)
                        si_i = sparsity_index.si[self.prune_scope][-1][name][p_idx, q_idx]
                        d = mask_i.float().sum().to(si_i.device)
                        m = gamma ** (1 / (1 - q)) * self.make_bound_si(si_i, d, p, q, gamma, eta_m) / d
                        if m > 1:
                            prune_ratio = 0.
                        else:
                            prune_ratio = min(eta_m * (1 - m), 0.9)
                        d_m = torch.floor(d * prune_ratio).long()
                        pivot_value = torch.sort(pivot_param.view(-1))[0][d_m]
                    elif self.prune_mode[0] in ['os', 'lt']:
                        prune_ratio = float(self.prune_ratio)
                        pivot_value = torch.quantile(pivot_param, prune_ratio)
                    else:
                        raise ValueError('Not valid prune mode')
                    pivot_mask = (param.data.abs() < pivot_value).to('cpu')
                    new_mask[name] = torch.where(pivot_mask, False, mask_i)
                    param.data = torch.where(new_mask[name].to(param.device), param.data,
                                             torch.tensor(0, dtype=torch.float, device=param.device))
        elif self.prune_mode[1] == 'global':
            pivot_param = []
            pivot_mask = []
            for name, param in model.named_parameters():
                parameter_type = name.split('.')[-1]
                if 'weight' in parameter_type and param.dim() > 1:
                    mask_i = mask.state_dict()[name]
                    pivot_param_i = param[mask_i].abs()
                    pivot_param.append(pivot_param_i.view(-1))
                    pivot_mask.append(mask_i.view(-1))
            pivot_param = torch.cat(pivot_param, dim=0)
            pivot_mask = torch.cat(pivot_mask, dim=0)
            if 'si' in self.prune_ratio:
                p, q, gamma, eta_m = self.prune_mode[1:]
                p_idx = sparsity_index.p.index(p)
                q_idx = sparsity_index.q.index(q)
                mask_i = pivot_mask
                si_i = sparsity_index.si[self.prune_scope][-1]['global'][p_idx, q_idx]
                d = mask_i.float().sum().to(si_i.device)
                m = gamma ** (1 / (1 - q)) * self.make_bound_si(si_i, d, p, q, gamma, eta_m) / d
                if m > 1:
                    prune_ratio = 0.
                else:
                    prune_ratio = min(eta_m * (1 - m), 0.9)
                d_m = torch.floor(d * prune_ratio).long()
                pivot_value = torch.sort(pivot_param.data.abs().view(-1))[0][d_m]
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
        self.mask.state_dict = new_mask
        return
