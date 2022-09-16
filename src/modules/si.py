import torch
from collections import OrderedDict


class SparsityIndex:
    def __init__(self, p, q):
        self.p = p
        self.q = q
        self.si = {'neuron': [], 'layer': [], 'global': []}

    def sparsity_index(self, x, mask, p, q, dim):
        d = mask.to(x.device).float().sum(dim=dim)
        si = (torch.linalg.norm(x, 1, dim=dim).pow(p) / d).pow(1 / p) / \
             (torch.linalg.norm(x, q, dim=dim).pow(q) / d).pow(1 / q)
        return si

    def make_sparsity_index(self, model, mask):
        self.si['neuron'].append(self.make_sparsity_index_(model, mask, 'neuron'))
        self.si['layer'].append(self.make_sparsity_index_(model, mask, 'layer'))
        self.si['global'].append(self.make_sparsity_index_(model, mask, 'global'))
        return

    def make_sparsity_index_(self, model, mask, mode):
        si = OrderedDict()
        if mode == 'neuron':
            for name, param in model.state_dict().items():
                parameter_type = name.split('.')[-1]
                if 'weight' in parameter_type and param.dim() > 1:
                    si_i = []
                    for i in range(len(self.p)):
                        for j in range(len(self.q)):
                            si_i.append(self.sparsity_index(param, mask.state_dict()[name], self.p[i], self.q[j], 1))
                    si_i = torch.cat(si_i, dim=0)
                    si[name] = si_i.reshape((len(self.p), len(self.q), -1))
        elif mode == 'layer':
            for name, param in model.state_dict().items():
                parameter_type = name.split('.')[-1]
                if 'weight' in parameter_type and param.dim() > 1:
                    si_i = []
                    for i in range(len(self.p)):
                        for j in range(len(self.q)):
                            si.append(self.sparsity_index(param, mask.state_dict()[name], self.p[i], self.q[j], -1))
                    si_i = torch.cat(si_i, dim=0)
                    si[name] = si_i.reshape((len(self.p), len(self.q), -1))
        elif mode == 'global':
            param_all = []
            mask_all = []
            for name, param in model.state_dict().items():
                parameter_type = name.split('.')[-1]
                if 'weight' in parameter_type and param.dim() > 1:
                    param_all.append(param.view(-1))
                    mask_all.append(mask.state_dict()[name].view(-1))
            param_all = torch.cat(param_all, dim=0)
            mask_all = torch.cat(mask_all, dim=0)
            si_i = []
            for i in range(len(self.p)):
                for j in range(len(self.q)):
                    si.append(self.sparsity_index(param_all, mask_all, self.p[i], self.q[j], -1))
            si_i = torch.cat(si_i, dim=0)
            si['global'] = si_i.reshape((len(self.p), len(self.q), -1))
        else:
            raise ValueError('Not valid mode')
        return si
