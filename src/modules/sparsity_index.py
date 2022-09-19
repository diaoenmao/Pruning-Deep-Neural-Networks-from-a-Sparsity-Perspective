import torch
from collections import OrderedDict


class SparsityIndex:
    def __init__(self, p, q):
        self.p = p
        self.q = q
        self.si = {'neuron': [], 'layer': [], 'global': []}
        self.gini = {'neuron': [], 'layer': [], 'global': []}

    def make_si(self, x, mask, dim, p, q):
        d = mask.to(x.device).float().sum(dim=dim)
        # si = (torch.linalg.norm(x, p, dim=dim).pow(p) / d).pow(1 / p) / \
        #      (torch.linalg.norm(x, q, dim=dim).pow(q) / d).pow(1 / q)
        # si = d ** ((1 / q) - (1 / p)) * torch.linalg.norm(x, p, dim=dim) / torch.linalg.norm(x, q, dim=dim)
        si = (torch.linalg.norm(x, q, dim=dim) -
              d ** ((1 / q) - (1 / p)) * torch.linalg.norm(x, p, dim=dim)) / torch.linalg.norm(x, q, dim=dim)
        numerical_check = d ** ((1 / q) - (1 / p)) < 1e-10
        si[numerical_check] = 1
        return si

    def make_gini(self, x, mask, dim, *args):
        x = x.abs()
        x += 1e-7
        x[~mask] = float('nan')
        x = torch.sort(x, dim=dim)[0]
        N = mask.to(x.device).float().sum(dim=dim)
        idx = torch.arange(1, x.size(dim) + 1).to(x.device)
        gini = ((torch.nansum((2 * idx - N.view(-1, 1) - 1) * x, dim=dim)) / (N * torch.nansum(x, dim=dim)))
        return gini

    def make_sparsity_index(self, model, mask):
        self.si['neuron'].append(self.make_sparsity_index_(model, mask, 'neuron', self.make_si))
        self.si['layer'].append(self.make_sparsity_index_(model, mask, 'layer', self.make_si))
        self.si['global'].append(self.make_sparsity_index_(model, mask, 'global', self.make_si))
        self.gini['neuron'].append(self.make_sparsity_index_(model, mask, 'neuron', self.make_gini))
        self.gini['layer'].append(self.make_sparsity_index_(model, mask, 'layer', self.make_gini))
        self.gini['global'].append(self.make_sparsity_index_(model, mask, 'global', self.make_gini))
        return

    def make_sparsity_index_(self, model, mask, scope, func):
        sparsity_index = OrderedDict()
        if scope == 'neuron':
            for name, param in model.state_dict().items():
                parameter_type = name.split('.')[-1]
                if 'weight' in parameter_type and param.dim() > 1:
                    sparsity_index_i = []
                    for i in range(len(self.p)):
                        for j in range(len(self.q)):
                            param_i = param.view(param.size(0), -1)
                            mask_i = mask.state_dict()[name].view(param.size(0), -1)
                            sparsity_index_i.append(func(param_i, mask_i, 1, self.p[i], self.q[j]))
                    sparsity_index_i = torch.cat(sparsity_index_i, dim=0)
                    sparsity_index[name] = sparsity_index_i.reshape((len(self.p), len(self.q), -1))
        elif scope == 'layer':
            for name, param in model.state_dict().items():
                parameter_type = name.split('.')[-1]
                if 'weight' in parameter_type and param.dim() > 1:
                    sparsity_index_i = []
                    for i in range(len(self.p)):
                        for j in range(len(self.q)):
                            param_i = param.view(-1)
                            mask_i = mask.state_dict()[name].view(-1)
                            sparsity_index_i.append(func(param_i, mask_i, -1, self.p[i], self.q[j]))
                    sparsity_index_i = torch.tensor(sparsity_index_i)
                    sparsity_index[name] = sparsity_index_i.reshape((len(self.p), len(self.q), -1))
        elif scope == 'global':
            param_all = []
            mask_all = []
            for name, param in model.state_dict().items():
                parameter_type = name.split('.')[-1]
                if 'weight' in parameter_type and param.dim() > 1:
                    param_all.append(param.view(-1))
                    mask_all.append(mask.state_dict()[name].view(-1))
            param_all = torch.cat(param_all, dim=0)
            mask_all = torch.cat(mask_all, dim=0)
            sparsity_index_i = []
            for i in range(len(self.p)):
                for j in range(len(self.q)):
                    sparsity_index_i.append(func(param_all, mask_all, -1, self.p[i], self.q[j]))
            sparsity_index_i = torch.tensor(sparsity_index_i)
            sparsity_index['global'] = sparsity_index_i.reshape((len(self.p), len(self.q), -1))
        else:
            raise ValueError('Not valid mode')
        return sparsity_index
