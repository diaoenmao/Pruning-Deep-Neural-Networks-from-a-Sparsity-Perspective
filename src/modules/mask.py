import torch
from collections import OrderedDict


class Mask:
    def __init__(self, model_state_dict):
        super().__init__()
        self.state_dict_ = self.init_mask(model_state_dict)

    def init_mask(self, model_state_dict):
        mask = OrderedDict()
        for name, param in model_state_dict.items():
            parameter_type = name.split('.')[-1]
            if 'weight' in parameter_type and param.dim() > 1:
                mask[name] = param.new_ones(param.size(), dtype=torch.bool)
        return mask

    def freeze_grad(self, model):
        for name, param in model.named_parameters():
            parameter_type = name.split('.')[-1]
            if 'weight' in parameter_type and param.dim() > 1:
                mask_i = self.state_dict()[name]
                param.grad.data = torch.where(mask_i.to(param.device), param.grad.data,
                                              torch.tensor(0, dtype=torch.float, device=param.device))
        return

    def load_state_dict(self, state_dict):
        self.state_dict_ = state_dict
        return

    def state_dict(self):
        return self.state_dict_
