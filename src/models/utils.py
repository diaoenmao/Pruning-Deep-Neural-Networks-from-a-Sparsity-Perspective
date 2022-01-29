import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


def init_param_classifier(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        if m.bias is not None:
            m.bias.data.zero_()
    return m


def init_param_generator(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    return m


def normalize(input):
    if cfg['data_name'] in cfg['stats']:
        broadcast_size = [1] * input.dim()
        broadcast_size[1] = input.size(1)
        m, s = cfg['stats'][cfg['data_name']]
        m, s = torch.tensor(m, dtype=input.dtype).view(broadcast_size).to(input.device), \
               torch.tensor(s, dtype=input.dtype).view(broadcast_size).to(input.device)
        input = input.sub(m).div(s)
    return input


def denormalize(input):
    if cfg['data_name'] in cfg['stats']:
        broadcast_size = [1] * input.dim()
        broadcast_size[1] = input.size(1)
        m, s = cfg['stats'][cfg['data_name']]
        m, s = torch.tensor(m, dtype=input.dtype).view(broadcast_size).to(input.device), \
               torch.tensor(s, dtype=input.dtype).view(broadcast_size).to(input.device)
        input = input.mul(s).add(m)
    return input


def make_batchnorm(m, momentum, track_running_stats):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.momentum = momentum
        m.track_running_stats = track_running_stats
        if track_running_stats:
            m.register_buffer('running_mean', torch.zeros(m.num_features, device=cfg['device']))
            m.register_buffer('running_var', torch.ones(m.num_features, device=cfg['device']))
            m.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long, device=cfg['device']))
        else:
            m.running_mean = None
            m.running_var = None
            m.num_batches_tracked = None
    return m


def loss_fn(output, target):
    if target.dtype == torch.int64:
        loss = F.cross_entropy(output, target)
    else:
        loss = kld_loss(output, target)
    return loss


def mae_loss(output, target, weight=None):
    mae = F.l1_loss(output, target, reduction='none')
    mae = weight * mae if weight is not None else mae
    mae = torch.sum(mae)
    mae /= output.size(0)
    return mae


def mse_loss(output, target, weight=None):
    mse = F.mse_loss(output, target, reduction='none')
    mse = weight * mse if weight is not None else mse
    mse = torch.sum(mse)
    mse /= output.size(0)
    return mse


def cross_entropy_loss(output, target, weight=None):
    if target.dtype != torch.int64:
        target = (target.topk(1, 1, True, True)[1]).view(-1)
    ce = F.cross_entropy(output, target, weight=weight, reduction='mean')
    return ce


def kld_loss(output, target, weight=None, T=1):
    kld = F.kl_div(F.log_softmax(output, dim=-1), F.softmax(target / T, dim=-1), reduction='none')
    kld = weight * kld if weight is not None else kld
    kld = torch.sum(kld)
    kld /= output.size(0)
    return kld


def margin_loss(output, target, weight=None):
    margin = F.multi_margin_loss(output, target, p=1, margin=1.0, weight=weight, reduction='mean')
    return margin


def make_weight(target):
    cls_indx, cls_counts = torch.unique(target, return_counts=True)
    num_samples_per_cls = torch.zeros(cfg['target_size'], dtype=torch.float32, device=target.device)
    num_samples_per_cls[cls_indx] = cls_counts.float()
    beta = torch.tensor(0.999, dtype=torch.float32, device=target.device)
    effective_num = 1.0 - beta.pow(num_samples_per_cls)
    weight = (1.0 - beta) / effective_num
    weight[torch.isinf(weight)] = 0
    weight = weight / torch.sum(weight) * (weight > 0).float().sum()
    return weight


def total_variation_loss(output):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    diff1 = output[:, :, :, :-1] - output[:, :, :, 1:]
    diff2 = output[:, :, :-1, :] - output[:, :, 1:, :]
    diff3 = output[:, :, 1:, :-1] - output[:, :, :-1, 1:]
    diff4 = output[:, :, :-1, :-1] - output[:, :, 1:, 1:]
    loss = torch.linalg.norm(diff1) + torch.linalg.norm(diff2) + torch.linalg.norm(diff3) + torch.linalg.norm(diff4)
    return loss


class MomentHook:
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        # forcing mean and variance to match between two distributions
        # other ways might work better, e.g. KL divergence
        self.moment_loss = torch.linalg.norm(module.running_var.data.type(var.type()) - var, 2) + torch.linalg.norm(
            module.running_mean.data.type(var.type()) - mean, 2)
        # must have no output

    def close(self):
        self.hook.remove()


def make_moment_loss(model):
    moment_loss = []
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            moment_loss.append(MomentHook(module))
    return moment_loss
