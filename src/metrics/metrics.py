import torch
import torch.nn.functional as F
from config import cfg
from utils import recur


def Accuracy(output, target, topk=1):
    with torch.no_grad():
        if target.dtype != torch.int64:
            target = (target.topk(1, 1, True, True)[1]).view(-1)
        batch_size = target.size(0)
        pred_k = output.topk(topk, 1, True, True)[1]
        correct_k = pred_k.eq(target.view(-1, 1).expand_as(pred_k)).float().sum()
        acc = (correct_k * (100.0 / batch_size)).item()
    return acc


def RMSE(output, target):
    with torch.no_grad():
        rmse = F.mse_loss(output, target).sqrt().item()
    return rmse


class Metric(object):
    def __init__(self, metric_name):
        self.metric_name = self.make_metric_name(metric_name)
        self.metric = {'Loss': (lambda input, output: output['loss'].item()),
                       'Accuracy': (lambda input, output: recur(Accuracy, output['target'], input['target'])),
                       'RMSE': (lambda input, output: recur(RMSE, output['target'], input['target']))}
        self.reset()

    def make_metric_name(self, metric_name):
        for split in metric_name:
            if cfg['data_name'] in ['MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'CIFAR100']:
                metric_name[split] += ['Accuracy']
            else:
                raise ValueError('Not valid data name')
        return metric_name

    def reset(self):
        if cfg['data_name'] in ['MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'CIFAR100']:
            pivot = -float('inf')
            pivot_direction = 'up'
            pivot_name = 'Accuracy'
        else:
            raise ValueError('Not valid data name')
        self.pivot, self.pivot_name, self.pivot_direction = pivot, pivot_name, pivot_direction
        return

    def evaluate(self, metric_names, input, output):
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output)
        return evaluation

    def compare(self, val):
        if self.pivot_direction == 'down':
            compared = self.pivot > val
        elif self.pivot_direction == 'up':
            compared = self.pivot < val
        else:
            raise ValueError('Not valid pivot direction')
        return compared

    def update(self, val):
        self.pivot = val
        return

    def load_state_dict(self, state_dict):
        self.pivot = state_dict['pivot']
        self.pivot_name = state_dict['pivot_name']
        self.pivot_direction = state_dict['pivot_direction']
        return

    def state_dict(self):
        return {'pivot': self.pivot, 'pivot_name': self.pivot_name, 'pivot_direction': self.pivot_direction}