import collections.abc as container_abcs
import errno
import numpy as np
import os
import pickle
import torch
import torch.optim as optim
from itertools import repeat
from torchvision.utils import save_image
from config import cfg


def check_exists(path):
    return os.path.exists(path)


def makedir_exist_ok(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return


def save(input, path, mode='torch'):
    dirname = os.path.dirname(path)
    makedir_exist_ok(dirname)
    if mode == 'torch':
        torch.save(input, path)
    elif mode == 'np':
        np.save(path, input, allow_pickle=True)
    elif mode == 'pickle':
        pickle.dump(input, open(path, 'wb'))
    else:
        raise ValueError('Not valid save mode')
    return


def load(path, mode='torch'):
    if mode == 'torch':
        return torch.load(path, map_location=lambda storage, loc: storage)
    elif mode == 'np':
        return np.load(path, allow_pickle=True)
    elif mode == 'pickle':
        return pickle.load(open(path, 'rb'))
    else:
        raise ValueError('Not valid save mode')
    return


def save_img(img, path, nrow=10, padding=1, pad_value=0, value_range=None):
    makedir_exist_ok(os.path.dirname(path))
    normalize = False if range is None else True
    save_image(img, path, nrow=nrow, padding=padding, pad_value=pad_value, normalize=normalize, value_range=value_range)
    return


def to_device(input, device):
    output = recur(lambda x, y: x.to(y), input, device)
    return output


def ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


def apply_fn(module, fn):
    for n, m in module.named_children():
        if hasattr(m, fn):
            exec('m.{0}()'.format(fn))
        if sum(1 for _ in m.named_children()) != 0:
            exec('apply_fn(m,\'{0}\')'.format(fn))
    return


def recur(fn, input, *args):
    if isinstance(input, torch.Tensor) or isinstance(input, np.ndarray):
        output = fn(input, *args)
    elif isinstance(input, list):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
    elif isinstance(input, tuple):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
        output = tuple(output)
    elif isinstance(input, dict):
        output = {}
        for key in input:
            output[key] = recur(fn, input[key], *args)
    elif isinstance(input, str):
        output = input
    elif input is None:
        output = None
    else:
        raise ValueError('Not valid input type')
    return output


def process_dataset(dataset):
    cfg['data_size'] = {'train': len(dataset['train']), 'test': len(dataset['test'])}
    cfg['target_size'] = dataset['train'].target_size
    return


def process_control():
    cfg['data_name'] = cfg['control']['data_name']
    if 'num_iters' in cfg['control']:
        cfg['num_iters'] = int(cfg['control']['num_iters'])
    data_shape = {'MNIST': [1, 28, 28], 'FashionMNIST': [1, 28, 28], 'SVHN': [3, 32, 32], 'CIFAR10': [3, 32, 32],
                  'CIFAR100': [3, 32, 32]}
    if 'Blob' in cfg['data_name']:
        data_control_list = cfg['data_name'].split('-')[1:]
        cfg['Blob'] = {'num_samples': int(data_control_list[0]), 'num_features': int(data_control_list[1]),
                       'num_centers': int(data_control_list[2]), 'noise': float(data_control_list[3])}
        data_shape[cfg['data_name']] = [cfg['Blob']['num_features']]
    elif 'Friedman' in cfg['data_name']:
        data_control_list = cfg['data_name'].split('-')[1:]
        cfg['Friedman'] = {'num_samples': int(data_control_list[0]), 'num_features': int(data_control_list[1]),
                           'noise': float(data_control_list[2])}
        data_shape[cfg['data_name']] = [cfg['Friedman']['num_features']]
    elif 'MLP' in cfg['data_name']:
        data_control_list = cfg['data_name'].split('-')[1:]
        cfg['MLP'] = {'mode': data_control_list[0], 'data_size': int(data_control_list[1]),
                      'input_size': int(data_control_list[2]), 'hidden_size': int(data_control_list[3]),
                      'scale_factor': float(data_control_list[4]), 'num_layers': int(data_control_list[5]),
                      'activation': data_control_list[6], 'target_size': int(data_control_list[7]),
                      'noise': float(data_control_list[8]), 'sparsity': float(data_control_list[9])}
        data_shape[cfg['data_name']] = [cfg['MLP']['input_size']]
    cfg['data_shape'] = data_shape[cfg['data_name']]
    if 'mlp' in cfg['control']['model_name']:
        mlp_control_list = cfg['control']['model_name'].split('-')
        cfg['model_name'] = mlp_control_list[0]
        cfg['mlp'] = {'hidden_size': int(mlp_control_list[1]), 'scale_factor': float(mlp_control_list[2]),
                      'num_layers': int(mlp_control_list[3]), 'activation': mlp_control_list[4]}
    else:
        cfg['model_name'] = cfg['control']['model_name']
    cfg['cnn'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['resnet9'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['resnet18'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['wresnet28x2'] = {'depth': 28, 'widen_factor': 2, 'drop_rate': 0.0}
    cfg['teacher'] = {}
    cfg['teacher']['shuffle'] = {'train': True, 'test': False}
    cfg['teacher']['optimizer_name'] = 'SGD'
    cfg['teacher']['lr'] = 3e-2
    cfg['teacher']['momentum'] = 0.9
    cfg['teacher']['weight_decay'] = 5e-4
    cfg['teacher']['nesterov'] = True
    cfg['teacher']['scheduler_name'] = 'CosineAnnealingLR'
    if 'Blob' in cfg['data_name']:
        cfg['teacher']['num_epochs'] = 100
        cfg['teacher']['batch_size'] = {'train': 250, 'test': 500}
    elif 'Friedman' in cfg['data_name']:
        cfg['teacher']['num_epochs'] = 100
        cfg['teacher']['batch_size'] = {'train': 250, 'test': 500}
    elif 'MLP' in cfg['data_name']:
        cfg['teacher']['num_epochs'] = 100
        cfg['teacher']['batch_size'] = {'train': 250, 'test': 500}
    elif cfg['data_name'] in ['MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'CIFAR100']:
        cfg['teacher']['num_epochs'] = 1
        cfg['teacher']['batch_size'] = {'train': 250, 'test': 500}
    else:
        raise ValueError('Not valid data name')
    cfg['prune_percent'] = 0.1
    cfg['stats'] = make_stats()
    return


def make_stats():
    stats = {}
    stats_path = './res/stats'
    makedir_exist_ok(stats_path)
    filenames = os.listdir(stats_path)
    for filename in filenames:
        stats_name = os.path.splitext(filename)[0]
        stats[stats_name] = load(os.path.join(stats_path, filename))
    return stats


class Stats(object):
    def __init__(self, dim):
        self.dim = dim
        self.n_samples = 0
        self.n_features = None
        self.mean = None
        self.std = None

    def update(self, data):
        data = data.transpose(self.dim, -1).reshape(-1, data.size(self.dim))
        if self.n_samples == 0:
            self.n_samples = data.size(0)
            self.n_features = data.size(1)
            self.mean = data.mean(dim=0)
            self.std = data.std(dim=0)
        else:
            m = float(self.n_samples)
            n = data.size(0)
            new_mean = data.mean(dim=0)
            new_std = 0 if n == 1 else data.std(dim=0)
            old_mean = self.mean
            old_std = self.std
            self.mean = m / (m + n) * old_mean + n / (m + n) * new_mean
            self.std = torch.sqrt(m / (m + n) * old_std ** 2 + n / (m + n) * new_std ** 2 + m * n / (m + n) ** 2 * (
                    old_mean - new_mean) ** 2)
            self.n_samples += n
        return


def make_optimizer(model, tag):
    if cfg[tag]['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg[tag]['lr'], momentum=cfg[tag]['momentum'],
                              weight_decay=cfg[tag]['weight_decay'], nesterov=cfg[tag]['nesterov'])
    elif cfg[tag]['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg[tag]['lr'], betas=cfg[tag]['betas'],
                               weight_decay=cfg[tag]['weight_decay'])
    elif cfg[tag]['optimizer_name'] == 'LBFGS':
        optimizer = optim.LBFGS(model.parameters(), lr=cfg[tag]['lr'])
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer


def make_scheduler(optimizer, tag):
    if cfg[tag]['scheduler_name'] == 'None':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65535])
    elif cfg[tag]['scheduler_name'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg[tag]['step_size'], gamma=cfg[tag]['factor'])
    elif cfg[tag]['scheduler_name'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg[tag]['milestones'],
                                                   gamma=cfg[tag]['factor'])
    elif cfg[tag]['scheduler_name'] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif cfg[tag]['scheduler_name'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg[tag]['num_epochs'], eta_min=0)
    elif cfg[tag]['scheduler_name'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg[tag]['factor'],
                                                         patience=cfg[tag]['patience'], verbose=False,
                                                         threshold=cfg[tag]['threshold'], threshold_mode='rel',
                                                         min_lr=cfg[tag]['min_lr'])
    elif cfg[tag]['scheduler_name'] == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg[tag]['lr'], max_lr=10 * cfg[tag]['lr'])
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler


def resume(model_tag, load_tag='checkpoint', verbose=True):
    if cfg['resume_mode'] == 1:
        if os.path.exists('./output/model/{}_{}.pt'.format(model_tag, load_tag)):
            result = load('./output/model/{}_{}.pt'.format(model_tag, load_tag))
            if verbose:
                print('Resume from {}'.format(result['epoch']))
        else:
            print('Not exists model tag: {}, start from scratch'.format(model_tag))
            result = None
    else:
        result = None
    return result


def collate(input):
    for k in input:
        input[k] = torch.stack(input[k], 0)
    return input


def make_layerwise_sparsity_index(model_state_dict):
    p = (torch.arange(1, 10) / 10).tolist()
    L = 0
    for k, v in model_state_dict[0].items():
        if 'weight' in k:
            L += 1
    sparsity_index = {'mean': np.zeros((len(p), L)),
                      'std': np.zeros((len(p), L))}
    for i in range(len(model_state_dict)):
        j = 0
        for k, v in model_state_dict[i].items():
            if 'weight' in k:
                mean = v.mean()
                std = v.std() if len(v) > 1 else 0
                sparsity_index['mean'][i, j] = mean
                sparsity_index['std'][i, j] = std
                j += 1
    return sparsity_index
