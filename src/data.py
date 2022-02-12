import copy
import torch
import numpy as np
import models
from config import cfg
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

data_stats = {'MNIST': ((0.1307,), (0.3081,)), 'FashionMNIST': ((0.2860,), (0.3530,)),
              'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
              'CIFAR100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
              'SVHN': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))}


def fetch_dataset(data_name, verbose=True):
    import datasets
    dataset = {}
    if verbose:
        print('fetching data {}...'.format(data_name))
    root = './data/{}'.format(data_name)
    if data_name in ['MNIST', 'FashionMNIST']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['train'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset['test'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['CIFAR10', 'CIFAR100']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['train'].transform = datasets.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset['test'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['SVHN']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['train'].transform = datasets.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset['test'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    else:
        raise ValueError('Not valid dataset name')
    if verbose:
        print('data ready')
    return dataset


def input_collate(batch):
    if isinstance(batch[0], dict):
        return {key: [b[key] for b in batch] for key in batch[0]}
    else:
        return default_collate(batch)


def make_data_loader(dataset, tag, batch_size=None, shuffle=None, sampler=None):
    data_loader = {}
    for k in dataset:
        _batch_size = cfg[tag]['batch_size'][k] if batch_size is None else batch_size[k]
        _shuffle = cfg[tag]['shuffle'][k] if shuffle is None else shuffle[k]
        if sampler is None:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, shuffle=_shuffle,
                                        pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))
        else:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, sampler=sampler[k],
                                        pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))
    return data_loader


def split_dataset(dataset, num_splits, data_split_mode):
    if data_split_mode == 'iid':
        data_split, target_split = iid(dataset, num_splits)
    elif 'non-iid' in cfg['data_split_mode']:
        data_split, target_split = non_iid(dataset, num_splits)
    else:
        raise ValueError('Not valid data split mode')
    return data_split, target_split


def iid(dataset, num_splits):
    data_split = [{k: None for k in dataset} for _ in range(num_splits)]
    target_split = [{k: None for k in dataset} for _ in range(num_splits)]
    idx = {k: list(range(len(dataset[k]))) for k in dataset}
    num_items = {k: int(len(dataset[k]) / num_splits) for k in dataset}
    for i in range(num_splits):
        for k in dataset:
            num_items_i_k = min(len(idx[k]), num_items[k])
            data_split[i][k] = torch.tensor(idx[k])[torch.randperm(len(idx[k]))[:num_items_i_k]].tolist()
            idx[k] = list(set(idx[k]) - set(data_split[i][k]))
            target_i_k = torch.tensor(dataset[k].target)[data_split[i][k]]
            if k == 'train':
                u_target_i_k, num_target_i = torch.unique(target_i_k, sorted=True, return_counts=True)
                target_split[i][k] = {u_target_i_k[j].item(): num_target_i[j].item() for j in range(len(u_target_i_k))}
            else:
                target_split[i][k] = {x: (target_i_k == x).sum().item() for x in target_split[i]['train']}
    return data_split, target_split


def non_iid(dataset, num_splits):
    data_split_mode_list = cfg['data_split_mode'].split('-')
    data_split_mode_tag = data_split_mode_list[-2]
    target_size = len(torch.unique(torch.tensor(dataset['train'].target)))
    if data_split_mode_tag == 'l':
        data_split = [{k: [] for k in dataset} for _ in range(num_splits)]
        shard_per_user = int(data_split_mode_list[-1])
        shard_per_class = int(shard_per_user * num_splits / target_size)
        target_idx_split = [{k: None for k in dataset} for _ in range(target_size)]
        for k in dataset:
            target = torch.tensor(dataset[k].target)
            for target_i in range(target_size):
                target_idx = torch.where(target == target_i)[0]
                num_leftover = len(target_idx) % shard_per_class
                leftover = target_idx[-num_leftover:] if num_leftover > 0 else []
                target_idx = target_idx[:-num_leftover] if num_leftover > 0 else target_idx
                target_idx = target_idx.reshape((shard_per_class, -1)).tolist()
                for i, leftover_target_idx in enumerate(leftover):
                    target_idx[i].append(leftover_target_idx.item())
                target_idx_split[target_i][k] = target_idx
        target_split_key = []
        for i in range(shard_per_class):
            target_split_key.append(torch.randperm(target_size))
        target_split_key = torch.cat(target_split_key, dim=0)
        target_split_key = target_split_key.reshape((num_splits, -1)).tolist()
        target_split = [{k: None for k in dataset} for _ in range(num_splits)]
        for i in range(num_splits):
            for k in dataset:
                target = torch.tensor(dataset[k].target)
                for target_i in target_split_key[i]:
                    idx = torch.randint(len(target_idx_split[target_i][k]), (1,)).item()
                    data_split[i][k].extend(target_idx_split[target_i][k].pop(idx))
                target_i, num_target_i = torch.unique(target[data_split[i][k]], return_counts=True)
                target_split[i][k] = {target_i[j].item(): num_target_i[j].item() for j in range(len(target_i))}
    elif data_split_mode_tag == 'd':
        min_size = 0
        required_min_size = 10
        target_split_consistent_flag = False
        while min_size < required_min_size or not target_split_consistent_flag:
            beta = float(data_split_mode_list[-1])
            dir = torch.distributions.dirichlet.Dirichlet(torch.tensor(beta).repeat(num_splits))
            data_split = [{k: [] for k in dataset} for _ in range(num_splits)]
            for target_i in range(target_size):
                proportions = dir.sample()
                for k in dataset:
                    target = torch.tensor(dataset[k].target)
                    target_idx = torch.where(target == target_i)[0]
                    proportions = torch.tensor([p * (len(data_split_idx[k]) < (target.size(0) / num_splits))
                                                for p, data_split_idx in zip(proportions, data_split)])
                    proportions = proportions / proportions.sum()
                    split_idx = (torch.cumsum(proportions, dim=-1) * len(target_idx)).long().tolist()[:-1]
                    split_idx = torch.tensor_split(target_idx, split_idx)
                    for i in range(len(split_idx)):
                        data_split[i][k].extend(split_idx[i].tolist())
            min_size = min([len(data_split[i][k]) for i in range(len(data_split)) for k in data_split[i]])
            target_split = [{k: None for k in dataset} for _ in range(num_splits)]
            target_split_consistent_flag = True
            for i in range(num_splits):
                for k in dataset:
                    target_i_k = torch.tensor(dataset[k].target)[data_split[i][k]]
                    if k == 'train':
                        u_target_i_k, num_target_i = torch.unique(target_i_k, sorted=True, return_counts=True)
                        target_i_train_set = set(u_target_i_k.tolist())
                        target_split[i][k] = {u_target_i_k[j].item(): num_target_i[j].item() for j in
                                              range(len(u_target_i_k))}
                    else:
                        u_target_i_k, num_target_i = torch.unique(target_i_k, sorted=True, return_counts=True)
                        target_i_k_set = set(u_target_i_k.tolist())
                        if target_i_k_set.issubset(target_i_train_set):
                            target_split[i][k] = {x: (target_i_k == x).sum().item() for x in target_split[i]['train']}
                        else:
                            target_split_consistent_flag = False
                            break
    else:
        raise ValueError('Not valid data split mode tag')
    return data_split, target_split


def make_split_dataset(dataset, data_split, target_split):
    split_dataset = []
    for i in range(len(data_split)):
        dataset_i = copy.deepcopy(dataset)
        target_split_key_i = list(target_split[i]['train'].keys())
        target_map = {target_split_key_i[j]: j for j in range(len(target_split_key_i))}
        for k in dataset:
            target_split_key_k = list(target_split[i][k].keys())
            target_map_k = {key: target_map[key] for key in target_split_key_k}
            dataset_i[k].data = dataset_i[k].data[data_split[i][k]]
            dataset_i[k].target = np.vectorize(target_map_k.get)(dataset_i[k].target[data_split[i][k]]).astype(
                dataset_i[k].target.dtype)
        split_dataset.append(dataset_i)
    return split_dataset


class BaseDataset(Dataset):
    def __init__(self, **kwargs):
        self.data = kwargs

    def __getitem__(self, index):
        input = {k: self.data[k][index] for k in self.data}
        return input

    def __len__(self):
        return len(self.data[list(self.data.keys())[0]])
