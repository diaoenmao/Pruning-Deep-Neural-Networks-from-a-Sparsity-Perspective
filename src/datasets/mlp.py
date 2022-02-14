import numpy as np
import os
import torch
import torch.nn as nn
import models
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from .utils import make_classes_counts


class MLP(Dataset):
    data_name = 'MLP'

    def __init__(self, root, split, mode, data_size=100, input_size=10, hidden_size=32, scale_factor=1, num_layers=1,
                 activation='sigmoid', target_size=10, noise=1.0, sparsity=0.5):
        self.root = os.path.expanduser(root)
        self.split = split
        self.mode = mode
        self.data_size = data_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.scale_factor = scale_factor
        self.num_layers = num_layers
        self.activation = activation
        self.target_size = target_size
        self.noise = noise
        self.sparsity = sparsity
        control_list = [self.data_size, self.input_size, self.hidden_size, self.scale_factor, self.num_layers,
                        self.activation, self.target_size, self.noise]
        self.control = '_'.join(['{}' for _ in range(len(control_list))]).format(*control_list)
        if not check_exists(self.processed_folder):
            self.process()
        self.id, self.data, self.target = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)),
                                               mode='pickle')
        if self.mode == 'r':
            self.target_size = load(os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')
        elif self.mode == 'c':
            self.classes_counts = make_classes_counts(self.target)
            self.classes_to_labels, self.target_size = load(os.path.join(self.processed_folder, 'meta.pt'),
                                                            mode='pickle')
        else:
            raise ValueError('Not valid mode')

    def __getitem__(self, index):
        id, data, target = torch.tensor(self.id[index]), torch.tensor(self.data[index]), torch.tensor(
            self.target[index])
        input = {'id': id, 'data': data, 'target': target}
        return input

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed', self.mode, self.control)

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        train_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'test.pt'), mode='pickle')
        save(meta, os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}'.format(self.__class__.__name__, self.__len__(), self.root,
                                                                     self.split)
        return fmt_str

    def make_data(self):
        x = torch.randn((self.data_size, self.input_size))
        data_shape = [self.input_size]
        model = models.MLP(data_shape, self.hidden_size, self.scale_factor, self.num_layers, self.activation,
                           self.target_size)
        self.sparsify(model)
        model.train(False)
        with torch.no_grad():
            y = model.f(x)
            y = y + self.noise * torch.randn(y.size())
            if self.mode == 'c':
                y = torch.softmax(y, dim=-1)
                y = torch.distributions.categorical.Categorical(probs=y).sample()
            x, y = x.numpy(), y.numpy()
        perm = np.random.permutation(len(x))
        x, y = x[perm], y[perm]
        split_idx = int(x.shape[0] * 0.8)
        train_data, test_data = x[:split_idx].astype(np.float32), x[split_idx:].astype(np.float32)
        train_target, test_target = y[:split_idx].astype(np.int64), y[split_idx:].astype(np.int64)
        train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)
        if self.mode == 'r':
            target_size = 1
            return (train_id, train_data, train_target), (test_id, test_data, test_target), target_size
        elif self.mode == 'c':
            classes = list(map(str, list(range(self.target_size))))
            classes_to_labels = {classes[i]: i for i in range(len(classes))}
            target_size = len(classes)
            return (train_id, train_data, train_target), (test_id, test_data, test_target), (
                classes_to_labels, target_size)
        else:
            raise ValueError('Not valid mode')
        return

    def sparsify(self, model):
        model_state_dict = model.state_dict()
        for k, v in model.named_parameters():
            parameter_type = k.split('.')[-1]
            if 'weight' in parameter_type or 'bias' in parameter_type:
                sparsified_v = v.data
                output_size = sparsified_v.size(0)
                sparsified_size = int(output_size * self.sparsity)
                sparsified_v[-sparsified_size:] = 0
                model_state_dict[k] = sparsified_v
        model.load_state_dict(model_state_dict)
        return
