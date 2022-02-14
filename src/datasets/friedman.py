import numpy as np
import os
import torch
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from .utils import make_classes_counts


class Friedman(Dataset):
    data_name = 'Friedman'

    def __init__(self, root, split, num_samples=100, num_features=10, noise=1.0):
        self.root = os.path.expanduser(root)
        self.split = split
        self.num_samples = num_samples
        self.num_features = num_features
        self.noise = noise
        control_list = [self.num_samples, self.num_features, self.noise]
        self.control = '_'.join(['{}' for _ in range(len(control_list))]).format(*control_list)
        if not check_exists(self.processed_folder):
            self.process()
        self.id, self.data, self.target = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)),
                                          mode='pickle')
        self.target_size = load(os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')

    def __getitem__(self, index):
        id, data, target = torch.tensor(self.id[index]), torch.tensor(self.data[index]), torch.tensor(
            self.target[index])
        input = {'id': id, 'data': data, 'target': target}
        return input

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed', self.control)

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
        from sklearn.datasets import make_friedman1
        x, y = make_friedman1(n_samples=self.num_samples, n_features=self.num_features, noise=self.noise,
                              random_state=0)
        perm = np.random.permutation(len(x))
        x, y = x[perm], y[perm]
        split_idx = int(x.shape[0] * 0.8)
        train_data, test_data = x[:split_idx].astype(np.float32), x[split_idx:].astype(np.float32)
        train_target, test_target = y[:split_idx].astype(np.int64), y[split_idx:].astype(np.int64)
        train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)
        target_size = 1
        return (train_id, train_data, train_target), (test_id, test_data, test_target), target_size
