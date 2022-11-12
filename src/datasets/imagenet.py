import numpy as np
import os
import shutil
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import check_exists, save, load, makedir_exist_ok
from .utils import IMG_EXTENSIONS
from .utils import extract_file, make_classes_counts, make_data_target


class ImageNet(Dataset):
    data_name = 'ImageNet'
    file = [('http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar',
             '1d675b47d978889d74fa0da5fadfb00e'),
            ('http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar',
             '29b22e2961454d5413ddabcf34fc5622'),
            ('http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t12.tar.gz',
             'fa75699e90414af021442c21a62c3abf')]

    def __init__(self, root, split, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        if not check_exists(self.processed_folder):
            self.process()
        self.id, self.data, self.target = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)),
                                               mode='pickle')
        self.other = {}
        self.classes_counts = make_classes_counts(self.target)
        self.classes_to_labels, self.target_size = load(os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')

    def __getitem__(self, index):
        id, data, target = torch.tensor(self.id[index]), Image.open(self.data[index], mode='r').convert('RGB'), \
                           torch.tensor(self.target[index])
        input = {'id': id, 'data': data, 'target': target}
        other = {k: torch.tensor(self.other[k][index]) for k in self.other}
        input = {**input, **other}

        if self.transform is not None:
            input = self.transform(input)
        return input

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            raise RuntimeError('Dataset not found')
        train_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'test.pt'), mode='pickle')
        save(meta, os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
        return fmt_str

    def make_data(self):
        train_path = os.path.join(self.raw_folder, 'ILSVRC2012_img_train')
        test_path = os.path.join(self.raw_folder, 'ILSVRC2012_img_val')
        meta_path = os.path.join(self.raw_folder, 'ILSVRC2012_devkit_t12')
        extract_file(os.path.join(self.raw_folder, 'ILSVRC2012_img_train.tar'), train_path)
        extract_file(os.path.join(self.raw_folder, 'ILSVRC2012_img_val.tar'), test_path)
        extract_file(os.path.join(self.raw_folder, 'ILSVRC2012_devkit_t12.tar'), meta_path)
        for archive in [os.path.join(train_path, archive) for archive in os.listdir(train_path)]:
            extract_file(archive, os.path.splitext(archive)[0], delete=True)
        classes_to_labels, target_size = make_meta(meta_path)
        with open(os.path.join(meta_path, 'data', 'ILSVRC2012_validation_ground_truth.txt'), 'r') as f:
            test_id = f.readlines()
        test_id = [int(i) for i in test_id]
        test_data = sorted([os.path.join(test_path, file) for file in os.listdir(test_path)])
        test_wnid = []
        class_names = list(classes_to_labels.keys())
        class_ids = list(classes_to_labels.values())
        for test_id_i in test_id:
            test_wnid_i = class_names[class_ids.index(test_id_i - 1)]
            test_wnid.append(test_wnid_i)
        for test_wnid_i in set(test_wnid):
            makedir_exist_ok(os.path.join(test_path, test_wnid_i))
        for test_wnid_i, test_data in zip(test_wnid, test_data):
            shutil.move(test_data, os.path.join(test_path, test_wnid_i, os.path.basename(test_data)))
        train_data, train_target = make_data_target(os.path.join(self.raw_folder, 'ILSVRC2012_img_train'),
                                                    classes_to_labels, IMG_EXTENSIONS)
        test_data, test_target = make_data_target(os.path.join(self.raw_folder, 'ILSVRC2012_img_val'),
                                                  classes_to_labels, IMG_EXTENSIONS)
        train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)
        return (train_id, train_data, train_target), (test_id, test_data, test_target), (classes_to_labels, target_size)


def make_meta(path):
    import scipy.io as sio
    meta = sio.loadmat(os.path.join(path, 'data', 'meta.mat'), squeeze_me=True)['synsets']
    num_children = list(zip(*meta))[4]
    classes = [meta[i] for (i, n) in enumerate(num_children) if n == 0]
    classes_to_labels = {}
    for i in range(len(classes)):
        classes_to_labels[classes[i][1]] = classes[i][0] - 1
    target_size = len(classes)
    return classes_to_labels, target_size
