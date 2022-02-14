from .blob import Blob
from .friedman import Friedman
from .mlp import MLP
from .mnist import MNIST, FashionMNIST
from .cifar import CIFAR10, CIFAR100
from .svhn import SVHN
from .utils import *

__all__ = ('Blob', 'Friedman', 'MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'SVHN')
