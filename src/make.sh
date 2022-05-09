# MLP
python make.py --mode init --model mlp --data MNIST --run make --round 16 --split_round 1
python make.py --mode init --model mlp --data CIFAR10 --run make --round 16 --split_round 1

python make.py --mode teacher --model mlp --data MNIST --run train --round 16 --split_round 1
python make.py --mode teacher --model mlp --data MNIST --run test --round 16 --split_round 1

python make.py --mode teacher --model mlp --data FashionMNIST --run train --round 16 --split_round 1
python make.py --mode teacher --model mlp --data FashionMNIST --run test --round 16 --split_round 1

python make.py --mode teacher --model mlp --data CIFAR10 --run train --round 16 --split_round 1
python make.py --mode teacher --model mlp --data CIFAR10 --run test --round 16 --split_round 1

python make.py --mode teacher --model mlp --data SVHN --run train --round 16 --split_round 1
python make.py --mode teacher --model mlp --data SVHN --run test --round 16 --split_round 1

python make.py --mode once --model mlp --data MNIST --run train --round 16 --split_round 1
python make.py --mode once --model mlp --data MNIST --run test --round 16 --split_round 1

python make.py --mode once --model mlp --data FashionMNIST --run train --round 16 --split_round 1
python make.py --mode once --model mlp --data FashionMNIST --run test --round 16 --split_round 1

python make.py --mode once --model mlp --data CIFAR10 --run train --round 16 --split_round 1
python make.py --mode once --model mlp --data CIFAR10 --run test --round 16 --split_round 1

python make.py --mode once --model mlp --data SVHN --run train --round 16 --split_round 1
python make.py --mode once --model mlp --data SVHN --run test --round 16 --split_round 1

python make.py --mode lt --model mlp --data MNIST --run train --round 16 --split_round 1
python make.py --mode lt --model mlp --data MNIST --run test --round 16 --split_round 1

python make.py --mode lt --model mlp --data FashionMNIST --run train --round 16 --split_round 1
python make.py --mode lt --model mlp --data FashionMNIST --run test --round 16 --split_round 1

python make.py --mode lt --model mlp --data CIFAR10 --run train --round 16 --split_round 1
python make.py --mode lt --model mlp --data CIFAR10 --run test --round 16 --split_round 1

python make.py --mode lt --model mlp --data SVHN --run train --round 16 --split_round 1
python make.py --mode lt --model mlp --data SVHN --run test --round 16 --split_round 1

python make.py --mode si --model mlp --data MNIST --run train --round 16 --split_round 1
python make.py --mode si --model mlp --data MNIST --run test --round 16 --split_round 1

python make.py --mode si --model mlp --data FashionMNIST --run train --round 16 --split_round 1
python make.py --mode si --model mlp --data FashionMNIST --run test --round 16 --split_round 1

python make.py --mode si --model mlp --data CIFAR10 --run train --round 16 --split_round 1
python make.py --mode si --model mlp --data CIFAR10 --run test --round 16 --split_round 1

python make.py --mode si --model mlp --data SVHN --run train --round 16 --split_round 1
python make.py --mode si --model mlp --data SVHN --run test --round 16 --split_round 1

# Ablation
python make.py --mode si-q --model mlp --data MNIST --run train --round 16 --split_round 1
python make.py --mode si-q --model mlp --data MNIST --run test --round 16 --split_round 1

python make.py --mode si-eta --model mlp --data MNIST --run train --round 16 --split_round 1
python make.py --mode si-eta --model mlp --data MNIST --run test --round 16 --split_round 1

python make.py --mode si-q --model mlp --data CIFAR10 --run train --round 16 --split_round 1
python make.py --mode si-q --model mlp --data CIFAR10 --run test --round 16 --split_round 1

python make.py --mode si-eta --model mlp --data CIFAR10 --run train --round 16 --split_round 1
python make.py --mode si-eta --model mlp --data CIFAR10 --run test --round 16 --split_round 1

# CNN
python make.py --mode init --model cnn --data CIFAR10 --run make --num_experiments 3 --round 16 --split_round 1 --init_seed 2

python make.py --mode teacher --model cnn --data CIFAR10 --run train --num_experiments 3 --round 16 --split_round 1 --init_seed 2
python make.py --mode teacher --model cnn --data CIFAR10 --run test --num_experiments 3 --round 16 --split_round 1 --init_seed 2

python make.py --mode once --model cnn --data CIFAR10 --run train --num_experiments 4 --round 16 --split_round 1
python make.py --mode once --model cnn --data CIFAR10 --run test --num_experiments 4 --round 16 --split_round 1

python make.py --mode lt --model cnn --data CIFAR10 --run train --num_experiments 4 --round 16 --split_round 1
python make.py --mode lt --model cnn --data CIFAR10 --run test --num_experiments 4 --round 16 --split_round 1

python make.py --mode si --model cnn --data CIFAR10 --run train --num_experiments 4 --round 16 --split_round 1
python make.py --mode si --model cnn --data CIFAR10 --run test --num_experiments 4 --round 16 --split_round 1

# ResNet
python make.py --mode init --model resnet18 --data CIFAR10 --run make --num_experiments 3 --round 16 --split_round 1 --init_gpu 1 --init_seed 2

python make.py --mode teacher --model resnet18 --data CIFAR10 --run train --num_experiments 3 --round 16 --split_round 1 --init_gpu 1 --init_seed 2
python make.py --mode teacher --model resnet18 --data CIFAR10 --run test --num_experiments 3 --round 16 --split_round 1 --init_gpu 1 --init_seed 2

python make.py --mode once --model resnet18 --data CIFAR10 --run train --num_experiments 4 --round 16 --split_round 1
python make.py --mode once --model resnet18 --data CIFAR10 --run test --num_experiments 4 --round 16 --split_round 1

python make.py --mode lt --model resnet18 --data CIFAR10 --run train --num_experiments 4 --round 16 --split_round 1
python make.py --mode lt --model resnet18 --data CIFAR10 --run test --num_experiments 4 --round 16 --split_round 1

python make.py --mode si --model resnet18 --data CIFAR10 --run train --num_experiments 4 --round 16 --split_round 1
python make.py --mode si --model resnet18 --data CIFAR10 --run test --num_experiments 4 --round 16 --split_round 1