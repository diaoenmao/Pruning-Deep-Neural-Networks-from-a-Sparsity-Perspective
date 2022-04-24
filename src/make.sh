python make.py --mode init --model mlp --data MNIST --run make --round 8 --split_round 1
python make.py --mode init --model mlp --data CIFAR10 --run make --round 8 --split_round 1

python make.py --mode teacher --model mlp --data MNIST --run train --round 8 --split_round 1
python make.py --mode teacher --model mlp --data MNIST --run test --round 8 --split_round 1

python make.py --mode teacher --model mlp --data FashionMNIST --run train --round 8 --split_round 1
python make.py --mode teacher --model mlp --data FashionMNIST --run test --round 8 --split_round 1

python make.py --mode teacher --model mlp --data CIFAR10 --run train --round 8 --split_round 1
python make.py --mode teacher --model mlp --data CIFAR10 --run test --round 8 --split_round 1

python make.py --mode teacher --model mlp --data SVHN --run train --round 8 --split_round 1
python make.py --mode teacher --model mlp --data SVHN --run test --round 8 --split_round 1

#python make.py --mode once --model mlp --data MNIST --run train --round 8 --split_round 1
#python make.py --mode once --model mlp --data MNIST --run test --round 8 --split_round 1
