python make.py --mode lt --model mlp --data MNIST --run train --round 8 --split_round 1
python make.py --mode lt --model mlp --data MNIST --run test --round 8 --split_round 1

python make.py --mode lt --model mlp --data FashionMNIST --run train --round 8 --split_round 1
python make.py --mode lt --model mlp --data FashionMNIST --run test --round 8 --split_round 1

python make.py --mode lt --model mlp --data CIFAR10 --run train --round 8 --split_round 1
python make.py --mode lt --model mlp --data CIFAR10 --run test --round 8 --split_round 1

python make.py --mode lt --model mlp --data CIFAR100 --run train --round 8 --split_round 1
python make.py --mode lt --model mlp --data CIFAR100 --run test --round 8 --split_round 1