python train_student.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name MNIST_mlp-256-1-4-relu_30_si-0.5-1_si-global&
python train_student.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name MNIST_mlp-256-1-4-relu_30_si-0.5-2_si-global&
python train_student.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name MNIST_mlp-256-1-4-relu_30_si-0.5-3_si-global&
python train_student.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name MNIST_mlp-256-1-4-relu_30_si-0.5-4_si-global&
python train_student.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name MNIST_mlp-256-1-4-relu_30_si-0.5-5_si-global
wait
python test_student.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name MNIST_mlp-256-1-4-relu_30_si-0.5-1_si-global&
python test_student.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name MNIST_mlp-256-1-4-relu_30_si-0.5-2_si-global&
python test_student.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name MNIST_mlp-256-1-4-relu_30_si-0.5-3_si-global&
python test_student.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name MNIST_mlp-256-1-4-relu_30_si-0.5-4_si-global&
python test_student.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --control_name MNIST_mlp-256-1-4-relu_30_si-0.5-5_si-global
