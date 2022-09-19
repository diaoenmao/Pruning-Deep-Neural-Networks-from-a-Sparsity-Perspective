#!/bin/bash

modes=('os', 'lt', 'si')
resume_mode=0
num_gpus=1
round=8
num_experiments=4

# os
python make.py --mode os --run train --resume_mode 0 --num_gpus 4 --round 24 --num_experiments 4
python make.py --mode os --run test --resume_mode 0 --num_gpus 4 --round 24 --num_experiments 4

# lt
python make.py --mode lt --run train --resume_mode 0 --num_gpus 4 --round 24 --num_experiments 4
python make.py --mode lt --run test --resume_mode 0 --num_gpus 4 --round 24 --num_experiments 4

# si
python make.py --mode si --run train --resume_mode 0 --num_gpus 4 --round 24 --num_experiments 4
python make.py --mode si --run test --resume_mode 0 --num_gpus 4 --round 24 --num_experiments 4

# si-pq
python make.py --mode si-pq --run train --resume_mode 0 --num_gpus 4 --round 24 --num_experiments 4 --split_round 4
python make.py --mode si-pq --run test --resume_mode 0 --num_gpus 4 --round 24 --num_experiments 4 --split_round 4

# si-eta_m
python make.py --mode si-eta_m --run train --resume_mode 0 --num_gpus 4 --round 24 --num_experiments 4
python make.py --mode si-eta_m --run test --resume_mode 0 --num_gpus 4 --round 24 --num_experiments 4

# si-gamma
python make.py --mode si-gamma --run train --resume_mode 0 --num_gpus 4 --round 24 --num_experiments 4
python make.py --mode si-gamma --run test --resume_mode 0 --num_gpus 4 --round 24 --num_experiments 4