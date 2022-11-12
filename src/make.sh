#!/bin/bash

resume_mode=0
num_gpus=1
round=8
num_experiments=4
world_size=1
model=0

# os
python make.py --mode os --model $model --run train --resume_mode 0 --num_gpus 4 --world_size $world_size --round 16 --num_experiments 4
python make.py --mode os --model $model --run test --resume_mode 0 --num_gpus 4 --world_size $world_size --round 16 --num_experiments 4

# lt
python make.py --mode lt --model $model --run train --resume_mode 0 --num_gpus 4 --world_size $world_size --round 16 --num_experiments 4
python make.py --mode lt --model $model --run test --resume_mode 0 --num_gpus 4 --world_size $world_size --round 16 --num_experiments 4

# si
python make.py --mode si --model $model --run train --resume_mode 0 --num_gpus 4 --world_size $world_size --round 16 --num_experiments 4 --split_round 2
python make.py --mode si --model $model --run test --resume_mode 0 --num_gpus 4 --world_size $world_size --round 16 --num_experiments 4 --split_round 2

# scope
python make.py --mode scope --model $model --run train --resume_mode 0 --num_gpus 4 --world_size $world_size --round 16 --num_experiments 4 --split_round 2
python make.py --mode scope --model $model --run test --resume_mode 0 --num_gpus 4 --world_size $world_size --round 16 --num_experiments 4 --split_round 2

# si-p
python make.py --mode si-p --model $model --run train --resume_mode 0 --num_gpus 4 --world_size $world_size --round 16 --num_experiments 4 --split_round 2
python make.py --mode si-p --model $model --run test --resume_mode 0 --num_gpus 4 --world_size $world_size --round 16 --num_experiments 4 --split_round 2

# si-q
python make.py --mode si-q --model $model --run train --resume_mode 0 --num_gpus 4 --world_size $world_size --round 16 --num_experiments 4 --split_round 2
python make.py --mode si-q --model $model --run test --resume_mode 0 --num_gpus 4 --world_size $world_size --round 16 --num_experiments 4 --split_round 2

# si-eta_m
python make.py --mode si-eta_m --model $model --run train --resume_mode 0 --num_gpus 4 --world_size $world_size --round 16 --num_experiments 4 --split_round 2
python make.py --mode si-eta_m --model $model --run test --resume_mode 0 --num_gpus 4 --world_size $world_size --round 16 --num_experiments 4 --split_round 2

# si-gamma
python make.py --mode si-gamma --model $model --run train --resume_mode 0 --num_gpus 4 --world_size $world_size --round 16 --num_experiments 4 --split_round 2
python make.py --mode si-gamma --model $model --run test --resume_mode 0 --num_gpus 4 --world_size $world_size --round 16 --num_experiments 4 --split_round 2