import argparse
import itertools

parser = argparse.ArgumentParser(description='config')
parser.add_argument('--run', default='train', type=str)
parser.add_argument('--init_gpu', default=0, type=int)
parser.add_argument('--num_gpus', default=4, type=int)
parser.add_argument('--world_size', default=1, type=int)
parser.add_argument('--init_seed', default=0, type=int)
parser.add_argument('--round', default=4, type=int)
parser.add_argument('--experiment_step', default=1, type=int)
parser.add_argument('--num_experiments', default=1, type=int)
parser.add_argument('--resume_mode', default=0, type=int)
parser.add_argument('--mode', default=None, type=str)
parser.add_argument('--split_round', default=65535, type=int)
parser.add_argument('--model', default=0, type=int)
args = vars(parser.parse_args())


def sort_fuc(control):
    control_name = control[-1]
    control_name_list = control_name.split('_')
    idx = '_'.join([control_name_list[0], control_name_list[1]])
    return idx


def make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    control_names = [control_names]
    controls = script_name + init_seeds + world_size + num_experiments + resume_mode + control_names
    controls = list(itertools.product(*controls))
    controls.sort(key=sort_fuc)
    return controls


def main():
    run = args['run']
    init_gpu = args['init_gpu']
    num_gpus = args['num_gpus']
    world_size = args['world_size']
    round = args['round']
    experiment_step = args['experiment_step']
    init_seed = args['init_seed']
    num_experiments = args['num_experiments']
    resume_mode = args['resume_mode']
    mode = args['mode']
    split_round = args['split_round']
    model = args['model']
    gpu_ids = [','.join(str(i) for i in list(range(x, x + world_size))) for x in
               list(range(init_gpu, init_gpu + num_gpus, world_size))]
    init_seeds = [list(range(init_seed, init_seed + num_experiments, experiment_step))]
    world_size = [[world_size]]
    num_experiments = [[experiment_step]]
    resume_mode = [[resume_mode]]
    filename = '{}_{}'.format(run, mode)
    if model == 0:
        model_names = ['linear', 'mlp']
        prune_iters = ['30']
    elif model == 1:
        model_names = ['cnn']
        prune_iters = ['30']
    elif model == 2:
        model_names = ['resnet18']
        prune_iters = ['15']
    elif model == 3:
        model_names = ['wresnet28x8']
        prune_iters = ['15']
    elif model == 4:
        model_names = ['resnet50']
        prune_iters = ['15']
    else:
        raise ValueError('Not valid model')
    if mode == 'os':
        script_name = [['{}_classifier.py'.format(run)]]
        # data_names = ['FashionMNIST', 'CIFAR10', 'CIFAR100', 'ImageNet']
        # data_names = ['CIFAR100']
        data_names = ['TinyImageNet']
        prune_scope = ['global']
        prune_mode = ['os-0.2']
        control_name = [[data_names, model_names, prune_iters, prune_scope, prune_mode]]
        controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
    elif mode == 'lt':
        script_name = [['{}_classifier.py'.format(run)]]
        # data_names = ['FashionMNIST', 'CIFAR10', 'CIFAR100', 'ImageNet']
        # data_names = ['CIFAR100']
        data_names = ['TinyImageNet']
        prune_scope = ['global']
        prune_mode = ['lt-0.2']
        control_name = [[data_names, model_names, prune_iters, prune_scope, prune_mode]]
        controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
    elif mode == 'si':
        script_name = [['{}_classifier.py'.format(run)]]
        # data_names = ['FashionMNIST', 'CIFAR10', 'CIFAR100', 'ImageNet']
        # data_names = ['CIFAR100']
        data_names = ['TinyImageNet']
        prune_scope = ['global']
        prune_mode = ['si-0.5-1-0-1', 'si-1-2-0-1']
        control_name = [[data_names, model_names, prune_iters, prune_scope, prune_mode]]
        controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
    elif mode == 'scope':
        script_name = [['{}_classifier.py'.format(run)]]
        # data_names = ['FashionMNIST', 'CIFAR10', 'CIFAR100', 'ImageNet']
        data_names = ['TinyImageNet']
        prune_scope = ['neuron', 'layer']
        prune_mode = ['si-0.5-1-0-1', 'si-1-2-0-1', 'lt-0.2', 'os-0.2']
        control_name = [[data_names, model_names, prune_iters, prune_scope, prune_mode]]
        controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
    elif mode == 'si-p':
        script_name = [['{}_classifier.py'.format(run)]]
        data_names = ['CIFAR10']
        prune_scope = ['global']
        prune_mode = ['si-0.2-1-0-1', 'si-0.4-1-0-1', 'si-0.6-1-0-1', 'si-0.8-1-0-1']
        control_name = [[data_names, model_names, prune_iters, prune_scope, prune_mode]]
        controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
    elif mode == 'si-q':
        script_name = [['{}_classifier.py'.format(run)]]
        data_names = ['CIFAR10']
        prune_scope = ['global']
        prune_mode = ['si-1-1.2-0-1', 'si-1-1.4-0-1', 'si-1-1.6-0-1', 'si-1-1.8-0-1']
        control_name = [[data_names, model_names, prune_iters, prune_scope, prune_mode]]
        controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
    elif mode == 'si-eta_m':
        script_name = [['{}_classifier.py'.format(run)]]
        data_names = ['CIFAR10']
        prune_scope = ['global']
        prune_mode = ['si-0.5-1-0.001-1', 'si-0.5-1-0.01-1', 'si-0.5-1-0.1-1', 'si-0.5-1-1-1']
        control_name = [[data_names, model_names, prune_iters, prune_scope, prune_mode]]
        controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
    elif mode == 'si-gamma':
        script_name = [['{}_classifier.py'.format(run)]]
        data_names = ['CIFAR10']
        prune_scope = ['global']
        prune_mode = ['si-0.5-1-0-3', 'si-0.5-1-0-5', 'si-0.5-1-0-7', 'si-0.5-1-0-9']
        control_name = [[data_names, model_names, prune_iters, prune_scope, prune_mode]]
        controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
    else:
        raise ValueError('Not valid mode')
    s = '#!/bin/bash\n'
    j = 1
    k = 1
    for i in range(len(controls)):
        controls[i] = list(controls[i])
        s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --init_seed {} --world_size {} --num_experiments {} ' \
                '--resume_mode {} --control_name {}&\n'.format(gpu_ids[i % len(gpu_ids)], *controls[i])
        if i % round == round - 1:
            s = s[:-2] + '\nwait\n'
            if j % split_round == 0:
                print(s)
                run_file = open('{}_{}.sh'.format(filename, k), 'w')
                run_file.write(s)
                run_file.close()
                s = '#!/bin/bash\n'
                k = k + 1
            j = j + 1
    if s != '#!/bin/bash\n':
        if s[-5:-1] != 'wait':
            s = s + 'wait\n'
        print(s)
        run_file = open('{}_{}.sh'.format(filename, k), 'w')
        run_file.write(s)
        run_file.close()
    return


if __name__ == '__main__':
    main()
