import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import models
from config import cfg, process_args
from data import fetch_dataset
from utils import save, process_control, process_dataset, makedir_exist_ok

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)

init_path = os.path.join('output', 'init')

if __name__ == "__main__":
    makedir_exist_ok(init_path)
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        cfg['seed'] = str(seeds[i])
        model_tag_list = [str(cfg['seed']), cfg['control_name']]
        model_tag = '_'.join([x for x in model_tag_list if x])
        dataset = fetch_dataset(cfg['data_name'])
        process_dataset(dataset)
        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        init = {'cfg': cfg, 'model_state_dict': model.state_dict()}
        save(init, os.path.join(init_path, '{}.pt'.format(model_tag)))
