import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import models
from config import cfg, process_args
from data import fetch_dataset, make_data_loader
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, resume, collate
from logger import make_logger
from modules import make_sparsity_index

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    result = resume(cfg['model_tag'], load_tag='best')
    last_epoch = result['epoch']
    model.load_state_dict(result['model_state_dict'])
    p = torch.arange(0.1, 0.9, 0.1)
    sparsity_index = test(p, model)
    result = {'cfg': cfg, 'epoch': last_epoch, 'p': p, 'sparsity_index': sparsity_index}
    save(result, './output/sparsity_index/{}.pt'.format(cfg['model_tag']))
    return


def test(p, model):
    sparsity_index = []
    for i in range(len(p)):
        sparsity_index_i = make_sparsity_index(model, p[i])
        sparsity_index.append(sparsity_index_i)
    return sparsity_index


if __name__ == "__main__":
    main()
