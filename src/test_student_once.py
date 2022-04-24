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
from modules import SparsityIndex

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
    data_loader = make_data_loader(dataset, 'teacher')
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    metric = Metric({'train': ['Loss'], 'test': ['Loss']})
    sparsity_index = SparsityIndex(cfg['q'])
    test_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    last_iter = 1
    for iter in range(last_iter, cfg['num_iters'] + 1):
        result = resume('./output/model/{}_{}_{}.pt'.format(cfg['model_tag'], iter, 'best'))
        if result is not None:
            last_epoch = result['epoch']
            model.load_state_dict(result['model_state_dict'])
            compression = result['compression']
            test(data_loader['test'], model, compression, sparsity_index, metric, test_logger, iter, last_epoch)
        test_logger.reset()
    result = resume('./output/model/{}_{}.pt'.format(cfg['model_tag'], 'checkpoint'))
    train_logger = result['logger']
    compression = result['compression']
    compression.init_model_state_dict = None
    result = {'cfg': cfg, 'compression': compression, 'sparsity_index': sparsity_index,
              'logger': {'train': train_logger, 'test': test_logger}}
    save(result, './output/result/{}.pt'.format(cfg['model_tag']))
    return


def test(data_loader, model, compression, sparsity_index, metric, logger, iter, epoch):
    logger.safe(True)
    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']),
                         'Test Epoch: {}({:.0f}%)'.format(epoch, 100.),
                         'Test Iter: {}/{}'.format(iter, cfg['num_iters'])]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    sparsity_index.make_sparsity_index(model, compression.mask[iter - 1])
    logger.safe(False)
    return


if __name__ == "__main__":
    main()
