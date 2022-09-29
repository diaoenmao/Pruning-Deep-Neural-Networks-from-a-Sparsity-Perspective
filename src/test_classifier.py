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
from modules import Compression, Mask, SparsityIndex

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
    model_path = os.path.join('output', 'model')
    checkpoint_path = os.path.join(model_path, '{}_{}.pt'.format(cfg['model_tag'], 'checkpoint'))
    result_path = os.path.join('output', 'result', '{}.pt'.format(cfg['model_tag']))
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    mask = Mask(to_device(model.state_dict(), 'cpu'))
    metric = Metric({'test': ['Loss', 'Accuracy']})
    result = resume(checkpoint_path)
    last_iter = result['iter']
    last_epoch = result['epoch']
    model_state_dict = result['model_state_dict']
    mask_state_dict = result['mask_state_dict']
    sparsity_index = result['sparsity_index']
    sparsity_index.reset()
    sparsity_index_pruned = SparsityIndex(cfg['p'], cfg['q'])
    compression = Compression(cfg['prune_scope'], cfg['prune_mode'])
    data_loader = make_data_loader(dataset, cfg['model_name'])
    test_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    test_pruned_logger = make_logger(os.path.join('output', 'runs', 'test_pruned_{}'.format(cfg['model_tag'])))
    for iter in range(len(model_state_dict)):
        model.load_state_dict(model_state_dict[iter])
        mask.load_state_dict(mask_state_dict[iter])
        test_logger.save(True)
        test(data_loader['test'], model, metric, test_logger, iter, last_epoch[iter])
        test_logger.save(False)
        sparsity_index.make_sparsity_index(model, mask)
        if iter < len(model_state_dict) - 1:
            if cfg['prune_mode'][0] in ['os']:
                model.load_state_dict(model_state_dict[0])
            mask.load_state_dict(mask_state_dict[iter + 1])
            sparsity_index_pruned.make_sparsity_index(model, mask)
            compression.init(model, mask, to_device(model.state_dict(), 'cpu'))
            test_pruned_logger.save(True)
            test(data_loader['test'], model, metric, test_pruned_logger, iter, last_epoch[iter])
            test_pruned_logger.save(False)
    result = resume(checkpoint_path)
    train_logger = result['logger'] if 'logger' in result else None
    result = {'cfg': cfg, 'iter': last_iter, 'epoch': last_epoch,
              'logger': {'train': train_logger, 'test': test_logger, 'test-pruned': test_pruned_logger},
              'mask_state_dict': mask_state_dict, 'sparsity_index': sparsity_index,
              'sparsity_index_pruned': sparsity_index_pruned}
    save(result, result_path)
    return


def test(data_loader, model, metric, logger, iter, epoch):
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
                         'Test Iter: {}/{}'.format(iter, cfg['prune_iters'])]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    return


if __name__ == "__main__":
    main()
