import argparse
import copy
import datetime
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from data import fetch_dataset, make_data_loader
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import make_logger
from modules import Compression, SparsityIndex

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
        teacher_model_tag_list = [str(seeds[i]), '_'.join(cfg['control_name'].split('_')[:2])]
        cfg['teacher_model_tag'] = '_'.join([x for x in teacher_model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)
    data_loader = make_data_loader(dataset, cfg['model_name'])
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    result = resume('./output/model/{}_{}.pt'.format(cfg['teacher_model_tag'], 'best'))
    model.load_state_dict(result['model_state_dict'])
    sparsity_index = SparsityIndex(cfg['si_q'])
    compression = Compression(cfg['prune_ratio'], cfg['prune_mode'])
    optimizer = make_optimizer(model, cfg['model_name'])
    scheduler = make_scheduler(optimizer, cfg['model_name'])
    metric = Metric({'train': ['Loss'], 'test': ['Loss']})
    result = resume('./output/model/{}_{}.pt'.format(cfg['model_tag'], 'checkpoint'), resume_mode=cfg['resume_mode'])
    if result is None:
        last_iter = 1
        last_epoch = 1
        logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    else:
        last_iter = result['iter']
        last_epoch = result['epoch']
        model.load_state_dict(result['model_state_dict'])
        optimizer.load_state_dict(result['optimizer_state_dict'])
        scheduler.load_state_dict(result['scheduler_state_dict'])
        compression = result['compression']
        logger = result['logger']
    for iter in range(last_iter, cfg['prune_iters'] + 1):
        if last_epoch == 1:
            if cfg['prune_mode'][0] == 'once':
                result = resume('./output/model/{}_{}.pt'.format(cfg['teacher_model_tag'], 'best'))
            elif cfg['prune_mode'][0] in ['lt', 'si']:
                if iter == 1:
                    result = resume('./output/model/{}_{}.pt'.format(cfg['teacher_model_tag'], 'best'))
                else:
                    result = resume('./output/model/{}_{}_{}.pt'.format(cfg['model_tag'], iter - 1, 'best'))
            else:
                raise ValueError('Not valid prune mode')
            model.load_state_dict(result['model_state_dict'])
            if cfg['prune_mode'][0] in ['si']:
                sparsity_index.make_sparsity_index(model, compression.mask[-1])
            compression.prune(model, sparsity_index)
            compression.init(model)
            optimizer = make_optimizer(model, cfg['model_name'])
            scheduler = make_scheduler(optimizer, cfg['model_name'])
            metric = Metric({'train': ['Loss'], 'test': ['Loss']})
        for epoch in range(last_epoch, cfg[cfg['model_name']]['num_epochs'] + 1):
            train(data_loader['train'], model, compression, optimizer, metric, logger, iter, epoch)
            test(data_loader['test'], model, metric, logger, iter, epoch)
            scheduler.step()
            result = {'cfg': cfg, 'epoch': epoch + 1, 'iter': iter, 'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                      'compression': compression, 'logger': logger}
            save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
            if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
                metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
                shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                            './output/model/{}_{}_best.pt'.format(cfg['model_tag'], iter))
            logger.reset()
        last_epoch = 1
    return


def train(data_loader, model, compression, optimizer, metric, logger, iter, epoch):
    logger.safe(True)
    model.train(True)
    start_time = time.time()
    for i, input in enumerate(data_loader):
        input = collate(input)
        input_size = input['data'].size(0)
        input = to_device(input, cfg['device'])
        optimizer.zero_grad()
        output = model(input)
        output['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        compression.freeze_grad(model)
        optimizer.step()
        evaluation = metric.evaluate(metric.metric_name['train'], input, output)
        logger.append(evaluation, 'train', n=input_size)
        if i % int((len(data_loader) * cfg['log_interval']) + 1) == 0:
            batch_time = (time.time() - start_time) / (i + 1)
            lr = optimizer.param_groups[0]['lr']
            epoch_finished_time = datetime.timedelta(seconds=round(batch_time * (len(data_loader) - i - 1)))
            exp_finished_time = (epoch_finished_time + datetime.timedelta(
                seconds=round((cfg[cfg['model_name']]['num_epochs'] - epoch) * batch_time * len(data_loader)))) + \
                                datetime.timedelta(seconds=round(batch_time * len(data_loader) *
                                                                 cfg[cfg['model_name']]['num_epochs'] *
                                                                 (cfg['prune_iters'] - iter)))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / len(data_loader)),
                             'Train Iter: {}/{}'.format(iter, cfg['prune_iters']),
                             'Learning rate: {:.6f}'.format(lr), 'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
    logger.safe(False)
    return


def test(data_loader, model, metric, logger, iter, epoch):
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
                         'Test Iter: {}/{}'.format(iter, cfg['prune_iters'])]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    logger.safe(False)
    return


if __name__ == "__main__":
    main()
