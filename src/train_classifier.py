import argparse
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
    best_path = os.path.join(model_path, '{}_{}.pt'.format(cfg['model_tag'], 'best'))
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)
    data_loader = make_data_loader(dataset, cfg['model_name'])
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    optimizer = make_optimizer(model.parameters(), cfg['model_name'])
    scheduler = make_scheduler(optimizer, cfg['model_name'])
    metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    mask = Mask(to_device(model.state_dict(), 'cpu'))
    result = resume(checkpoint_path, resume_mode=cfg['resume_mode'])
    if result is None:
        last_iter = 0
        last_epoch = [0]
        logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
        init_model_state_dict = to_device(model.state_dict(), 'cpu')
        model_state_dict = [to_device(model.state_dict(), 'cpu')]
        mask_state_dict = [to_device(mask.state_dict(), 'cpu')]
    else:
        last_iter = result['iter']
        last_epoch = result['epoch']
        init_model_state_dict = result['init_model_state_dict']
        model_state_dict = result['model_state_dict']
        model.load_state_dict(model_state_dict[-1])
        optimizer.load_state_dict(result['optimizer_state_dict'])
        scheduler.load_state_dict(result['scheduler_state_dict'])
        metric.load_state_dict(result['metric_state_dict'])
        mask_state_dict = result['mask_state_dict']
        mask.load_state_dict(mask_state_dict[-1])
        logger = result['logger']
    if cfg['world_size'] > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(cfg['world_size'])))
    compression = Compression(cfg['prune_scope'], cfg['prune_mode'])
    sparsity_index = SparsityIndex(cfg['p'], cfg['q'])
    for iter in range(last_iter, cfg['prune_iters'] + 1):
        if last_epoch[-1] == 0:
            metric.reset()
            if cfg['world_size'] > 1:
                compression.init(model.module, mask, init_model_state_dict)
            else:
                compression.init(model, mask, init_model_state_dict)
            optimizer = make_optimizer(model.parameters(), cfg['model_name'])
            scheduler = make_scheduler(optimizer, cfg['model_name'])
            if iter > 0:
                if cfg['world_size'] > 1:
                    model_state_dict.append(to_device(model.module.state_dict(), 'cpu'))
                else:
                    model_state_dict.append(to_device(model.state_dict(), 'cpu'))
                mask_state_dict.append(mask.state_dict())
        for epoch in range(last_epoch[-1] + 1, cfg[cfg['model_name']]['num_epochs'] + 1):
            logger.save(True)
            train(data_loader['train'], model, optimizer, mask, metric, logger, iter, epoch)
            test(data_loader['test'], model, metric, logger, iter, epoch)
            logger.save(False)
            scheduler.step()
            last_epoch[-1] = epoch
            if cfg['world_size'] > 1:
                model_state_dict[-1] = to_device(model.module.state_dict(), 'cpu')
            else:
                model_state_dict[-1] = to_device(model.state_dict(), 'cpu')
            result = {'cfg': cfg, 'iter': iter, 'epoch': last_epoch, 'init_model_state_dict': init_model_state_dict,
                      'model_state_dict': model_state_dict, 'optimizer_state_dict': optimizer.state_dict(),
                      'scheduler_state_dict': scheduler.state_dict(), 'metric_state_dict': metric.state_dict(),
                      'mask_state_dict': mask_state_dict, 'logger': logger, 'sparsity_index': sparsity_index}
            save(result, checkpoint_path)
            if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
                metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
                shutil.copy(checkpoint_path, best_path)
            logger.reset()
        if iter < cfg['prune_iters']:
            last_epoch.append(0)
            result = resume(best_path, verbose=False)
            if cfg['prune_mode'][0] in ['os']:
                if cfg['world_size'] > 1:
                    model.module.load_state_dict(result['model_state_dict'][0])
                else:
                    model.load_state_dict(result['model_state_dict'][0])
            elif cfg['prune_mode'][0] in ['lt', 'si']:
                if cfg['world_size'] > 1:
                    model.module.load_state_dict(result['model_state_dict'][-1])
                else:
                    model.load_state_dict(result['model_state_dict'][-1])
            else:
                raise ValueError('Not valid prune mode')
            if cfg['world_size'] > 1:
                sparsity_index.make_sparsity_index(model.module, mask)
                compression.compress(model.module, mask, sparsity_index)
            else:
                sparsity_index.make_sparsity_index(model, mask)
                compression.compress(model, mask, sparsity_index)
    return


def train(data_loader, model, optimizer, mask, metric, logger, iter, epoch):
    model.train(True)
    start_time = time.time()
    for i, input in enumerate(data_loader):
        input = collate(input)
        input_size = input['data'].size(0)
        input = to_device(input, cfg['device'])
        optimizer.zero_grad()
        output = model(input)
        output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
        output['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        if cfg['world_size'] > 1:
            mask.freeze_grad(model.module)
        else:
            mask.freeze_grad(model)
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
