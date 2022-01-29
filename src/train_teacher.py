import argparse
import datetime
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from data import fetch_dataset, make_data_loader, split_dataset, make_split_dataset
from metrics import Metric
from utils import save, process_control, process_dataset, make_optimizer, make_scheduler, resume, \
    make_model_name
from logger import make_logger
from modules import Teacher

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
    dataset = fetch_dataset(cfg['teacher_data_name'])
    process_dataset(dataset)
    data_split, target_split = split_dataset(dataset, cfg['num_teachers'], cfg['data_split_mode'])
    dataset = make_split_dataset(dataset, data_split, target_split)
    teacher_model_name = make_model_name(cfg['teacher_model_name'], cfg['num_teachers'])
    teacher = make_teacher(teacher_model_name, data_split, target_split)
    data_loader = {'train': [], 'test': []}
    for i in range(len(dataset)):
        data_loader_i = make_data_loader(dataset[i], 'teacher')
        data_loader['train'].append(data_loader_i['train'])
        data_loader['test'].append(data_loader_i['test'])
    metric = Metric(cfg['teacher_data_name'], {'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'])
        last_epoch = result['epoch']
        if last_epoch > 1:
            data_split = result['data_split']
            target_split = result['target_split']
            teacher = result['teacher']
            logger = result['logger']
        else:
            logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    else:
        last_epoch = 1
        logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    for epoch in range(last_epoch, cfg['teacher']['num_epochs'] + 1):
        train(teacher, data_loader['train'], metric, logger, epoch)
        test(teacher, data_loader['test'], metric, logger, epoch)
        result = {'cfg': cfg, 'epoch': epoch + 1, 'data_split': data_split, 'target_split': target_split,
                  'teacher': teacher, 'logger': logger}
        save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    return


def make_teacher(model_name, data_split, target_split):
    model_name = model_name
    teacher = [None for _ in range(len(data_split))]
    for i in range(len(data_split)):
        model_name_i = model_name[i]
        data_split_i = data_split[i]
        target_split_i = target_split[i]
        teacher[i] = Teacher(i, model_name_i, data_split_i, target_split_i)
    return teacher


def train(teacher, data_loader, metric, logger, epoch):
    logger.safe(True)
    start_time = time.time()
    num_teachers = len(teacher)
    for i in range(num_teachers):
        teacher[i].train(data_loader[i], metric, logger, epoch)
        if i % int((num_teachers * cfg['log_interval']) + 1) == 0:
            local_time = (time.time() - start_time) / (i + 1)
            epoch_finished_time = datetime.timedelta(seconds=local_time * (num_teachers - i - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['teacher']['num_epochs'] - epoch) * local_time * num_teachers))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / num_teachers),
                             'ID: {}/{}'.format(i + 1, num_teachers),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
    logger.safe(False)
    return


def test(teacher, data_loader, metric, logger, epoch):
    logger.safe(True)
    for i in range(len(teacher)):
        teacher[i].test(data_loader[i], metric, logger)
    info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
    logger.append(info, 'test', mean=False)
    print(logger.write('test', metric.metric_name['test']))
    logger.safe(False)
    return


if __name__ == "__main__":
    main()
