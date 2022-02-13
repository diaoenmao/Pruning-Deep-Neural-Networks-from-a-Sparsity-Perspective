import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import models
from config import cfg, process_args
from data import fetch_dataset, make_data_loader, make_split_dataset
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, resume, collate, make_model_name
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
    result = resume(cfg['model_tag'], load_tag='best')
    last_epoch = result['epoch']
    data_split = result['data_split']
    target_split = result['target_split']
    teacher = result['teacher']
    dataset = make_split_dataset(dataset, data_split, target_split)
    data_loader = {'train': [], 'test': []}
    for i in range(len(dataset)):
        data_loader_i = make_data_loader(dataset[i], 'teacher')
        data_loader['train'].append(data_loader_i['train'])
        data_loader['test'].append(data_loader_i['test'])
    metric = Metric(cfg['teacher_data_name'], {'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    test_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    test_each_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    test_each(teacher, data_loader['test'], metric, test_each_logger, last_epoch)
    test(teacher, data_loader['test'], metric, test_logger, last_epoch)
    result = resume(cfg['model_tag'], load_tag='checkpoint')
    train_logger = result['logger'] if 'logger' in result else None
    result = {'cfg': cfg, 'epoch': last_epoch,
              'logger': {'train': train_logger, 'test': test_logger, 'test_each': test_each_logger}}
    save(result, './output/result/{}.pt'.format(cfg['model_tag']))
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


def test_each(teacher, data_loader, metric, each_logger, epoch):
    for i in range(len(teacher)):
        each_logger.safe(True)
        teacher[i].test(data_loader[i], metric, each_logger)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.),
                         'ID: {}/{}'.format(i + 1, len(teacher))]}
        each_logger.append(info, 'test', mean=False)
        print(each_logger.write('test', metric.metric_name['test']))
        each_logger.safe(False)
        each_logger.reset()
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
