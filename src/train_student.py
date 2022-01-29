import argparse
import datetime
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from data import fetch_dataset, make_data_loader, make_dataset_normal, make_batchnorm_stats, BaseDataset
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import make_logger

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
        teacher_model_tag_list = [str(seeds[i]), '_'.join(cfg['control_name'].split('_')[:2])]
        cfg['teacher_model_tag'] = '_'.join([x for x in teacher_model_tag_list if x])
        generator_model_tag_list = [str(seeds[i]), '_'.join(cfg['control_name'].split('_')[:3])]
        cfg['generator_model_tag'] = '_'.join([x for x in generator_model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)
    data_loader = make_data_loader(dataset, 'student')
    teacher = eval('models.{}().to(cfg["device"])'.format(cfg['teacher_model_name']))
    result = resume(cfg['teacher_model_tag'], load_tag='best')
    teacher.load_state_dict(result['model_state_dict'])
    generator = eval('models.{}().to(cfg["device"])'.format(cfg['generator_model_name']))
    model = eval('models.{}().to(cfg["device"])'.format(cfg['student_model_name']))
    # Create hooks for feature statistics catching
    moment_loss_list = models.make_moment_loss(teacher)
    generator_optimizer = make_optimizer(generator, 'generator')
    optimizer = make_optimizer(model, 'student')
    generator_scheduler = make_scheduler(generator_optimizer, 'generator')
    scheduler = make_scheduler(optimizer, 'student')
    metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    if cfg['resume_mode'] == 1:
        generator_result = resume(cfg['generator_model_tag'])
        result = resume(cfg['model_tag'])
        last_epoch = result['epoch']
        if last_epoch > 1:
            generator.load_state_dict(generator_result['model_state_dict'])
            model.load_state_dict(result['model_state_dict'])
            generator_optimizer.load_state_dict(generator_result['optimizer_state_dict'])
            optimizer.load_state_dict(result['optimizer_state_dict'])
            generator_scheduler.load_state_dict(generator_result['scheduler_state_dict'])
            scheduler.load_state_dict(result['scheduler_state_dict'])
            logger = result['logger']
        else:
            logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    else:
        last_epoch = 1
        logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    if cfg['world_size'] > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(cfg['world_size'])))
    for epoch in range(last_epoch, cfg['student']['num_epochs'] + 1):
        train(teacher, generator, model, moment_loss_list, generator_optimizer, optimizer, metric, logger, epoch)
        test(data_loader['test'], model, metric, logger, epoch)
        scheduler.step()
        generator_model_state_dict = generator.module.state_dict() if cfg['world_size'] > 1 else generator.state_dict()
        model_state_dict = model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict()
        result = {'cfg': cfg, 'epoch': epoch + 1, 'model_state_dict': generator_model_state_dict,
                  'optimizer_state_dict': generator_optimizer.state_dict(),
                  'scheduler_state_dict': generator_scheduler.state_dict(), 'logger': logger}
        save(result, './output/model/{}_checkpoint.pt'.format(cfg['generator_model_tag']))
        result = {'cfg': cfg, 'epoch': epoch + 1, 'model_state_dict': model_state_dict,
                  'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                  'logger': logger}
        save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['generator_model_tag']),
                        './output/model/{}_best.pt'.format(cfg['generator_model_tag']))
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    return


def train(teacher, generator, model, moment_loss_list, optimizer_generator, optimizer, metric, logger, epoch):
    logger.safe(True)
    teacher.train(False)
    generator.train(True)
    model.train(True)
    start_time = time.time()
    batch_size = cfg['student']['batch_size']['train']
    num_iter = cfg['student']['num_iter']
    for i in range(num_iter):
        input_size = batch_size
        z_1 = torch.randn(batch_size // 2, cfg['generator']['latent_size']).to(cfg['device'])
        z_2 = torch.randn(batch_size // 2, cfg['generator']['latent_size']).to(cfg['device'])
        z = torch.cat([z_1, z_2], dim=0)
        input = {'z': z}
        input = to_device(input, cfg['device'])
        optimizer_generator.zero_grad()
        optimizer.zero_grad()
        output = generator(input)
        generated = output['x']
        x_1, x_2 = torch.chunk(generated, 2, dim=0)
        # mode seeking loss
        mode_seeking_loss = torch.mean(torch.abs(x_1 - x_2)) / torch.mean(torch.abs(z_1 - z_2))
        mode_seeking_loss = 1 / (mode_seeking_loss + 1e-5)
        input = {'data': generated}
        teacher_output = teacher(input)
        teacher_pred = torch.softmax(teacher_output['target'], dim=-1)
        # ce loss
        threshold = 0
        max_p, teacher_target = torch.max(teacher_pred, dim=-1)
        mask = max_p.ge(threshold)
        if torch.any(mask):
            ce_loss = torch.nn.functional.cross_entropy(teacher_output['target'][mask], teacher_target[mask])
        else:
            ce_loss = 0
        # momentum_loss
        momentum_loss = sum([m.moment_loss for m in moment_loss_list])
        # information_entropy loss
        prob = torch.softmax(teacher_pred, dim=-1)
        py = prob.mean(dim=0)
        information_entropy_loss = (py * torch.log10(py)).sum()
        # inception loss
        inception_loss = -torch.nn.functional.kl_div(py.log().view(1, -1).expand_as(prob), prob,
                                                     reduction='batchmean').exp()
        input = {'data': generated.detach(), 'target': teacher_output['target'].detach()}
        output = model(input)
        kld_loss = output['loss']
        # output['loss'] = kld_loss + 1 * ce_loss + 1 * information_entropy_loss + \
        #                  1 * mode_seeking_loss + 1 * momentum_loss
        # output['loss'] = kld_loss + 0.05 * ce_loss + 5 * information_entropy_loss + 1 * momentum_loss
        output['loss'] = kld_loss + 0.05 * ce_loss + 5 * information_entropy_loss
        output['loss'].backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer_generator.step()
        optimizer.step()
        evaluation = metric.evaluate(metric.metric_name['train'], input, output)
        logger.append(evaluation, 'train', n=input_size)
        if i % int((num_iter * cfg['log_interval']) + 1) == 0:
            _time = (time.time() - start_time) / (i + 1)
            lr = optimizer.param_groups[0]['lr']
            epoch_finished_time = datetime.timedelta(seconds=round(_time * (num_iter - i - 1)))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['student']['num_epochs'] - epoch) * _time * num_iter))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / num_iter),
                             'Learning rate: {:.6f}'.format(lr), 'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
    logger.safe(False)
    return


def test(data_loader, model, metric, logger, epoch):
    logger.safe(True)
    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    logger.safe(False)
    return


if __name__ == "__main__":
    main()
