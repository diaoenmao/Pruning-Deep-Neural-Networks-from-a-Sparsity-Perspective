import copy
import datetime
import time
import sys
import torch
import models
from config import cfg
from utils import to_device, collate, make_optimizer, make_scheduler


class Teacher:
    def __init__(self, teacher_id, model_name, data_split, target_split):
        self.teacher_id = teacher_id
        self.model_name = model_name
        self.data_split = data_split
        self.target_split = target_split
        self.model_state_dict = None
        self.optimizer_state_dict = None
        self.scheduler_state_dict = None

    def train(self, data_loader, metric, logger, epoch):
        model = eval('models.{}(cfg["teacher_data_shape"], '
                       'len(self.target_split["train"])).to(cfg["device"])'.format(self.model_name))
        optimizer = make_optimizer(model, 'teacher')
        scheduler = make_scheduler(optimizer, 'teacher')
        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict)
            optimizer.load_state_dict(self.optimizer_state_dict)
            scheduler.load_state_dict(self.scheduler_state_dict)
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
            optimizer.step()
            evaluation = metric.evaluate(metric.metric_name['train'], input, output)
            logger.append(evaluation, 'train', n=input_size)
            if i % int((len(data_loader) * cfg['log_interval']) + 1) == 0:
                _time = (time.time() - start_time) / (i + 1)
                lr = optimizer.param_groups[0]['lr']
                epoch_finished_time = datetime.timedelta(seconds=round(_time * (len(data_loader) - i - 1)))
                info = {'info': ['Model: {}'.format(cfg['model_tag']),
                                 'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / len(data_loader)),
                                 'Learning rate: {:.6f}'.format(lr),
                                 'Epoch Finished Time: {}'.format(epoch_finished_time)]}
                logger.append(info, 'train', mean=False)
                print(logger.write('train', metric.metric_name['train']), end='\r', flush=True)
        scheduler.step()
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        self.optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
        self.scheduler_state_dict = copy.deepcopy(scheduler.state_dict())
        sys.stdout.write('\x1b[2K')
        return

    def test(self, data_loader, metric, logger):
        model = eval('models.{}(cfg["teacher_data_shape"], '
                       'len(self.target_split["train"])).to(cfg["device"])'.format(self.model_name))
        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict)
        model = model.to(cfg['device'])
        with torch.no_grad():
            model.train(False)
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                output = model(input)
                evaluation = metric.evaluate(metric.metric_name['train'], input, output)
                logger.append(evaluation, 'test', n=input_size)
        return

# class Client:
#     def __init__(self, client_id, model, data_split, target_split):
#         self.client_id = client_id
#         self.data_split = data_split
#         self.target_split = target_split
#         self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
#         optimizer = make_optimizer(model, 'client')
#         self.optimizer_state_dict = optimizer.state_dict()
#         self.active = False
#         if cfg['loss_mode'] == 'feature-gen':
#             self.generator_state_dict = None
#             self.label_count = None
#
#     def train(self, dataset, lr, metric, logger):
#         model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
#         model.load_state_dict(self.model_state_dict, strict=False)
#         self.optimizer_state_dict['param_groups'][0]['lr'] = lr
#         optimizer = make_optimizer(model, 'client')
#         optimizer.load_state_dict(self.optimizer_state_dict)
#         model.train(True)
#         if cfg['loss_mode'] == 'feature-gen':
#             generator = models.feature_generator().to(cfg['device'])
#             generator.load_state_dict(self.generator_state_dict)
#             generator.train(False)
#             data_loader = make_data_loader({'train': dataset}, 'client')['train']
#             label_count = self.label_count.to(cfg['device'])
#             for epoch in range(1, cfg['client']['num_epochs'] + 1):
#                 for i, input in enumerate(data_loader):
#                     input = collate(input)
#                     input_size = input['data'].size(0)
#                     input = to_device(input, cfg['device'])
#                     optimizer.zero_grad()
#                     output = model(input)
#                     if label_count.sum() > 0:
#                         with torch.no_grad():
#                             z = torch.randn(input_size, cfg['feature_generator']['input_size'] // 2,
#                                             device=cfg['device'])
#                             y = torch.multinomial(label_count, input_size, replacement=True)
#                             input_ = {'z': z, 'y': y}
#                             feature = generator(input_)['x']
#                         y_hat = model.classify(feature)
#                         gen_loss = F.cross_entropy(y_hat, input_['y'])
#                         output['loss'] = output['loss'] + 1 * gen_loss
#                     output['loss'].backward()
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
#                     optimizer.step()
#                     evaluation = metric.evaluate(metric.metric_name['train'], input, output)
#                     logger.append(evaluation, 'train', n=input_size)
#         else:
#             if cfg['loss_mode'] in ['weight']:
#                 self.weight = models.make_weight(torch.tensor(dataset.target))
#             data_loader = make_data_loader({'train': dataset}, 'client')['train']
#             for epoch in range(1, cfg['client']['num_epochs'] + 1):
#                 for i, input in enumerate(data_loader):
#                     input = collate(input)
#                     input_size = input['data'].size(0)
#                     if cfg['loss_mode'] in ['weight']:
#                         input['target_split'] = torch.tensor(list(self.target_split.keys()))
#                         input['weight'] = self.weight
#                     input = to_device(input, cfg['device'])
#                     optimizer.zero_grad()
#                     output = model(input)
#                     output['loss'].backward()
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
#                     optimizer.step()
#                     evaluation = metric.evaluate(metric.metric_name['train'], input, output)
#                     logger.append(evaluation, 'train', n=input_size)
#         self.optimizer_state_dict = optimizer.state_dict()
#         self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
#         return
