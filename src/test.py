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

# if __name__ == "__main__":
#     cfg['seed'] = 0
#     torch.manual_seed(cfg['seed'])
#     torch.cuda.manual_seed(cfg['seed'])
#     # cfg['control']['data_name'] = 'MLP-r-100-10-32-1-1-sigmoid-1-1-0.5'
#     cfg['control']['data_name'] = 'MLP-c-100-10-32-1-1-sigmoid-10-1-0.5'
#     cfg['control']['model_name'] = 'conv'
#     process_control()
#     dataset = fetch_dataset(cfg['data_name'])
#     process_dataset(dataset)
#     data_loader = make_data_loader(dataset, cfg['model_name'])
#     print(len(dataset['train']), len(dataset['test']))
#     print(len(data_loader['train']), len(data_loader['test']))
#     for i, input in enumerate(data_loader['train']):
#         input = collate(input)
#         print(i, input['data'].shape, input['target'].shape)
#         break
#     for i, input in enumerate(data_loader['test']):
#         input = collate(input)
#         print(i, input['data'].shape, input['target'].shape)
#         break

# def make_si(x, q=0.5):
#     si = (torch.linalg.norm(x, 1).pow(1) / x.size(0)).pow(1) / \
#          (torch.linalg.norm(x, q).pow(q) / x.size(0)).pow(1 / q)
#     si = (torch.linalg.norm(x, 1).pow(1) / x.size(0)).pow(1) / \
#          (torch.linalg.norm(x, q).pow(q) / x.size(0)).pow(1 / q)
#     return si
#
#
# if __name__ == "__main__":
#     a = torch.randn(100)
#     b = a.clone()
#     b[b.abs() < 0.3] = 1e-5
#     c = torch.randn(1000)
#     d = c.clone()
#     d[d.abs() < 0.3] = 1e-5
#     e = a * 10
#     f = b * 10
#     g = torch.cat([a, a], dim=0)
#     h = torch.cat([b, b], dim=0)
#     si_a = torch.linalg.norm(a, 1) / torch.linalg.norm(a, 0.5)
#     si_b = torch.linalg.norm(b, 1) / torch.linalg.norm(b, 0.5)
#     si_c = torch.linalg.norm(a, 1) / torch.linalg.norm(c, 0.5)
#     si_d = torch.linalg.norm(b, 1) / torch.linalg.norm(d, 0.5)
#
#     normalized_si_a = (torch.linalg.norm(a, 1).pow(1) / a.size(0)).pow(1) / \
#                       (torch.linalg.norm(a, 0.5).pow(0.5) / a.size(0)).pow(1 / 0.5)
#     normalized_si_b = (torch.linalg.norm(b, 1).pow(1) / b.size(0)).pow(1) / \
#                       (torch.linalg.norm(b, 0.5).pow(0.5) / b.size(0)).pow(1 / 0.5)
#     normalized_si_c = (torch.linalg.norm(c, 1).pow(1) / c.size(0)).pow(1) / \
#                       (torch.linalg.norm(c, 0.5).pow(0.5) / c.size(0)).pow(1 / 0.5)
#     normalized_si_d = (torch.linalg.norm(d, 1).pow(1) / d.size(0)).pow(1) / \
#                       (torch.linalg.norm(d, 0.5).pow(0.5) / d.size(0)).pow(1 / 0.5)
#     normalized_si_e = (torch.linalg.norm(e, 1).pow(1) / e.size(0)).pow(1) / \
#                       (torch.linalg.norm(e, 0.5).pow(0.5) / e.size(0)).pow(1 / 0.5)
#     normalized_si_f = (torch.linalg.norm(f, 1).pow(1) / f.size(0)).pow(1) / \
#                       (torch.linalg.norm(f, 0.5).pow(0.5) / f.size(0)).pow(1 / 0.5)
#     normalized_si_g = (torch.linalg.norm(g, 1).pow(1) / g.size(0)).pow(1) / \
#                       (torch.linalg.norm(g, 0.5).pow(0.5) / g.size(0)).pow(1 / 0.5)
#     normalized_si_h = (torch.linalg.norm(h, 1).pow(1) / h.size(0)).pow(1) / \
#                       (torch.linalg.norm(h, 0.5).pow(0.5) / h.size(0)).pow(1 / 0.5)
#     print('si_a', si_a)
#     print('si_b', si_b)
#     print('si_c', si_c)
#     print('si_d', si_d)
#
#     print('normalized_si_a', normalized_si_a)
#     print('normalized_si_b', normalized_si_b)
#     print('normalized_si_c', normalized_si_c)
#     print('normalized_si_d', normalized_si_d)
#     print('normalized_si_e', normalized_si_e)
#     print('normalized_si_f', normalized_si_f)
#     print('normalized_si_g', normalized_si_g)
#     print('normalized_si_h', normalized_si_h)
#
#     rb = torch.rand(100).sort(descending=True)[0]
#     alpha = torch.diff(rb).abs().min() / 2 - 1e-7
#     er = torch.tensor([-1, 1]).repeat(rb.size(0) // 2) * torch.ones(rb.size(0)) * alpha
#     rb_er = rb + er
#     normalized_si_rb = make_si(rb)
#     normalized_si_rb_er = make_si(rb_er)
#     print(alpha)
#     print('normalized_si_rb', normalized_si_rb)
#     print('normalized_si_rb_er', normalized_si_rb_er)
#     print(normalized_si_rb > normalized_si_rb_er)
#
#     rt = torch.rand(100)
#     alpha = 0.3
#     rt_alpha = rt + alpha
#     normalized_si_rt = make_si(rt)
#     normalized_si_rt_alpha = make_si(rt_alpha)
#     print('normalized_si_rt', normalized_si_rt)
#     print('normalized_si_rt_alpha', normalized_si_rt_alpha)
#     print(normalized_si_rt > normalized_si_rt_alpha)
#
#     a = torch.randn(100)
#     b = a.clone()
#     c = a.clone()
#     d = a.clone()
#     b[b.abs() < 0.3] = 1e-5
#     c[c.abs() < 0.4] = 1e-5
#     d[d.abs() < 0.5] = 1e-5
#     si_a = make_si(a)
#     si_b = make_si(b)
#     si_c = make_si(c)
#     si_d = make_si(d)
#     a_b = torch.cat([a, b], dim=0)
#     c_d = torch.cat([c, d], dim=0)
#     si_a_b = make_si(a_b)
#     si_c_d = make_si(c_d)
#     print('si_a_b', si_a_b)
#     print('si_c_d', si_c_d)
#
#     # D1
#     a = torch.tensor([0, 1, 3, 5]).float()
#     b = torch.tensor([0, 2, 3, 4]).float()
#     si_a = make_si(a)
#     si_b = make_si(b)
#     print('D1', si_a > si_b)
#
#     # D2
#     a = torch.tensor([0, 1, 3, 5]).float()
#     b = torch.tensor([0, 2, 6, 10]).float()
#     si_a = make_si(a)
#     si_b = make_si(b)
#     print('D2', si_a == si_b)
#
#     # D3
#     a = torch.tensor([1, 3, 5]).float()
#     b = torch.tensor([1.5, 3.5, 5.5]).float()
#     q = 0.2
#     si_a = make_si(a, q)
#     si_b = make_si(b, q)
#     print(si_a, si_b)
#     print('D3', si_a > si_b)
#
#     # D4
#     a = torch.tensor([0, 1, 3, 5]).float()
#     b = torch.tensor([0, 0, 1, 1, 3, 3, 5, 5]).float()
#     si_a = make_si(a, q)
#     si_b = make_si(b, q)
#     print(si_a, si_b)
#     print('D4', si_a == si_b)
#
#     # P1
#     a = torch.tensor([0, 1, 3, 5]).float()
#     b = torch.tensor([0, 1, 3, 20]).float()
#     si_a = make_si(a, q)
#     si_b = make_si(b, q)
#     print('P1', si_a < si_b)
#
#     # P2
#     a = torch.tensor([0, 1, 3, 5]).float()
#     b = torch.tensor([0, 0, 0, 1, 3, 5]).float()
#     si_a = make_si(a, q)
#     si_b = make_si(b, q)
#     print('P2', si_a < si_b)

# def make_si(x, q=0.5):
#     si = (torch.linalg.norm(x, 1).pow(1) / x.size(0)).pow(1) / \
#          (torch.linalg.norm(x, q).pow(q) / x.size(0)).pow(1 / q)
#     return si
#
#
# def make_si_corrected(x, q=0.5):
#     num_valid = (x != 0).float().sum()
#     si = (torch.linalg.norm(x, 1).pow(1) / num_valid).pow(1) / \
#          (torch.linalg.norm(x, q).pow(q) / num_valid).pow(1 / q)
#     return si
#
#
# if __name__ == "__main__":
#     a = torch.randn(100)
#     b = a.clone()
#     b[:50] = 0
#     c = b[50:].clone()
#     si_a = make_si(a)
#     si_b = make_si(b)
#     si_c = make_si(c)
#     print('si_a', si_a)
#     print('si_b', si_b)
#     print('si_c', si_c)
#     corrected_si_a = make_si_corrected(a)
#     corrected_si_b = make_si_corrected(b)
#     corrected_si_c = make_si_corrected(c)
#     print('corrected_si_a', corrected_si_a)
#     print('corrected_si_b', corrected_si_b)
#     print('corrected_si_c', corrected_si_c)


# if __name__ == "__main__":
#     from utils import save_model_state_dict, save_optimizer_state_dict, save_scheduler_state_dict
#     process_control()
#     dataset = fetch_dataset(cfg['data_name'])
#     process_dataset(dataset)
#     model = models.linear().to('cuda')
#     optimizer = make_optimizer(model.parameters(), cfg['model_name'])
#     scheduler = make_scheduler(optimizer, cfg['model_name'])
#     model_state_dict = save_model_state_dict(model.state_dict())
#     optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
#     scheduler_state_dict = save_scheduler_state_dict(scheduler.state_dict())
#     print(model_state_dict)
#     print(optimizer_state_dict)
#     print(scheduler_state_dict)

# if __name__ == "__main__":
#     dim = -1
#     d = 100
#     x = torch.randn(100)
#     p = 0.5
#     q = 1
#     # si = (torch.linalg.norm(x, q, dim=dim).pow(q) / d).pow(1 / q) / \
#     #      (torch.linalg.norm(x, p, dim=dim).pow(p) / d).pow(1 / p)
#     si = (torch.linalg.norm(x, p, dim=dim).pow(p) / d).pow(1 / p) / \
#          (torch.linalg.norm(x, q, dim=dim).pow(q) / d).pow(1 / q)
#     print(si)

if __name__ == "__main__":
    cfg['target_size'] = 10
    model = models.resnet50()
    print(model)
