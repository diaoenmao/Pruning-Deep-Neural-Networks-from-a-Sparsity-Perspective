from config import cfg
from data import fetch_dataset, make_data_loader
from utils import collate, process_dataset, save_img, process_control, resume, to_device
import torch
import models

# if __name__ == "__main__":
#     cfg['seed'] = 0
#     torch.manual_seed(cfg['seed'])
#     torch.cuda.manual_seed(cfg['seed'])
#     cfg['data_name'] = 'CIFAR100'
#     cfg['model_name'] = 'conv'
#     process_control()
#     dataset = fetch_dataset(cfg['data_name'])
#     process_dataset(dataset)
#     data_loader = make_data_loader(dataset, 'teacher')
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