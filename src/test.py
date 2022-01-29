from config import cfg
from data import fetch_dataset, make_data_loader, make_batchnorm_stats
from utils import collate, process_dataset, save_img, process_control, resume, to_device
import torch
import models

# from data import BaseDataset
#
# if __name__ == "__main__":
#     data = BaseDataset(data=torch.randn(100), target=torch.randn(100))
#     x = next(iter(data))
#     print(len(data), x)


# if __name__ == "__main__":
#     x = torch.randn(2, 3, 4, 4)
#     loss = models.total_variation_loss(x)
#     loss = torch.linalg.norm(x, dim=(-2, -1)).mean()
#     print(loss)
#     exit()
