import torch
import ataxx_dataset
import config as C

config = C.Config('config.yaml')

d = ataxx_dataset.AtaxxData(config.train_ataxx_data_path, config)
#d = torch.utils.data.DataLoader(d, num_workers=config.num_workers, worker_init_fn=nnue_bin_dataset.worker_init_fn)

for i in d:
  print(i)
