from os import path
import torch
import ataxx

import config as C
import util
import model

config = C.Config('config.yaml')

M = model.NNUE().to(config.device)

if (path.exists(config.model_save_path)):
  print('Loading model ... ')
  M.load_state_dict(torch.load(config.model_save_path, map_location=config.device))

num_parameters = sum(map(lambda x: torch.numel(x), M.parameters()))

print(num_parameters)

M.cpu()

while True:
  bd = ataxx.Board(input("fen: "))
  w, b = util.to_tensors(bd)
  white, black = util.to_tensors(bd)
  val = M(torch.tensor([bd.turn]).float(), white.unsqueeze(0).float(), black.unsqueeze(0).float())
  print(val)
