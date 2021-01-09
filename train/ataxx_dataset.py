import os
import math
import bitstring
import struct
import random
import torch
import ataxx
import torch.nn.functional as F
import numpy as np
import util

def num_samples(file_path):
  with open(file_path) as file:
    return sum(1 for _ in file)


def worker_init_fn(worker_id):
  worker_info = torch.utils.data.get_worker_info()
  dataset = worker_info.dataset
  per_worker = dataset.cardinality() // worker_info.num_workers
  start = worker_id * per_worker
  dataset.set_range(start, start + per_worker)


class AtaxxData(torch.utils.data.IterableDataset):
  def __init__(self, file_path, config):
    super(AtaxxData, self).__init__()
    self.config = config
    self.batch_size = config.batch_size

    self.file_path = file_path
    self.total_samples = num_samples(file_path)
    self.shuffle_buffer = [None] * config.shuffle_buffer_size

    self.start_idx = 0
    self.end_idx = self.total_samples

  def cardinality(self):
    return self.total_samples

  def set_range(self, start_idx, end_idx):
    self.start_idx = start_idx
    self.end_idx = end_idx

  def get_shuffled(self, elem):
    shuffle_buffer_idx = random.randrange(len(self.shuffle_buffer))
    result = self.shuffle_buffer[shuffle_buffer_idx]
    self.shuffle_buffer[shuffle_buffer_idx] = elem
    return result

  def from_line(self, line):
    val, fen = line.split(sep=None, maxsplit=1)
    bd = ataxx.Board(fen)
    white, black = util.to_tensors(bd)
    return torch.tensor([bd.turn]).float(), white.float(), black.float(), torch.tensor([int(val)]).float()

  def __iter__(self):
    with open(self.file_path) as file:
      for idx, line in enumerate(file):
        if idx < self.start_idx or idx >= self.end_idx:
          continue
        val = self.get_shuffled(self.from_line(line))
        if val != None:
          yield val
      for val in self.shuffle_buffer:
        if val != None:
          yield val

