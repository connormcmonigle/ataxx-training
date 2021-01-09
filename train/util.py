import os
import math
import functools
import itertools
import ataxx
import torch
import torch.nn.functional as F
import numpy as np
import random

def side_size():
  return (1, 7, 7)

def side_numel():
  return functools.reduce(lambda a, b: a*b, side_size())

def state_size():
  return (2, 7, 7)

def state_numel():
  return functools.reduce(lambda a, b: a*b, state_size())

def half_input_numel():
  return state_numel()


def cp_conversion(x, alpha=0.00167):
  return (x * alpha).sigmoid()


def half_input(us, them):
  return torch.cat([us, them], dim=1).flatten(start_dim=1)


def side_to_tensor(bd, color):
  tensor = torch.zeros(side_size(), dtype=torch.bool)
  
  for i, j in itertools.product(range(7), range(7)):
    tensor[:, i, j] = bd.get(i, j) == color

  return tensor


def to_tensors(bd):
  white = side_to_tensor(bd, color=ataxx.WHITE)
  black = side_to_tensor(bd, color=ataxx.BLACK)
  return white, black

