import os
import math
import functools
import itertools
import ataxx
import torch
import torch.nn.functional as F
import numpy as np
import random

def plane_size():
  return (1, 7, 7)

def side_size():
  return (6, 7, 7)

def side_numel():
  return functools.reduce(lambda a, b: a*b, side_size())

def state_size():
  return (12, 7, 7)

def state_numel():
  return functools.reduce(lambda a, b: a*b, state_size())

def half_input_numel():
  return state_numel()


def cp_conversion(x, alpha=0.00167):
  return (x * alpha).sigmoid()


def half_input(us, them):
  return torch.cat([us, them], dim=1).flatten(start_dim=1)


def side_to_tensor(bd, color):
  limit = lambda x: max(0, min(6, x))

  us = torch.zeros(plane_size(), dtype=torch.bool)
  them = torch.zeros(plane_size(), dtype=torch.bool)

  them_right = torch.zeros(plane_size(), dtype=torch.bool)
  them_left = torch.zeros(plane_size(), dtype=torch.bool)
  them_above = torch.zeros(plane_size(), dtype=torch.bool)
  them_below = torch.zeros(plane_size(), dtype=torch.bool)

  for i, j in itertools.product(range(7), range(7)):
    us[:, j, i] = bd.get(i, j) == color
    them[:, j, i] = bd.get(i, j) == (not color)
    them_right[:, j, i] = bd.get(limit(i + 1), j) == (not color)
    them_left[:, j, i] = bd.get(limit(i - 1), j) == (not color)
    them_above[:, j, i] = bd.get(i, limit(j + 1)) == (not color)
    them_below[:, j, i] = bd.get(i, limit(j - 1)) == (not color)

  them_singles = F.max_pool2d(them.float(), kernel_size=(3, 3), stride=1, padding=1).gt(0.5)

  tensor = torch.cat([torch.logical_and(them_right, us),\
                      torch.logical_and(them_left, us),\
                      torch.logical_and(them_above, us),\
                      torch.logical_and(them_below, us),\
                      torch.logical_and(them_singles, us),\
                      us], dim=0)

  return tensor


def to_tensors(bd):
  white = side_to_tensor(bd, color=ataxx.WHITE)
  black = side_to_tensor(bd, color=ataxx.BLACK)
  return white, black

