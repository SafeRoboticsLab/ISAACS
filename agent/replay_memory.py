# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Classes for building blocks for actors and critics.

modified from: https://github.com/SafeRoboticsLab/SimLabReal/blob/main/agent/replay_memory.py
"""

from typing import List
import numpy as np
import torch as th
from collections import deque, namedtuple

# `Transition` is a named tuple representing a single transition in our
# RL environment. All the other information is stored in the `info`, e.g.,
# `g_x`, `l_x`, `binary_cost`, `append`, and `latent`, etc. Note that we also
# require all the values to be np.ndarray or float.
Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'done', 'info'])


class Batch(object):

  def __init__(self, transitions: List[Transition], device: th.device):
    self.device = device
    batch = Transition(*zip(*transitions))

    # Reward and Done.
    self.reward = th.FloatTensor(batch.r).to(device)
    self.non_final_mask = th.BoolTensor(
        np.logical_not(np.asarray(batch.done))
    ).to(device)

    # State.
    self.non_final_state_nxt = th.cat(batch.s_)[self.non_final_mask].to(device)
    self.state = th.cat(batch.s).to(device)

    # Action.
    self.action = {}
    for key in batch.a[0].keys():
      self.action[key] = th.cat([a[key] for a in batch.a]).to(device)

    # Info.
    self.info = {}
    for key, value in batch.info[0].items():
      if isinstance(value, np.ndarray) or isinstance(value, float):
        self.info[key] = th.FloatTensor(
            np.asarray([info[key] for info in batch.info])
        ).to(device)
    if 'append' in self.info:
      self.info['non_final_append_nxt'] = (
          batch.info['append_nxt'][self.non_final_mask]
      )


class ReplayMemory(object):

  def __init__(self, capacity, seed):
    self.reset(capacity)
    self.capacity = capacity
    self.seed = seed
    self.rng = np.random.default_rng(seed=self.seed)

  def reset(self, capacity):
    if capacity is None:
      capacity = self.capacity
    self.memory = deque(maxlen=capacity)

  def update(self, transition):
    self.memory.appendleft(transition)  # pop from right if full

  def sample(self, batch_size):
    length = len(self.memory)
    indices = self.rng.integers(low=0, high=length, size=(batch_size,))
    return [self.memory[i] for i in indices]

  def sample_recent(self, batch_size, recent_size):
    recent_size = min(len(self.memory), recent_size)
    indices = self.rng.integers(low=0, high=recent_size, size=(batch_size,))
    return [self.memory[i] for i in indices]

  def __len__(self):
    return len(self.memory)
