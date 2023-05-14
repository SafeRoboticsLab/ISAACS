# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for basic soft actor-critic.
"""

from typing import Optional
import torch
import numpy as np

from agent.base_ma_sac import BaseMASAC


class SAC(BaseMASAC):

  @property
  def actor_type(self):
    return self.actor.actor_type

  def build_network(self, verbose: bool = True):
    super().build_network(verbose)
    self.actor = self.actors['ctrl']  # alias

  def value(
      self, obs: np.ndarray, append: Optional[np.ndarray] = None
  ) -> np.ndarray:
    with torch.no_grad():
      action = self.actor.net(obs, append=append)
    return self.critic.value(obs, action, append=append)
