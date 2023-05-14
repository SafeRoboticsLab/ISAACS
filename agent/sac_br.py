# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for soft actor-critic being a best response to a fixed target.

This file implements a soft actor-critic (SAC) agent that is a best response
to a fixed target. Currently it only supports targets as neural-network-based
policies (actors).
# TODO: Supports targets as ILQR, MPC, etc.
"""

from typing import Optional
import torch
import numpy as np

from agent.sac import SAC


class SACBestResponse(SAC):

  def build_network(self, verbose: bool = True):
    super().build_network(verbose)
    self.ctrl = self.actors['ctrl']  # alias
    self.dstb = self.actors['dstb']  # alias
    self.actor = self.actors['dstb']  # alias, the thing we want to optimize
    self.dstb_use_ctrl = self.dstb.obs_other_list is not None

  # Overrides SAC.value() since we need to concatenate actions from 'ctrl' and
  # 'dstb' to feed into (central) critic.
  def value(
      self, obs: np.ndarray, append: Optional[np.ndarray] = None
  ) -> np.ndarray:
    with torch.no_grad():
      action_ctrl = self.ctrl.net(obs, append=append)
      if self.dstb_use_ctrl:
        state_dstb = np.concatenate((obs, action_ctrl), axis=-1)
      else:
        state_dstb = obs
      action_dstb = self.dstb.net(state_dstb, append=append)
      action = np.concatenate((action_ctrl, action_dstb), axis=-1)
    action = torch.FloatTensor(action).to(self.device)

    q_pi_1, q_pi_2 = self.critic.net(obs, action, append=append)
    value = (q_pi_1+q_pi_2) / 2
    assert isinstance(value, np.ndarray)
    return value
