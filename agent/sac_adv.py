# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for soft actor-critic with adversarial training.

This file implements a soft actor-critic (SAC) agent with adversarial training.
There are two actors in this agent, 'ctrl' and 'dstb', and a central critic.
"""

from typing import Optional
import numpy as np
import torch

from agent.base_ma_sac import BaseMASAC
from agent.replay_memory import Batch


class SACAdv(BaseMASAC):

  def build_network(self, verbose: bool = True):
    super().build_network(verbose)
    self.ctrl = self.actors['ctrl']  # alias
    self.dstb = self.actors['dstb']  # alias
    self.dstb_use_ctrl = self.dstb.obs_other_list is not None

    self.action_dim_ctrl = self.ctrl.action_dim
    self.action_dim_dstb = self.dstb.action_dim

  # Overrides BaseMASAC.update() since we sometimes only want to update a
  # specific actor.
  def update(
      self, batch: Batch, timer: int, update_ctrl: bool, update_dstb: bool
  ) -> dict:
    loss_c_dict = self.update_critic(batch)
    loss_a_dict = {}
    flag_dict = {}
    for key, actor in self.actors.items():
      if actor.eval:
        continue
      if key == 'ctrl' and not update_ctrl:
        continue
      if key == 'dstb' and not update_dstb:
        continue
      if timer % actor.update_period == 0:
        loss_a_dict[key] = actor.update(batch, self.critics[actor.pg_target])
        flag_dict[key] = True
      else:
        loss_a_dict[key] = 0., 0., 0.  # ! dummy values
        flag_dict[key] = False
    self.update_critic_target(timer)
    return {'critic': loss_c_dict, 'actor': loss_a_dict, 'flag': flag_dict}

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

  # Overrides BaseMASAC.remove() since we sometimes only want to update a
  # specific actor.
  def remove(self, step: int, model_folder: str, rm_ctrl: bool, rm_dstb: bool):
    for critic in self.critics.values():
      if critic.eval:
        continue
      critic.remove(step, model_folder)
    if rm_ctrl:
      self.ctrl.remove(step, model_folder)
    if rm_dstb:
      self.dstb.remove(step, model_folder)
