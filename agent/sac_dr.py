# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for soft actor-critic with domain randomization.

This file implements a soft actor-critic (SAC) agent with domain randomization.
It currently takes in `dstb_range` as a nested list, where each element is a
list of two floats, representing the lower and upper bound of the disturbance
at this axis. For example, `dstb_range` = [[-0.1, 0.1], [-0.3, 0.3]] means
that the disturbance is a 2D vector with the first element in [-0.1, 0.1] and
the second element in [-0.3, 0.3].
"""

from typing import Dict
import torch
import numpy as np

from agent.sac import SAC
from agent.replay_memory import Batch


class SACDomainRandomization(SAC):

  def __init__(self, cfg, cfg_arch, rng: np.random.Generator):
    super().__init__(cfg, cfg_arch, rng)
    self.dstb_range = np.asarray(cfg.dstb_range, dtype=np.float32)

  def build_network(self, verbose: bool = True):
    super().build_network(verbose)
    self.critic_aux = self.critics['aux']  # alias

  # Overrides BaseMASAC.update_critic() since we need to have 'dstb' in
  # action_nxt_dict.
  def update_critic(self, batch: Batch) -> Dict[str, float]:
    # Computes action_nxt and feeds to critic.update().
    action_nxt_dict = {}
    entropy_motives_dict = {}
    non_final_append_nxt = batch.info.get('non_final_append_nxt', None)
    with torch.no_grad():
      for key, actor in self.actors.items():
        action_nxt, log_prob_nxt = actor.net.sample(
            batch.non_final_state_nxt, append=non_final_append_nxt
        )
        action_nxt_dict[key] = action_nxt
        entropy_motives_dict[key] = actor.alpha * log_prob_nxt.view(-1)

    # Adds random dstb to action_nxt_dict and entropy_motives_dict.
    action_nxt_dict['dstb'] = torch.FloatTensor(
        self.rng.uniform(
            low=self.dstb_range[:, 0], high=self.dstb_range[:, 1],
            size=(action_nxt.shape[0], self.dstb_range.shape[0])
        )
    ).to(self.device)
    entropy_motives_dict['dstb'] = torch.zeros((action_nxt.shape[0],)
                                              ).to(self.device)

    loss_dict = {}
    for key, critic in self.critics.items():
      loss_dict[key] = critic.update(
          batch, action_nxt_dict, entropy_motives_dict
      )
    return loss_dict
