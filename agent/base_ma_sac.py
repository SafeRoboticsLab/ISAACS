# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A parent class for soft actor-critic variants.

This file implements the multi-agent SAC base class following the centralized
training and decentralized execution framework. It assumes all actors take in
the same state, but some actors may have the `privilege` to see other actors.
We support this functionality through `obs_other_list` (Please refer to L209 in
`base_block.py`). Also, there is a central critic for providing policy
graidents for actors.

Note that the `cfg_arch` needs to specify the exact architecture. Parameters in
`cfg` won't influence the architecture but only the training. Please refer to
`config/isaacs.yaml` for an example.
"""

from typing import Optional, Dict, Tuple
from collections import defaultdict
import copy
import numpy as np
import torch

from agent.base_block import Actor, Critic
from agent.replay_memory import Batch


class BaseMASAC():

  def __init__(self, cfg, cfg_arch, rng: np.random.Generator):
    """
    Args:
        cfg (object): update-related hyper-parameters configuration.
        cfg_arch (object): NN architecture configuration.
    """
    self.cfg = copy.deepcopy(cfg)
    self.cfg_arch = copy.deepcopy(cfg_arch)
    self.rng = rng
    self.device: str = cfg.device

    self.num_actors = int(cfg.num_actors)
    self.num_critics = int(cfg.num_critics)

    self.actors: Dict[str, Actor] = {}
    self.critics: Dict[str, Critic] = {}
    # self.pg_target_map[critic] contains actors using this critic for policy
    # gradient.
    self.pg_target_map = defaultdict(list)

  def build_network(self, verbose: bool = True):
    for idx in range(self.num_critics):
      cfg_critic = getattr(self.cfg, f"critic_{idx}")
      cfg_arch_critic = getattr(self.cfg_arch, f"critic_{idx}")
      critic = Critic(
          cfg=cfg_critic, cfg_arch=cfg_arch_critic, verbose=verbose,
          device=self.device
      )
      self.critics[cfg_critic.net_name] = critic
    assert "central" in self.critics, "Must have a central critic."

    for idx in range(self.num_actors):
      cfg_actor = getattr(self.cfg, f"actor_{idx}")
      cfg_arch_actor = getattr(self.cfg_arch, f"actor_{idx}")
      actor = Actor(
          cfg=cfg_actor, cfg_arch=cfg_arch_actor, verbose=verbose,
          device=self.device
      )
      self.actors[cfg_actor.net_name] = actor
      if not actor.eval:
        self.pg_target_map[actor.pg_target].append(actor.net_name)
    self.action_dim_all: int = np.sum(
        np.array([x.action_dim for x in self.actors.values()])
    )
    self.critic = self.critics['central']  # alias

  def update_critic_hyper_param(self):
    for critic_name, critic in self.critics.items():
      if critic.eval:
        continue
      flag_rst_alpha = critic.update_hyper_param()
      if flag_rst_alpha:
        for actor_name in self.pg_target_map[critic_name]:
          self.actors[actor_name].reset_alpha()

  def update_actor_hyper_param(self):
    for actor in self.actors.values():
      if actor.eval:
        continue
      actor.update_hyper_param()

  def update_hyper_param(self):
    self.update_critic_hyper_param()
    self.update_actor_hyper_param()

  def update_actor(
      self, batch: Batch, timer: int
  ) -> Tuple[Dict[str, Tuple[float, float, float]], Dict[str, bool]]:
    """Updates all the actors (simultaneously).

    Args:
        batch (Batch): sampled transitions.
        timer (int): current optimization step.

    Returns:
        Dict[str, (float, float, float)]: dictionary of loss for each actor.
    """
    loss_dict = {}
    flag_dict = {}
    for key, actor in self.actors.items():
      if actor.eval:
        continue
      if timer % actor.update_period == 0:
        loss_dict[key] = actor.update(batch, self.critics[actor.pg_target])
        flag_dict[key] = True
      else:
        loss_dict[key] = 0., 0., 0.  # ! dummy values
        flag_dict[key] = False
    return loss_dict, flag_dict

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

    loss_dict = {}
    for key, critic in self.critics.items():
      if critic.eval:
        continue
      loss_dict[key] = critic.update(
          batch, action_nxt_dict, entropy_motives_dict
      )
    return loss_dict

  def update_critic_target(self, timer: int):
    for critic in self.critics.values():
      if timer % critic.update_target_period == 0:
        critic.update_target()

  def update(self, batch: Batch, timer: int) -> dict:
    loss_c_dict = self.update_critic(batch)
    loss_a_dict, flag_dict = self.update_actor(batch, timer)
    self.update_critic_target(timer)
    return {'critic': loss_c_dict, 'actor': loss_a_dict, 'flag': flag_dict}

  def save(
      self, step: int, model_folder: str, max_model: Optional[int] = None
  ):
    for critic in self.critics.values():
      if critic.eval:
        continue
      critic.save(step, model_folder, max_model)
    for actor in self.actors.values():
      if actor.eval:
        continue
      actor.save(step, model_folder, max_model)

  def restore(self, step: int, model_folder: str):
    for critic in self.critics.values():
      if critic.eval:
        continue
      critic.restore(step, model_folder)
    for actor in self.actors.values():
      if actor.eval:
        continue
      actor.restore(step, model_folder)

  def remove(self, step: int, model_folder: str):
    for critic in self.critics.values():
      if critic.eval:
        continue
      critic.remove(step, model_folder)
    for actor in self.actors.values():
      if actor.eval:
        continue
      actor.remove(step, model_folder)

  def move2device(self, device: str):
    for critic in self.critics.values():
      critic.net.to(device)
    for actor in self.actors.values():
      actor.net.to(device)
