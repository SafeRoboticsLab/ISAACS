# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Classes for basic building blocks in soft actor-critic (SAC).

This file implements `Actor` and `Critic` building blocks. Each block has a
`net` attribute, which is a `torch.Module`. `BaseBlock` implements basic
operators, e.g., build_optimizer, save, restore, and, remove, but requires its
children to implement `build_network()` and `update()`
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, Dict
import os
import copy
import numpy as np
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from torch.nn.functional import mse_loss

from agent.model import GaussianPolicy, TwinnedQNetwork
from agent.scheduler import StepLRMargin
from agent.replay_memory import Batch
from utils.train import soft_update, save_model, get_bellman_update


class BaseBlock(ABC):
  net: torch.nn.Module

  def __init__(self, cfg, cfg_arch, device: str) -> None:
    self.eval = cfg.eval
    self.device = device
    self.net_name: str = cfg.net_name
    self.has_latent: bool = cfg_arch.latent_dim > 0

  @abstractmethod
  def build_network(self, verbose: bool = True):
    raise NotImplementedError

  def build_optimizer(self, cfg):
    if cfg.opt_type == "AdamW":
      self.opt_cls = AdamW
    elif cfg.opt_type == "AdamW":
      self.opt_cls = Adam
    else:
      raise ValueError("Not supported optimizer type!")

    # Learning Rate
    self.lr_schedule: bool = cfg.lr_schedule
    if self.lr_schedule:
      self.lr_period = int(cfg.lr_period)
      self.lr_decay = float(cfg.lr_decay)
      self.lr_end = float(cfg.lr_end)
    self.lr = float(cfg.lr)

    # Builds the optimizer.
    self.optimizer = self.opt_cls(
        self.net.parameters(), lr=self.lr, weight_decay=0.01
    )
    if self.lr_schedule:
      self.scheduler = StepLR(
          self.optimizer, step_size=self.lr_period, gamma=self.lr_decay
      )

  def update_hyper_param(self):
    if self.lr_schedule:
      lr = self.optimizer.state_dict()['param_groups'][0]['lr']
      if lr <= self.lr_end:
        for param_group in self.optimizer.param_groups:
          param_group['lr'] = self.lr_end
      else:
        self.scheduler.step()

  @abstractmethod
  def update(self, batch: Batch):
    raise NotImplementedError

  def save(
      self, step: int, model_folder: str, max_model: Optional[int] = None,
      verbose: bool = True
  ) -> str:
    path = os.path.join(model_folder, self.net_name)
    save_model(self.net, step, path, self.net_name, max_model)
    if verbose:
      print(f"  => Saves {self.net_name} at {path}.")
    return path

  def restore(self, step: int, model_folder: str, verbose: bool = True) -> str:
    path = os.path.join(
        model_folder, self.net_name, f'{self.net_name}-{step}.pth'
    )
    self.net.load_state_dict(torch.load(path, map_location=self.device))
    self.net.to(self.device)
    if verbose:
      print(f"  => Restores {self.net_name} at {path}.")
    return path

  def remove(self, step: int, model_folder: str, verbose: bool = True) -> str:
    path = os.path.join(
        model_folder, self.net_name, f'{self.net_name}-{step}.pth'
    )
    if verbose:
      print(f"  => Removes {self.net_name} at {path}.")
    if os.path.exists(path):
      os.remove(path)
    return path


class Actor(BaseBlock):
  # TODO: different actor types.
  net: GaussianPolicy

  def __init__(self, cfg, cfg_arch, device: str, verbose: bool = True) -> None:
    super().__init__(cfg, cfg_arch, device)
    self.action_dim = int(cfg_arch.action_dim)
    self.action_range = np.array(cfg_arch.action_range, dtype=np.float32)
    self.actor_type: str = cfg.actor_type
    self.obs_other_list: list = getattr(cfg, "obs_other_list", None)

    if not self.eval:
      self.pg_target: str = cfg.pg_target
      self.update_period = int(cfg.update_period)

    self.build_network(cfg, cfg_arch, verbose=verbose)

  @property
  def alpha(self):
    return self.log_alpha.exp()

  def build_network(self, cfg, cfg_arch, verbose: bool = True):
    self.net = GaussianPolicy(
        obs_dim=cfg_arch.obs_dim, mlp_dim=cfg_arch.mlp_dim,
        action_dim=self.action_dim, action_range=self.action_range,
        append_dim=cfg_arch.append_dim, latent_dim=cfg_arch.latent_dim,
        activation_type=cfg_arch.activation, device=self.device,
        verbose=verbose
    )

    # Loads model if specified.
    if hasattr(cfg_arch, "pretrained_path"):
      if cfg_arch.pretrained_path is not None:
        pretrained_path = cfg_arch.pretrained_path
        self.net.load_state_dict(
            torch.load(pretrained_path, map_location=self.device)
        )
        print(f"--> Loads {self.net_name} from {pretrained_path}.")

    if self.eval:
      self.net.eval()
      for _, param in self.net.named_parameters():
        param.requires_grad = False
      self.log_alpha = torch.log(torch.FloatTensor([1e-8])).to(self.device)
    else:
      self.build_optimizer(cfg)

  def build_optimizer(self, cfg):
    super().build_optimizer(cfg)

    # entropy-related parameters
    self.init_alpha = torch.log(torch.FloatTensor([cfg.alpha])).to(self.device)
    self.learn_alpha: bool = cfg.learn_alpha
    self.log_alpha = self.init_alpha.detach().clone()
    self.target_entropy = torch.tensor(-self.action_dim).to(self.device)
    if self.learn_alpha:
      self.log_alpha.requires_grad = True
      self.lr_al: float = cfg.lr_al
      self.lr_al_schedule: bool = cfg.lr_al_schedule
      self.log_alpha_optimizer = self.opt_cls([self.log_alpha], lr=self.lr_al,
                                              weight_decay=0.01)
      if self.lr_al_schedule:
        self.lr_al_period: int = cfg.lr_al_period
        self.lr_al_decay: float = cfg.lr_al_decay
        self.lr_al_end: float = cfg.lr_al_end
        self.log_alpha_scheduler = StepLR(
            self.log_alpha_optimizer, step_size=self.lr_al_period,
            gamma=self.lr_al_decay
        )

  def update_hyper_param(self):
    super().update_hyper_param()

    if self.learn_alpha and self.lr_al_schedule:
      lr = self.log_alpha_optimizer.state_dict()['param_groups'][0]['lr']
      if lr <= self.lr_al_end:
        for param_group in self.log_alpha_optimizer.param_groups:
          param_group['lr'] = self.lr_al_end
      else:
        self.log_alpha_scheduler.step()

  def reset_alpha(self):
    self.log_alpha = self.init_alpha.detach().clone()
    if self.learn_alpha:
      self.log_alpha.requires_grad = True
      self.log_alpha_optimizer = self.opt_cls([self.log_alpha], lr=self.lr_al,
                                              weight_decay=0.01)
      if self.lr_al_schedule:
        self.log_alpha_scheduler = StepLR(
            self.log_alpha_optimizer, step_size=self.lr_al_period,
            gamma=self.lr_al_decay
        )

  def update(self, batch: Batch, critic: Critic) -> Tuple[float, float, float]:
    self.net.train()
    critic.net.eval()

    if 'append' in batch.info:
      append = batch.info['append']
    else:
      append = None

    if 'latent' in batch.info and self.has_latent:
      latent = batch.info['latent']
    else:
      latent = None

    if self.obs_other_list is not None:
      other_actions = torch.cat([
          batch.action[key] for key in self.obs_other_list
      ], dim=-1)
      state = torch.cat([batch.state, other_actions], dim=-1)
    else:
      state = batch.state

    action_sample, log_prob = self.net.sample(
        state=state, append=append, latent=latent
    )
    if isinstance(critic.action_src, list):
      tmp_action_list = []
      for key in critic.action_src:
        if key == self.net_name:
          tmp_action_list.append(action_sample)
        else:
          tmp_action_list.append(batch.action[key])
      action = torch.cat(tmp_action_list, dim=-1)
    else:
      action = action_sample

    q_pi_1, q_pi_2 = critic.net(
        batch.state, action, append=append, latent=latent
    )

    if self.actor_type == "min":
      q_pi = torch.max(q_pi_1, q_pi_2)
    elif self.actor_type == "max":
      q_pi = torch.min(q_pi_1, q_pi_2)

    # cost: min_theta E[ Q + alpha * (log pi + H)]
    # loss_pi = Q + alpha * log pi
    # reward: max_theta E[ Q - alpha * (log pi + H)]
    # loss_pi = -Q + alpha * log pi
    loss_entropy = self.alpha * log_prob.view(-1).mean()
    if self.actor_type == "min":
      loss_q_eval = q_pi.mean()
    elif self.actor_type == "max":
      loss_q_eval = -q_pi.mean()
    loss_pi = loss_q_eval + loss_entropy

    self.optimizer.zero_grad()
    loss_pi.backward()
    self.optimizer.step()

    # Automatic temperature tuning
    loss_alpha = (self.alpha *
                  (-log_prob.detach() - self.target_entropy)).mean()
    if self.learn_alpha:
      self.log_alpha_optimizer.zero_grad()
      loss_alpha.backward()
      self.log_alpha_optimizer.step()
      self.log_alpha.data = torch.min(
          self.log_alpha.data, self.init_alpha.data
      )

    return loss_q_eval.item(), loss_entropy.item(), loss_alpha.item()


class Critic(BaseBlock):
  net: TwinnedQNetwork

  def __init__(self, cfg, cfg_arch, device: str, verbose: bool = True) -> None:
    super().__init__(cfg, cfg_arch, device)
    # The name of actors that this critic uses.
    if isinstance(cfg.action_src, str):
      self.action_src = cfg.action_src
    else:
      self.action_src = list(cfg.action_src)

    if not self.eval:
      self.mode: str = cfg.mode
      self.update_target_period = int(cfg.update_target_period)

    self.build_network(cfg, cfg_arch, verbose=verbose)

  def build_network(self, cfg, cfg_arch, verbose: bool = True):

    self.net = TwinnedQNetwork(
        obs_dim=cfg_arch.obs_dim, mlp_dim=cfg_arch.mlp_dim,
        action_dim=cfg_arch.action_dim, append_dim=cfg_arch.append_dim,
        latent_dim=cfg_arch.latent_dim, activation_type=cfg_arch.activation,
        device=self.device, verbose=verbose
    )

    # Loads model if specified.
    if hasattr(cfg_arch, "pretrained_path"):
      if cfg_arch.pretrained_path is not None:
        pretrained_path = cfg_arch.pretrained_path
        self.net.load_state_dict(
            torch.load(pretrained_path, map_location=self.device)
        )
        print(f"--> Loads {self.net_name} from {pretrained_path}.")

    if self.eval:
      self.net.eval()
      for _, param in self.net.named_parameters():
        param.requires_grad = False
      self.target = self.net  # alias
    else:
      self.target = copy.deepcopy(self.net)
      self.build_optimizer(cfg)

  def build_optimizer(self, cfg):
    super().build_optimizer(cfg)
    self.terminal_type: str = cfg.terminal_type
    self.tau = float(cfg.tau)

    # Discount factor
    self.gamma_schedule: bool = cfg.gamma_schedule
    if self.gamma_schedule:
      self.gamma_scheduler = StepLRMargin(
          init_value=cfg.gamma, period=cfg.gamma_period, decay=cfg.gamma_decay,
          end_value=cfg.gamma_end, goal_value=1.
      )
      self.gamma: float = self.gamma_scheduler.get_variable()
    else:
      self.gamma: float = cfg.gamma

  def update_hyper_param(self) -> bool:
    """Updates the hyper-parameters of the critic, e.g., discount factor.

    Returns:
        bool: True if the discount factor (gamma) is updated.
    """
    super().update_hyper_param()
    if self.gamma_schedule:
      old_gamma = self.gamma_scheduler.get_variable()
      self.gamma_scheduler.step()
      self.gamma = self.gamma_scheduler.get_variable()
      if self.gamma != old_gamma:
        return True
    return False

  def update(
      self, batch: Batch, action_nxt_dict: Dict[str, torch.FloatTensor],
      entropy_motives_dict: Dict[str, torch.FloatTensor]
  ) -> float:
    self.net.train()
    self.target.eval()

    if 'append' in batch.info:
      append = batch.info['append']
      non_final_append_nxt = batch.info['non_final_append_nxt']
    else:
      append = None
      non_final_append_nxt = None

    if 'latent' in batch.info and self.has_latent:
      latent = batch.info['latent']
    else:
      latent = None

    if isinstance(self.action_src, list):
      action = torch.cat([batch.action[key] for key in self.action_src],
                         dim=-1)
      action_nxt = torch.cat([action_nxt_dict[key] for key in self.action_src],
                             dim=-1)
      tmp = torch.cat([entropy_motives_dict[key] for key in self.action_src],
                      dim=-1)
      entropy_motives = torch.sum(tmp, dim=-1)
    else:
      action = batch.action[self.action_src]
      action_nxt = action_nxt_dict[self.action_src]
      entropy_motives = entropy_motives_dict[self.action_src]

    # Gets Q(s, a).
    q1, q2 = self.net(batch.state, action, append=append, latent=latent)

    next_q1, next_q2 = self.target(
        batch.non_final_state_nxt, action_nxt, append=non_final_append_nxt,
        latent=latent
    )

    # Gets Bellman update.
    y = get_bellman_update(
        self.mode, q1.shape[0], self.device, next_q1, next_q2,
        batch.non_final_mask, batch.reward, batch.info['g_x'],
        batch.info['l_x'], batch.info['binary_cost'], self.gamma,
        terminal_type=self.terminal_type
    )

    if self.mode == 'performance':
      y[batch.non_final_mask] += self.gamma * entropy_motives

    # Regresses MSE loss for both Q1 and Q2.
    loss_q1 = mse_loss(input=q1.view(-1), target=y)
    loss_q2 = mse_loss(input=q2.view(-1), target=y)
    loss_q = loss_q1 + loss_q2

    # Backpropagates.
    self.optimizer.zero_grad()
    loss_q.backward()
    self.optimizer.step()

    return loss_q.item()

  def update_target(self):
    soft_update(self.target, self.net, self.tau)

  def restore(self, step: int, model_folder: str, verbose: bool = True):
    path = super().restore(step, model_folder, verbose=verbose)
    if not self.eval:
      self.target.load_state_dict(torch.load(path, map_location=self.device))
      self.target.to(self.device)

  def value(
      self,
      obs: np.ndarray,
      action: np.ndarray,
      append: Optional[Union[np.ndarray, torch.Tensor]] = None,
      latent: Optional[Union[np.ndarray, torch.Tensor]] = None,
  ) -> np.ndarray:
    action = torch.FloatTensor(action).to(self.device)
    if not self.has_latent:
      latent = None
    with torch.no_grad():
      q_pi_1, q_pi_2 = self.net(obs, action, append=append, latent=latent)
    value = (q_pi_1+q_pi_2) / 2
    return value
