# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A parent class for training agents.

This file implements a parent class for all training agents, modified from
https://github.com/SafeRoboticsLab/SimLabReal/blob/main/agent/base_training.py
"""

from abc import ABC, abstractmethod
import warnings
from typing import Optional, Union, List, Tuple, Dict, Callable
from collections import defaultdict
from queue import PriorityQueue
import os
import copy
import numpy as np
import torch
import wandb

from agent.replay_memory import ReplayMemory, Transition, Batch
from agent.base_ma_sac import BaseMASAC
from simulators import BaseEnv, BaseSingleEnv, BaseZeroSumEnv
from simulators.vec_env.vec_env import VecEnvBase
from simulators.agent import Agent


class BaseTraining(ABC):
  module_all: List[BaseMASAC]
  module_folder_all: List[str]
  policy: Optional[BaseMASAC]
  performance: Optional[BaseMASAC]
  backup: Optional[BaseMASAC]

  def __init__(self, cfg_agent, cfg_env):
    self.cfg_agent = copy.deepcopy(cfg_agent)

    self.device = torch.device(cfg_agent.device)
    self.n_envs = int(cfg_agent.num_envs)

    # ! We assume all modules use the same parameters.
    self.batch_size = int(cfg_agent.batch_size)
    self.max_model = int(cfg_agent.max_model)

    # Replay Buffer.
    self.build_memory(cfg_agent.memory_capacity, cfg_env.seed)
    self.rng = np.random.default_rng(seed=cfg_env.seed)
    self.transition_cls = Transition

    # Checkpoints.
    if isinstance(cfg_agent.save_top_k, int):
      self.save_top_k = int(cfg_agent.save_top_k)
    else:
      self.save_top_k = cfg_agent.save_top_k
    self.pq_top_k = PriorityQueue()

    self.use_wandb = cfg_agent.use_wandb

    # Placeholders.
    self.policy = None
    self.performance = None
    self.backup = None

  @property
  @abstractmethod
  def has_backup(self):
    raise NotImplementedError

  def build_memory(self, capacity, seed):
    self.memory = ReplayMemory(capacity, seed)

  def sample_batch(
      self, batch_size: Optional[int] = None, recent_size: int = 0
  ) -> Batch:
    if batch_size is None:
      batch_size = self.batch_size
    if recent_size > 0:  # use recent
      transitions = self.memory.sample_recent(batch_size, recent_size)
    else:
      transitions = self.memory.sample(batch_size)
    return Batch(transitions, device=self.device)

  def store_transition(self, *args):
    self.memory.update(self.transition_cls(*args))

  @abstractmethod
  def save(
      self, venv: VecEnvBase, force_save: bool = False,
      reset_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      action_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      rollout_step_callback: Optional[Callable] = None,
      rollout_episode_callback: Optional[Callable] = None
  ) -> Dict:
    raise NotImplementedError

  def _save(self, metric: Optional[float] = None, force_save: bool = False):
    if self.cnt_step == 0:
      return

    assert metric is not None or force_save, (
        "should provide metric of force save"
    )
    save_current = False
    if force_save:
      save_current = True
    elif self.pq_top_k.qsize() < self.save_top_k:
      self.pq_top_k.put((metric, self.cnt_step))
      save_current = True
    elif metric > self.pq_top_k.queue[0][0]:  # overwrite
      # Remove old one
      _, step_remove = self.pq_top_k.get()
      for module, module_folder in zip(
          self.module_all, self.module_folder_all
      ):
        module.remove(int(step_remove), module_folder)
      self.pq_top_k.put((metric, self.cnt_step))
      save_current = True

    if save_current:
      print('Saving current model...')
      for module, module_folder in zip(
          self.module_all, self.module_folder_all
      ):
        module.save(self.cnt_step, module_folder, self.max_model)
      print(self.pq_top_k.queue)

  def restore(
      self, step, model_folder: str, agent_type: Optional[str] = None,
      actor_path: Optional[Union[List[str], str]] = None
  ):
    """Restore the weights of the neural network.

    Args:
        step (int): #updates trained.
        model_folder (str): the path to the models, under this folder there
            should be a folder named "agent_type". There are critic/ and agent/
            folders under model_folder/agent_type.
        agent_type (str, optional): performance, backup, or single agent
            (None). Defaults to None.
        actor_path (str, optional): the path to the actor model. Defaults to
            None.
    """
    if agent_type is None:
      agent_type = "agent"
      model_folder = os.path.join(model_folder)
    else:
      model_folder = os.path.join(model_folder, agent_type)

    if agent_type == 'agent':
      self.policy.restore(step, model_folder, actor_path)
    elif agent_type == 'backup':
      self.backup.restore(step, model_folder, actor_path)
    elif agent_type == 'performance':
      self.performance.restore(step, model_folder, actor_path)
    else:
      raise ValueError("Agent type ({}) is not supported".format(agent_type))
    print(
        '  <= Restore {} with {} updates from {}.'.format(
            agent_type, step, model_folder
        )
    )

  @abstractmethod
  def learn(
      self, env: BaseEnv, reset_kwargs: Optional[Dict] = None,
      action_kwargs: Optional[Dict] = None,
      rollout_step_callback: Optional[Callable] = None,
      rollout_episode_callback: Optional[Callable] = None,
      visualize_callback: Optional[Callable] = None
  ):
    raise NotImplementedError

  def init_learn(self, env: BaseEnv) -> Tuple[VecEnvBase, List[Agent]]:

    # Training hyper-parameters.
    self.max_steps = int(self.cfg_agent.max_steps)
    self.opt_freq = int(self.cfg_agent.opt_freq)
    self.num_update_per_opt = int(self.cfg_agent.update_per_opt)
    self.check_opt_freq = int(self.cfg_agent.check_opt_freq)
    self.min_steps_b4_opt = int(self.cfg_agent.min_steps_b4_opt)
    self.out_folder: str = self.cfg_agent.out_folder
    self.rollout_end_criterion: str = self.cfg_agent.rollout_end_criterion

    # Placeholders for training records.
    self.train_record = []
    self.train_progress = []
    self.violation_record = []
    self.episode_record = []
    self.cnt_opt: int = 0
    self.cnt_opt_period: int = 0
    self.cnt_safety_violation: int = 0
    self.cnt_num_episode: int = 0
    self.cnt_step: int = 0
    self.first_update: bool = True

    # Logs checkpoints and visualizations.
    self.model_folder = os.path.join(self.out_folder, 'model')
    os.makedirs(self.model_folder, exist_ok=True)
    self.figure_folder = os.path.join(self.out_folder, 'figure')
    os.makedirs(self.figure_folder, exist_ok=True)
    self.module_folder_all = [self.model_folder]

    venv = VecEnvBase([copy.deepcopy(env) for _ in range(self.n_envs)],
                      device=self.device)
    venv.seed(env.seed_val)

    # Venv can run on a different device.
    agent_copy_list = [copy.deepcopy(env.agent) for _ in range(self.n_envs)]
    for agent in agent_copy_list:
      agent.policy.to(torch.device(self.cfg_agent.venv_device))
    venv.set_attr("agent", agent_copy_list, value_batch=True)
    return venv, agent_copy_list

  def update(self, num_update_per_opt: int,
             **kwargs) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    loss_critic_dict_all = defaultdict(list)
    loss_actor_dict_all = defaultdict(list)

    for timer in range(num_update_per_opt):
      sample = True
      cnt = 0
      while sample:
        batch = self.sample_batch()
        sample = torch.logical_not(torch.any(batch.non_final_mask))
        cnt += 1
        if cnt >= 10:
          break
      if sample:
        warnings.warn("Cannot get a valid batch!!", UserWarning)
        continue

      loss_dict = self.policy.update(batch, timer, **kwargs)
      loss_critic_dict: dict = loss_dict['critic']
      loss_actor_dict: dict = loss_dict['actor']
      flag_dict: dict = loss_dict['flag']

      for key, value in loss_critic_dict.items():
        loss_critic_dict_all[key].append(value)

      for key, value in loss_actor_dict.items():
        if flag_dict[key]:
          loss_actor_dict_all[key].append(value)

    loss_critic_dict_avg = {
        key: np.mean(np.asarray(value))
        for key, value in loss_critic_dict_all.items()
    }
    loss_actor_dict_avg = {
        key: np.mean(np.asarray(value), axis=0)
        for key, value in loss_actor_dict_all.items()
    }
    return loss_critic_dict_avg, loss_actor_dict_avg

  def check(
      self, env: Union[BaseSingleEnv, BaseZeroSumEnv], venv: VecEnvBase,
      reset_kwargs_list: Optional[List[Dict]] = None,
      action_kwargs: Optional[Dict] = None,
      rollout_step_callback: Optional[Callable] = None,
      rollout_episode_callback: Optional[Callable] = None,
      visualize_callback: Optional[Callable] = None
  ):
    if self.cnt_step == 0:
      print("Before training:")
    else:
      print(f"Checks at sample step {self.cnt_step}:")
      self.first_update = False
      self.cnt_opt = 0
    save_dict = self.save(
        venv, force_save=False, reset_kwargs_list=reset_kwargs_list,
        action_kwargs_list=action_kwargs,
        rollout_step_callback=rollout_step_callback,
        rollout_episode_callback=rollout_episode_callback
    )

    # Logs.
    if self.cnt_step != 0:
      print('  => Safety violations: {:d}'.format(self.cnt_safety_violation))
    if self.rollout_end_criterion == "reach-avoid":
      success_rate = save_dict['metric']
      safe_rate = save_dict['safe_rate']
      self.train_progress.append(np.array([success_rate, safe_rate]))
      print('  => Success rate: {:.2f}'.format(success_rate))
    elif self.rollout_end_criterion == "failure":
      safe_rate = save_dict['metric']
      self.train_progress.append(np.array([safe_rate]))
      print('  => Safe rate: {:.2f}'.format(safe_rate))
    else:
      raise ValueError(f"Invalid end criterion {self.rollout_end_criterion}!")

    if self.cfg_agent.use_wandb:
      log_dict = {}
      if self.rollout_end_criterion == "reach-avoid":
        log_dict["metrics/success_rate"] = save_dict['metric']
      elif self.rollout_end_criterion == "failure":
        log_dict["metrics/safe_rate"] = save_dict['metric']
      save_dict.pop('metric')
      for k, v in save_dict.items():
        log_dict[f"metrics/{k}"] = v
      wandb.log(log_dict, step=self.cnt_step, commit=True)
    torch.save({
        'train_record': self.train_record,
        'train_progress': self.train_progress,
        'violation_record': self.violation_record,
    }, os.path.join(self.out_folder, 'train_details'))

    # Visualizes.
    if visualize_callback is not None:
      visualize_callback(
          env, self.policy,
          os.path.join(self.figure_folder, f"{self.cnt_step}.png")
      )
