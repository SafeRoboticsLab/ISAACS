# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for training SAC with domain randomization.
"""

from typing import Optional, Callable, Dict, Union, List
import time
import numpy as np
import torch
import wandb
from functools import partial

from agent.sac_dr import SACDomainRandomization
from agent.base_training import BaseTraining
from agent.naive_rl import NaiveRL
from simulators.vec_env.vec_env import VecEnvBase
from simulators.base_zs_env import BaseZeroSumEnv
from utils.dstb import random_dstb


class NaiveRLDomainRandomization(NaiveRL):
  policy: SACDomainRandomization

  def __init__(
      self, cfg_agent, cfg_train, cfg_arch, cfg_env, verbose: bool = True
  ):
    BaseTraining.__init__(self, cfg_agent, cfg_env)

    print("= Constructing policy agent")
    self.policy = SACDomainRandomization(cfg_train, cfg_arch, self.rng)
    self.policy.build_network(verbose=verbose)
    self.warmup_action_range = np.array(
        cfg_agent.warmup_action_range, dtype=np.float32
    )

    # alias
    self.module_all = [self.policy]

  def learn(
      self, env: BaseZeroSumEnv, reset_kwargs: Optional[Dict] = None,
      action_kwargs: Optional[Dict] = None,
      rollout_step_callback: Optional[Callable] = None,
      rollout_episode_callback: Optional[Callable] = None,
      visualize_callback: Optional[Callable] = None
  ):
    venv, agent_copy_list = self.init_learn(env)

    start_learning = time.time()
    if reset_kwargs is None:
      reset_kwargs = {}
    obs_all = venv.reset()
    while self.cnt_step <= self.max_steps:
      # Selects action.
      dstb_all_np = self.rng.uniform(
          low=self.policy.dstb_range[:, 0], high=self.policy.dstb_range[:, 1],
          size=(self.n_envs, self.policy.dstb_range.shape[0])
      )
      dstb_all = torch.FloatTensor(dstb_all_np).to(self.device)

      if self.cnt_step < self.min_steps_b4_opt:
        ctrl_all = self.rng.uniform(
            low=self.warmup_action_range[:, 0],
            high=self.warmup_action_range[:, 1],
            size=(self.n_envs, self.warmup_action_range.shape[0])
        )
        ctrl_all = torch.FloatTensor(ctrl_all).to(self.device)
      else:
        with torch.no_grad():
          ctrl_all, _ = self.policy.actor.net.sample(
              obs_all.float().to(self.device), append=None, latent=None
          )
      action_all_np = [{
          'ctrl': ctrl_all[i].cpu().numpy(),
          'dstb': dstb_all_np[i]
      } for i in range(self.n_envs)]
      action_all = [{
          'ctrl': ctrl_all[[i]],
          'dstb': dstb_all[[i]]
      } for i in range(self.n_envs)]

      # Interacts with the env.
      obs_nxt_all, r_all, done_all, info_all = venv.step(action_all_np)
      for env_idx, (done, info) in enumerate(zip(done_all, info_all)):
        # Store the transition in memory
        self.store_transition(
            obs_all[[env_idx]], action_all[env_idx], r_all[env_idx],
            obs_nxt_all[[env_idx]], done, info
        )

        if done:
          obs = venv.reset_one(index=env_idx)
          obs_nxt_all[env_idx] = obs
          g_x = info['g_x']
          if g_x > 0:
            self.cnt_safety_violation += 1
          self.cnt_num_episode += 1
      self.violation_record.append(self.cnt_safety_violation)
      self.episode_record.append(self.cnt_num_episode)
      obs_all = obs_nxt_all

      # Optimizes NNs and checks performance.
      if (
          self.cnt_step >= self.min_steps_b4_opt
          and self.cnt_opt_period >= self.opt_freq
      ):
        print(f"Updates at sample step {self.cnt_step}")
        self.cnt_opt_period = 0
        loss_critic_dict, loss_actor_dict = self.update(
            self.num_update_per_opt
        )
        loss = [
            loss_critic_dict['central'], loss_critic_dict['aux'],
            loss_actor_dict['ctrl'][0], loss_actor_dict['ctrl'][1],
            loss_actor_dict['ctrl'][2]
        ]
        self.train_record.append(loss)
        self.cnt_opt += 1  # Counts number of optimization.
        if self.cfg_agent.use_wandb:
          log_dict = {
              "loss/critic": loss[0],
              "loss/critic_aux": loss[1],
              "loss/policy": loss[2],
              "loss/entropy": loss[3],
              "loss/alpha": loss[4],
              "metrics/cnt_safety_violation": self.cnt_safety_violation,
              "metrics/cnt_num_episode": self.cnt_num_episode,
              "hyper_parameters/alpha": self.policy.actor.alpha,
              "hyper_parameters/gamma": self.policy.critic.gamma,
          }
          wandb.log(log_dict, step=self.cnt_step, commit=False)

        # Checks after fixed number of gradient updates.
        if self.cnt_opt % self.check_opt_freq == 0 or self.first_update:
          # Updates the agent in the environment with the newest policy.
          env.agent.policy.update_policy(self.policy.actor.net)
          for agent in agent_copy_list:
            agent.policy.update_policy(self.policy.actor.net)
          venv.set_attr("agent", agent_copy_list, value_batch=True)

          reset_kwargs_list = []  # Same initial states.
          for _ in range(self.cfg_agent.num_eval_traj):
            env.reset()
            reset_kwargs_list.append({"state": np.copy(env.state)})

          self.check(
              env=env, venv=venv, action_kwargs=action_kwargs,
              rollout_step_callback=rollout_step_callback,
              rollout_episode_callback=rollout_episode_callback,
              visualize_callback=visualize_callback
          )

          # Resets anyway.
          obs_all = venv.reset()

      # Updates counter.
      self.cnt_step += self.n_envs
      self.cnt_opt_period += self.n_envs

      # Updates gamma, lr, etc.
      for _ in range(self.n_envs):
        self.policy.update_hyper_param()

    self.save(venv, force_save=True)
    end_learning = time.time()
    time_learning = end_learning - start_learning
    print('\nLearning: {:.1f}'.format(time_learning))

    train_record = np.array(self.train_record)
    train_progress = np.array(self.train_progress)
    violation_record = np.array(self.violation_record)
    episode_record = np.array(self.episode_record)
    return (
        train_record, train_progress, violation_record, episode_record,
        self.pq_top_k
    )

  def save(
      self, venv: VecEnvBase, force_save: bool = False,
      reset_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      action_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      rollout_step_callback: Optional[Callable] = None,
      rollout_episode_callback: Optional[Callable] = None
  ) -> Dict:
    if force_save:
      info = {}
      metric = 0.
    else:
      num_eval_traj: int = self.cfg_agent.num_eval_traj
      rollout_end_criterion: str = self.cfg_agent.rollout_end_criterion
      adv_fn_list = [
          partial(
              random_dstb, rng=self.rng, dstb_range=self.policy.dstb_range
          ) for _ in range(self.n_envs)
      ]

      _, results, length = venv.simulate_trajectories_zs(
          num_trajectories=num_eval_traj,
          T_rollout=self.cfg_agent.eval_timeout,
          end_criterion=rollout_end_criterion, adversary=adv_fn_list,
          reset_kwargs_list=reset_kwargs_list,
          action_kwargs_list=action_kwargs_list,
          rollout_step_callback=rollout_step_callback,
          rollout_episode_callback=rollout_episode_callback, use_tqdm=True
      )

      safe_rate = np.sum(results != -1) / num_eval_traj
      if rollout_end_criterion == "reach-avoid":
        success_rate = np.sum(results == 1) / num_eval_traj
        metric = success_rate
        info = dict(
            safe_rate=safe_rate, ep_length=np.mean(length), metric=metric
        )
      else:
        metric = safe_rate
        info = dict(ep_length=np.mean(length), metric=metric)

    if self.policy.actor_type == 'max':
      # Maximizes cost -> minimizes safe/success rate.
      BaseTraining._save(self, metric=1 - metric, force_save=force_save)
    else:
      BaseTraining._save(self, metric=metric, force_save=force_save)
    return info
