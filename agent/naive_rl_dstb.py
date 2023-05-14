# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for training SAC to attack a fixed control policy.
"""

from typing import Optional, Callable, Dict, Tuple, Union, List
import copy
import time
import numpy as np
import torch
import wandb
from functools import partial

from agent.naive_rl import NaiveRL
from agent.base_training import BaseTraining
from agent.sac_br import SACBestResponse
from simulators.base_zs_env import BaseZeroSumEnv
from simulators.vec_env.vec_env import VecEnvBase
from utils.dstb import adv_dstb


class NaiveRLDisturbance(NaiveRL):
  policy: SACBestResponse

  def __init__(
      self, cfg_agent, cfg_train, cfg_arch, cfg_env, verbose: bool = True
  ):
    BaseTraining.__init__(self, cfg_agent, cfg_env)

    print("= Constructing policy agent")
    self.policy = SACBestResponse(cfg_train, cfg_arch, self.rng)
    self.policy.build_network(verbose=verbose)
    self.warmup_action_range = np.array(
        cfg_agent.warmup_action_range, dtype=np.float32
    )

    # alias
    self.module_all = [self.policy]

  def sample_action(
      self, obs_all: torch.Tensor
  ) -> Tuple[Dict[str, torch.FloatTensor], Dict[str, np.ndarray]]:
    # Gets controls.
    with torch.no_grad():
      ctrl_all, _ = self.policy.ctrl.net.sample(
          obs_all.float().to(self.device), append=None, latent=None
      )

    # Gets disturbances.
    if self.policy.dstb_use_ctrl:
      obs_dstb_all = torch.cat((obs_all, ctrl_all), dim=-1)
    else:
      obs_dstb_all = obs_all

    if self.cnt_step < self.min_steps_b4_opt:
      dstb_all = self.rng.uniform(
          low=self.warmup_action_range[:, 0],
          high=self.warmup_action_range[:, 1],
          size=(self.n_envs, self.warmup_action_range.shape[0]),
      )
      dstb_all = torch.FloatTensor(dstb_all)
    else:
      with torch.no_grad():
        dstb_all, _ = self.policy.dstb.net.sample(
            obs_dstb_all.float().to(self.device), append=None, latent=None
        )

    action_all_np = [{
        'ctrl': ctrl_all[i].cpu().numpy(),
        'dstb': dstb_all[i].cpu().numpy()
    } for i in range(self.n_envs)]
    action_all = [{
        'ctrl': ctrl_all[[i]].to(self.device),
        'dstb': dstb_all[[i]].to(self.device)
    } for i in range(self.n_envs)]
    return action_all, action_all_np

  def learn(
      self, env: BaseZeroSumEnv, reset_kwargs: Optional[Dict] = None,
      action_kwargs: Optional[Dict] = None,
      rollout_step_callback: Optional[Callable] = None,
      rollout_episode_callback: Optional[Callable] = None,
      visualize_callback: Optional[Callable] = None
  ):
    venv, _ = self.init_learn(env)

    self.check(
        env=env, venv=venv, action_kwargs=action_kwargs,
        rollout_step_callback=rollout_step_callback,
        rollout_episode_callback=rollout_episode_callback,
        visualize_callback=None
    )

    start_learning = time.time()
    if reset_kwargs is None:
      reset_kwargs = {}
    obs_all = venv.reset()
    obs_prev_all = [None for _ in range(self.n_envs)]
    action_prev_all = [None for _ in range(self.n_envs)]
    r_prev_all = [None for _ in range(self.n_envs)]
    done_prev_all = [None for _ in range(self.n_envs)]
    info_prev_all = [None for _ in range(self.n_envs)]
    while self.cnt_step <= self.max_steps:
      # Selects action.
      action_all, action_all_np = self.sample_action(obs_all)

      # Interacts with the env.
      obs_nxt_all, r_all, done_all, info_all = venv.step(action_all_np)
      for env_idx, (done, info) in enumerate(zip(done_all, info_all)):
        if obs_prev_all[env_idx] is not None:
          # Stores the transition in memory.
          self.store_transition(
              obs_prev_all[env_idx].unsqueeze(0), action_prev_all[env_idx],
              r_prev_all[env_idx], obs_all[[env_idx]], done_prev_all[env_idx],
              info_prev_all[env_idx]
          )
        if done:
          obs = venv.reset_one(index=env_idx)
          obs_nxt_all[env_idx] = obs
          g_x = info['g_x']
          if g_x > 0:
            self.cnt_safety_violation += 1
          self.cnt_num_episode += 1

          # Stores this transition with zero controls.
          action = {
              'ctrl': torch.zeros(1, env.action_dim_ctrl).to(self.device),
              'dstb': action_all[env_idx]['dstb']
          }

          self.store_transition(
              obs_all[[env_idx]], action, r_all[env_idx],
              obs_nxt_all[[env_idx]], done, info
          )
          obs_prev_all[env_idx] = None
          action_prev_all[env_idx] = None
          r_prev_all[env_idx] = None
          done_prev_all[env_idx] = None
          info_prev_all[env_idx] = None
        else:
          # Updates the temporary placeholder.
          obs_prev_all[env_idx] = obs_all[env_idx]
          action_prev_all[env_idx] = action_all[env_idx]
          r_prev_all[env_idx] = r_all[env_idx]
          done_prev_all[env_idx] = done_all[env_idx]
          info_prev_all[env_idx] = info_all[env_idx]

      self.violation_record.append(self.cnt_safety_violation)
      self.episode_record.append(self.cnt_num_episode)
      obs_all = obs_nxt_all

      # Optimizes NNs and checks performance.
      if (
          self.cnt_step >= self.min_steps_b4_opt
          and self.cnt_opt_period >= self.opt_freq
      ):
        print(f"Updates at sample step {self.cnt_step}")
        self.policy.move2device(device=self.device)
        self.cnt_opt_period = 0
        loss_critic_dict, loss_actor_dict = self.update(
            self.num_update_per_opt
        )
        loss = [
            loss_critic_dict['central'], loss_actor_dict['dstb'][0],
            loss_actor_dict['dstb'][1], loss_actor_dict['dstb'][2]
        ]
        self.train_record.append(loss)
        self.cnt_opt += 1  # Counts number of optimization.
        if self.cfg_agent.use_wandb:
          log_dict = {
              "loss/critic": loss[0],
              "loss/policy": loss[1],
              "loss/entropy": loss[2],
              "loss/alpha": loss[3],
              "metrics/cnt_safety_violation": self.cnt_safety_violation,
              "metrics/cnt_num_episode": self.cnt_num_episode,
              "hyper_parameters/alpha": self.policy.actor.alpha,
              "hyper_parameters/gamma": self.policy.critic.gamma,
          }
          wandb.log(log_dict, step=self.cnt_step, commit=False)

        # Checks after fixed number of gradient updates.
        if self.cnt_opt % self.check_opt_freq == 0 or self.first_update:
          self.check(
              env=env, venv=venv, action_kwargs=action_kwargs,
              rollout_step_callback=rollout_step_callback,
              rollout_episode_callback=rollout_episode_callback,
              visualize_callback=visualize_callback
          )

          # Resets anyway.
          obs_all = venv.reset()
          obs_prev_all = [None for _ in range(self.n_envs)]
          action_prev_all = [None for _ in range(self.n_envs)]
          r_prev_all = [None for _ in range(self.n_envs)]
          done_prev_all = [None for _ in range(self.n_envs)]
          info_prev_all = [None for _ in range(self.n_envs)]

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
      adv_fn_list = []
      for _ in range(self.n_envs):
        dstb_policy = copy.deepcopy(self.policy.dstb.net)
        dstb_policy.device = "cpu"
        dstb_policy.to("cpu")
        adv_fn_list.append(
            partial(
                adv_dstb, dstb_policy=dstb_policy,
                use_ctrl=self.policy.dstb_use_ctrl
            )
        )
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
