# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for traing control and disturbance policies under self-play.

Please refer to Sec 3.1 and Algorithm 1 in https://arxiv.org/abs/2212.03228
for implementation details and derivations.
"""

from typing import Optional, Callable, Dict, Tuple, Union, List, Any
import copy
import time
import numpy as np
import torch
import wandb
from functools import partial

from agent.sac_adv import SACAdv
from agent.base_training import BaseTraining
from simulators.base_zs_env import BaseZeroSumEnv
from simulators.vec_env.vec_env import VecEnvBase
from utils.dstb import adv_dstb, dummy_dstb


class ISAACS(BaseTraining):
  policy: SACAdv

  def __init__(
      self, cfg_agent, cfg_train, cfg_arch, cfg_env, verbose: bool = True
  ):
    BaseTraining.__init__(self, cfg_agent, cfg_env)

    print("= Constructing policy agent")
    self.policy = SACAdv(cfg_train, cfg_arch, self.rng)
    self.policy.build_network(verbose=verbose)
    self.dstb_dup = copy.deepcopy(self.policy.dstb)

    self.save_top_k = cfg_agent.save_top_k
    self.dstb_res_dict: Dict[int, Tuple] = {
        -1: (0, 0.)
    }  # key: step, value: (#gameplays, metric_avg)

    # alias
    self.module_all = [self.policy]
    self.performance = self.policy

    # This algorithm can also train a single actor by fixing the other.
    self.fix_ctrl: bool = self.policy.ctrl.eval
    self.fix_dstb: bool = self.policy.dstb.eval
    if not self.fix_ctrl:
      self.warmup_ctrl_range = np.array(
          cfg_agent.warmup_action_range.ctrl, dtype=np.float32
      )
    if not self.fix_dstb:
      self.warmup_dstb_range = np.array(
          cfg_agent.warmup_action_range.dstb, dtype=np.float32
      )

  @property
  def has_backup(self):
    return False

  def dummy_dstb_sample(self, obs, append=None, latent=None, **kwargs):
    return torch.zeros(self.policy.action_dim_dstb).to(self.device), 0.

  def random_dstb_sample(self, obs, **kwargs):
    dstb_range = self.policy.action_range[1]
    dstb = self.rng.uniform(low=dstb_range[:, 0], high=dstb_range[:, 1])
    return torch.FloatTensor(dstb).to(self.device), 0.

  def sample_action(
      self, obs_all: torch.Tensor, dstb_sample_fn_all: List[Callable]
  ) -> Tuple[Dict[str, torch.FloatTensor], Dict[str, np.ndarray]]:

    # Gets controls.
    if self.fix_ctrl:
      with torch.no_grad():
        ctrl_all, _ = self.policy.ctrl.net.sample(
            obs_all.float().to(self.device), append=None, latent=None
        )
    else:
      if self.cnt_step < self.min_steps_b4_exploit:
        ctrl_all = self.rng.uniform(
            low=self.warmup_ctrl_range[:, 0],
            high=self.warmup_ctrl_range[:, 1],
            size=(self.n_envs, self.warmup_ctrl_range.shape[0]),
        )
        ctrl_all = torch.FloatTensor(ctrl_all)
      else:
        with torch.no_grad():
          ctrl_all, _ = self.policy.ctrl.net.sample(
              obs_all.float().to(self.device), append=None, latent=None
          )

    # Gets disturbances.
    if self.policy.dstb_use_ctrl:
      obs_dstb_all = torch.cat((obs_all, ctrl_all), dim=-1)
    else:
      obs_dstb_all = obs_all

    if self.fix_dstb:
      dstb_all, _ = self.policy.dstb.net.sample(
          obs_dstb_all.float().to(self.device), append=None, latent=None
      )
    else:
      if self.cnt_step < self.min_steps_b4_exploit:
        dstb_all = self.rng.uniform(
            low=self.warmup_dstb_range[:, 0],
            high=self.warmup_dstb_range[:, 1],
            size=(self.n_envs, self.warmup_dstb_range.shape[0]),
        )
        dstb_all = torch.FloatTensor(dstb_all)
      else:
        with torch.no_grad():
          dstb_all = torch.empty(
              size=(self.n_envs, self.policy.dstb.action_dim)
          ).to(self.device)
          for i, dstb_sample_fn in enumerate(dstb_sample_fn_all):
            dstb: torch.Tensor = dstb_sample_fn(
                obs_dstb_all[i], append=None, latent=None
            )[0]
            dstb_all[i] = dstb

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
    venv, agent_copy_list = self.init_learn(env)
    self.min_steps_b4_exploit = int(self.cfg_agent.min_steps_b4_exploit)
    if not (self.fix_ctrl or self.fix_dstb):
      self.cnt_dstb_opt = 0
      self.ctrl_opt_freq = int(self.cfg_agent.ctrl_opt_freq)

    if self.fix_ctrl:
      print("\nThis trains dstb to be a best response to the fixed ctrl.")
    if self.fix_dstb:
      print("\nThis trains ctrl to be a best response to the fixed dstb.")

    reset_kwargs_list = []  # Same initial states.
    for _ in range(self.cfg_agent.num_eval_traj):
      env.reset()
      reset_kwargs_list.append({"state": np.copy(env.state)})
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
    dstb_sample_fn_all = [  # `min_steps_b4_exploit` supersedes this.
        self.get_dstb_sample_fn(verbose=False) for _ in range(self.n_envs)
    ]

    print("\nMain loop")
    while self.cnt_step <= self.max_steps:
      # Selects action.
      action_all, action_all_np = self.sample_action(
          obs_all, dstb_sample_fn_all=dstb_sample_fn_all
      )

      # Interacts with the env.
      obs_nxt_all, r_all, done_all, info_all = venv.step(action_all_np)
      for env_idx, (done, info) in enumerate(zip(done_all, info_all)):
        # Stores the transition in memory.
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
          dstb_sample_fn_all[env_idx] = self.get_dstb_sample_fn(verbose=False)
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
        if self.fix_ctrl:
          update_ctrl = False
          update_dstb = True
        elif self.fix_dstb:
          update_ctrl = True
          update_dstb = False
        else:
          update_dstb = True
          if self.cnt_dstb_opt == (self.ctrl_opt_freq - 1):
            update_ctrl = True
            self.cnt_dstb_opt = 0
          else:
            update_ctrl = False
            self.cnt_dstb_opt += 1
        loss_critic_dict, loss_actor_dict = self.update(
            self.num_update_per_opt, update_ctrl=update_ctrl,
            update_dstb=update_dstb
        )
        self.train_record.append((loss_critic_dict, loss_actor_dict))
        self.cnt_opt += 1  # Counts number of optimization.
        if self.cfg_agent.use_wandb:
          # self.policy.critic_optimizer.state_dict()['param _groups'][0]['lr']
          log_dict = {
              "metrics/cnt_safety_violation": self.cnt_safety_violation,
              "metrics/cnt_num_episode": self.cnt_num_episode,
              "hyper_parameters/gamma": self.policy.critic.gamma,
              "loss/critic": loss_critic_dict['central'],
          }
          if 'dstb' in loss_actor_dict:
            log_dict["loss/dstb"] = loss_actor_dict['dstb'][0]
            log_dict["loss/ent_dstb"] = loss_actor_dict['dstb'][1]
            log_dict["loss/alpha_dstb"] = loss_actor_dict['dstb'][2]
            log_dict["hyper_parameters/alpha_dstb"] = self.policy.dstb.alpha
          if 'ctrl' in loss_actor_dict:
            log_dict["loss/ctrl"] = loss_actor_dict['ctrl'][0]
            log_dict["loss/ent_ctrl"] = loss_actor_dict['ctrl'][1]
            log_dict["loss/alpha_ctrl"] = loss_actor_dict['ctrl'][2]
            log_dict["hyper_parameters/alpha_ctrl"] = self.policy.ctrl.alpha
          wandb.log(log_dict, step=self.cnt_step, commit=False)

        # Checks after fixed number of gradient updates.
        if self.cnt_opt % self.check_opt_freq == 0 or self.first_update:
          # Updates the agent in the environment with the newest policy.
          env.agent.policy.update_policy(self.policy.ctrl.net)
          for agent in agent_copy_list:
            agent.policy.update_policy(self.policy.ctrl.net)
          venv.set_attr("agent", agent_copy_list, value_batch=True)

          # Has gameplays with all stored dstb checkpoints.
          reset_kwargs_list = []  # Same initial states.
          for _ in range(self.cfg_agent.num_eval_traj):
            env.reset()
            reset_kwargs_list.append({"state": np.copy(env.state)})
          self.check(
              env=env, venv=venv, reset_kwargs_list=reset_kwargs_list,
              action_kwargs=action_kwargs,
              rollout_step_callback=rollout_step_callback,
              rollout_episode_callback=rollout_episode_callback,
              visualize_callback=visualize_callback
          )

          # Resets anyway.
          obs_all = venv.reset()
          dstb_sample_fn_all = [
              self.get_dstb_sample_fn(verbose=True)
              for _ in range(self.n_envs)
          ]

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

    train_progress = np.array(self.train_progress)
    violation_record = np.array(self.violation_record)
    episode_record = np.array(self.episode_record)
    return (
        self.train_record, train_progress, violation_record, episode_record,
        self.pq_top_k
    )

  def save(
      self, venv: VecEnvBase, force_save: bool = False,
      reset_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      action_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      rollout_step_callback: Optional[Callable] = None,
      rollout_episode_callback: Optional[Callable] = None
  ) -> Dict:
    if self.fix_ctrl or self.fix_dstb:
      return self.save_best_response(
          venv=venv, force_save=force_save,
          reset_kwargs_list=reset_kwargs_list,
          action_kwargs_list=action_kwargs_list,
          rollout_step_callback=rollout_step_callback,
          rollout_episode_callback=rollout_episode_callback
      )
    else:
      return self.save_matrix_game(
          venv=venv, force_save=force_save,
          reset_kwargs_list=reset_kwargs_list,
          action_kwargs_list=action_kwargs_list,
          rollout_step_callback=rollout_step_callback,
          rollout_episode_callback=rollout_episode_callback
      )

  def save_matrix_game(
      self, venv: VecEnvBase, force_save: bool = False,
      reset_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      action_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      rollout_step_callback: Optional[Callable] = None,
      rollout_episode_callback: Optional[Callable] = None
  ) -> Dict:
    """
    This saving utils is called if both actors are trained. Within this
    function, we play games between all dstb checkpoints and the current ctrl.
    """
    save_ctrl = False
    save_dstb = False
    if force_save:
      save_ctrl = True
      save_dstb = True
      info = None
    else:
      num_eval_traj = self.cfg_agent.num_eval_traj
      eval_timeout = self.cfg_agent.eval_timeout
      rollout_end_criterion = self.cfg_agent.rollout_end_criterion
      history_weight = self.cfg_agent.history_weight

      steps = list(self.dstb_res_dict.keys())
      steps.append(self.cnt_step)

      metrics_weighted = []
      metrics_cur = []
      info_list = []
      for step in steps:
        adv_fn_list = []
        for _ in range(self.n_envs):
          if step == self.cnt_step:
            dstb_policy = copy.deepcopy(self.policy.dstb.net)
            dstb_policy.device = "cpu"
            dstb_policy.to("cpu")
            adv_fn_list.append(
                partial(
                    adv_dstb, dstb_policy=dstb_policy,
                    use_ctrl=self.policy.dstb_use_ctrl
                )
            )
          elif step == -1:
            adv_fn_list.append(
                partial(dummy_dstb, dim=self.policy.action_dim_dstb)
            )
          else:
            self.dstb_dup.restore(
                step=step, model_folder=self.module_folder_all[0],
                verbose=False
            )
            dstb_policy = copy.deepcopy(self.dstb_dup.net)
            dstb_policy.device = "cpu"
            dstb_policy.to("cpu")
            adv_fn_list.append(
                partial(
                    adv_dstb, dstb_policy=dstb_policy,
                    use_ctrl=self.policy.dstb_use_ctrl
                )
            )

        # Evaluates the policies.
        _, results, length = venv.simulate_trajectories_zs(
            num_trajectories=num_eval_traj, T_rollout=eval_timeout,
            end_criterion=rollout_end_criterion, adversary=adv_fn_list,
            reset_kwargs_list=reset_kwargs_list,
            action_kwargs_list=action_kwargs_list,
            rollout_step_callback=rollout_step_callback,
            rollout_episode_callback=rollout_episode_callback, use_tqdm=True
        )

        del adv_fn_list
        safe_rate = np.sum(results != -1) / num_eval_traj
        if rollout_end_criterion == "reach-avoid":
          success_rate = np.sum(results == 1) / num_eval_traj
          metric = success_rate
          _info = dict(
              safe_rate=safe_rate, ep_length=np.mean(length), metric=metric
          )
        else:
          metric = safe_rate
          _info = dict(ep_length=np.mean(length), metric=metric)

        if step in self.dstb_res_dict:
          _n_gameplays, _metric_avg = self.dstb_res_dict[step]
          n_gameplays = _n_gameplays + 1
          metric_avg = (history_weight*_n_gameplays*_metric_avg
                        + metric) / (history_weight*_n_gameplays + 1)
          self.dstb_res_dict[step] = (n_gameplays, metric_avg)
          metrics_weighted.append(metric_avg)
        else:
          metrics_weighted.append(metric)
        info_list.append(_info)
        metrics_cur.append(metric)

      indices_weighted = np.argsort(np.array(metrics_weighted))
      # Gets the step that is not -1 (we don't remove the dummy disturbance)
      # and has the highest metric. Notes that the indices are sorted in the
      # ascending order.
      for i in range(len(indices_weighted) - 1, -1, -1):
        step_rm_dstb = steps[indices_weighted[i]]
        if step_rm_dstb != -1:
          break
      info = info_list[indices_weighted[0]]
      metric_lowest = np.min(np.array(metrics_cur))
      print("  => Gameplays results:")
      print("     ", end='')
      for k, v in zip(steps, metrics_weighted):
        print(k, end=': ')
        print(v, end=" | ")
      print()

      # == Removes and saves checkpoints ==
      model_folder = self.module_folder_all[0]
      # Removes the worst dstb checkpoint.
      if len(self.dstb_res_dict) < self.save_top_k.dstb + 1:
        save_dstb = True
        self.dstb_res_dict[self.cnt_step] = (1, metrics_weighted[-1])
      elif step_rm_dstb != self.cnt_step:  # current dstb is not the worst.
        self.policy.dstb.remove(
            step=int(step_rm_dstb), model_folder=model_folder
        )
        self.dstb_res_dict.pop(step_rm_dstb)
        save_dstb = True
        self.dstb_res_dict[self.cnt_step] = (1, metrics_weighted[-1])

      # Removes the worst ctrl checkpoint.
      if self.pq_top_k.qsize() < self.save_top_k.ctrl:
        self.pq_top_k.put((metric_lowest, self.cnt_step))
        save_ctrl = True
      elif metric_lowest > self.pq_top_k.queue[0][0]:
        self.pq_top_k.put((metric_lowest, self.cnt_step))
        save_ctrl = True
        _, step_rm_ctrl = self.pq_top_k.get()
        self.policy.ctrl.remove(
            step=int(step_rm_ctrl), model_folder=model_folder
        )

    self._save(save_ctrl, save_dstb)
    return info

  def save_best_response(
      self, venv: VecEnvBase, force_save: bool = False,
      reset_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      action_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      rollout_step_callback: Optional[Callable] = None,
      rollout_episode_callback: Optional[Callable] = None
  ) -> Dict:
    """
    This saving utils is called if one of the actors is fixed. We only keep a
    single heap.
    """
    if force_save:
      save_ctrl = not self.fix_ctrl
      save_dstb = not self.fix_dstb
      info = None
    else:
      num_eval_traj: int = self.cfg_agent.num_eval_traj
      eval_timeout: int = self.cfg_agent.eval_timeout
      rollout_end_criterion: str = self.cfg_agent.rollout_end_criterion
      save_ctrl = False
      save_dstb = False

      # Gets the adversary.
      if self.fix_ctrl:
        dstb_policy = copy.deepcopy(self.policy.dstb.net)
      elif self.fix_dstb:
        dstb_policy = copy.deepcopy(self.policy_sample.dstb.net)
      dstb_policy.device = "cpu"
      dstb_policy.to(torch.device("cpu"))
      adv_fn_list = [
          partial(
              adv_dstb, dstb_policy=copy.deepcopy(dstb_policy),
              use_ctrl=self.policy.dstb_use_ctrl
          ) for _ in range(self.n_envs)
      ]

      # Evaluates the policies.
      _, results, length = venv.simulate_trajectories_zs(
          num_trajectories=num_eval_traj, T_rollout=eval_timeout,
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

      if self.fix_ctrl:
        # Dstb wants to maximize (1-metric).
        if self.pq_top_k.qsize() < self.save_top_k.ctrl:
          self.pq_top_k.put((1 - metric, self.cnt_step))
          save_dstb = True
        elif 1 - metric > self.pq_top_k.queue[0][0]:
          self.pq_top_k.put((1 - metric, self.cnt_step))
          save_dstb = True
          _, step_rm_dstb = self.pq_top_k.get()
          for module, module_folder in zip(
              self.module_all, self.module_folder_all
          ):
            module.remove(
                int(step_rm_dstb), module_folder, rm_dstb=True, rm_ctrl=False
            )
      elif self.fix_dstb:
        # Ctrl wants to maximize metric.
        if self.pq_top_k.qsize() < self.save_top_k.ctrl:
          self.pq_top_k.put((metric, self.cnt_step))
          save_ctrl = True
        elif metric > self.pq_top_k.queue[0][0]:
          self.pq_top_k.put((metric, self.cnt_step))
          save_ctrl = True
          _, step_rm_ctrl = self.pq_top_k.get()
          for module, module_folder in zip(
              self.module_all, self.module_folder_all
          ):
            module.remove(
                int(step_rm_ctrl), module_folder, rm_dstb=False, rm_ctrl=True
            )

    self._save(save_ctrl, save_dstb)
    return info

  # Overrides BaseTraining._save().
  def _save(self, save_ctrl: bool, save_dstb: bool):

    # Makes sure that we don't violate the fix_ctrl and fix_dstb flags.
    if self.fix_dstb:
      assert not save_dstb
    if self.fix_ctrl:
      assert not save_ctrl

    if save_ctrl or save_dstb:
      print("  => priority queue:", self.pq_top_k.queue)

    model_folder = self.module_folder_all[0]
    self.policy.critic.save(
        step=self.cnt_step, model_folder=model_folder, max_model=self.max_model
    )

    if save_ctrl:
      self.policy.ctrl.save(
          step=self.cnt_step, model_folder=model_folder,
          max_model=self.max_model
      )
    if save_dstb:
      self.policy.dstb.save(
          step=self.cnt_step, model_folder=model_folder,
          max_model=self.max_model
      )

  def get_dstb_sample_fn(
      self, verbose: bool = False, dstb_sample_type: Optional[str] = None
  ) -> Callable[[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],
                Tuple[torch.Tensor, Any]]:
    """
    Gets a disturbance sampler. This can be: (1) sampling from the checkpoints,
    (2) picking the one has the highest failure rate in the last checking
    round, (3) using the current one, (4) no disturbance, or (5) sampling
    randomly from the disturbance set.

    Args:
        verbose (bool, optional): Defaults to False.
        dstb_sample_type (Optional[str], optional): Defaults to None.

    Returns:
        Callable: a function to sample disturbances from.
    """
    if self.fix_ctrl or self.fix_dstb:
      step_to_restore = self.cnt_step  # always uses the current dstb.
    else:
      if dstb_sample_type is None:
        dstb_sample_type: str = self.cfg_agent.dstb_sample_type

      if dstb_sample_type == "recent":
        step_to_restore = self.cnt_step
      else:
        if dstb_sample_type == "strongest":
          lowest_metric = float("inf")
          for k, v in self.dstb_res_dict.items():
            if v[1] < lowest_metric:
              step_to_restore = k
              lowest_metric = v[1]
        elif dstb_sample_type == "softmax":
          dstb_sample_cur_weight = float(self.cfg_agent.dstb_sample_cur_weight)
          dstb_sample_sm_weight = float(
              getattr(self.cfg_agent, "dstb_sample_sm_weight", 5.)
          )
          steps = []
          metrics = []
          for k, v in self.dstb_res_dict.items():
            steps.append(k)
            metrics.append(1 - v[1])
          steps.append(self.cnt_step)
          metrics = np.array(metrics)
          e_x: np.ndarray = np.exp(
              dstb_sample_sm_weight * (metrics - np.max(metrics))
          )
          probs = (e_x / e_x.sum()) * (1-dstb_sample_cur_weight)
          probs = np.append(probs, dstb_sample_cur_weight)
          step_to_restore = self.rng.choice(steps, p=probs)
        else:
          raise ValueError(
              f"dstb_sample_type ({dstb_sample_type}) is not supported!"
          )

    if step_to_restore == -1:
      dstb_sample_fn = self.dummy_dstb_sample
      if verbose:
        print("  => Uses dummy disturbance sampler.")
    elif step_to_restore == self.cnt_step:
      dstb_sample_fn = self.policy.dstb.net.sample
      if verbose:
        print("  => Uses the current disturbance sampler.")
    else:
      self.dstb_dup.restore(
          step=step_to_restore, model_folder=self.module_folder_all[0],
          verbose=False
      )
      dstb_sample_fn = self.dstb_dup.net.sample
      if verbose:
        print(f"  => Uses disturbance sampler from {step_to_restore}.")

    return dstb_sample_fn
