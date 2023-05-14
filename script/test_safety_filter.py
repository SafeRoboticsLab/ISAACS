# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import sys
import copy
from types import SimpleNamespace
import numpy as np
import argparse
from shutil import copyfile
import pickle
import time
import jax
from omegaconf import OmegaConf
import warnings

from simulators import (
    RaceCarDstb5DEnv, PrintLogger, Bicycle5D, NeuralNetworkControlSystem,
    ILQRSpline, VecEnvBase, save_obj
)
from agent import SACAdv
from utils.eval import get_disturbance
from utils.safety_filter import SafetyFilter
from utils.safety_monitor import ValueMonitor, RolloutMonitor, RobustFRSMonitor

jax.config.update('jax_platform_name', 'cpu')


def rollout_episode_callback(env, *args, **kwargs):
  env.agent.policy.clear_cache()


def main(config_file: str):
  cfg = OmegaConf.load(config_file)
  out_folder = cfg.main.out_folder
  os.makedirs(out_folder, exist_ok=True)
  copyfile(config_file, os.path.join(out_folder, 'config.yaml'))
  sys.stderr = PrintLogger(os.path.join(out_folder, 'log.txt'))
  sys.stdout = PrintLogger(os.path.join(out_folder, 'log.txt'))

  print("Constructs testing environment...")
  env = RaceCarDstb5DEnv(cfg.environment, cfg.agent, cfg.cost)
  print("Constructs imaginary environment...")
  cfg_env_im = copy.deepcopy(cfg.environment)
  cfg_cost_im = copy.deepcopy(cfg.cost)
  cfg_env_im.track_len = (
      cfg.environment.track_len + cfg.agent.v_max * cfg.agent.dt
  )  # for rollout/FRS monitor
  cfg_cost_im.buffer = cfg.main.safety.filter.buffer
  env_im = RaceCarDstb5DEnv(cfg_env_im, cfg.agent, cfg_cost_im)
  rng = np.random.default_rng(seed=cfg.environment.seed)

  print("Loading reset kwargs from", cfg.main.reset_kwargs_file)
  with open(cfg.main.reset_kwargs_file, "rb") as f:
    _reset_kwargs_list = pickle.load(f)
  if isinstance(cfg.main.num_trajs, int):
    num_trajs = cfg.main.num_trajs
    if len(_reset_kwargs_list) >= num_trajs:
      reset_kwargs_list = _reset_kwargs_list[:num_trajs]
    else:
      raise ValueError(
          f"Only {len(_reset_kwargs_list)} reset kwargs are available."
      )
  else:
    reset_kwargs_list = [_reset_kwargs_list[x] for x in cfg.main.num_trajs]
    num_trajs = len(reset_kwargs_list)

  # region: policy
  cfg.train.device = cfg.main.device
  model_folder = os.path.join(cfg.main.safety.root, "model")
  sac_adv = SACAdv(cfg.train, cfg.arch, rng)
  sac_adv.build_network(verbose=False)

  print("\n== Sets up disturbance functions ==")
  real_dstb_fn_list = get_disturbance(
      dstb_type=cfg.main.real_dstb.type,
      num_envs=cfg.main.num_envs,
      key='real disturbance',
      odp_folder=cfg.main.real_dstb.odp_folder,
      dstb_actor=sac_adv.dstb,
      dstb_step=cfg.main.real_dstb.step,
      model_folder=model_folder,
      action_dim_dstb=sac_adv.dstb.action_dim,
      rng=rng,
      dstb_range=np.asarray(cfg.main.real_dstb.range),
  )
  if cfg.main.imag_dstb.type == cfg.main.real_dstb.type:
    print("Uses imaginary disturbance the same as the real one.")
    imag_dstb_fn_list = real_dstb_fn_list
  else:
    imag_dstb_fn_list = get_disturbance(
        dstb_type=cfg.main.imag_dstb.type,
        num_envs=cfg.main.num_envs,
        key='imaginary disturbance',
        odp_folder=cfg.main.imag_dstb.odp_folder,
        dstb_actor=sac_adv.dstb,
        dstb_step=cfg.main.imag_dstb.step,
        model_folder=model_folder,
        action_dim_dstb=sac_adv.dstb.action_dim,
        rng=rng,
        dstb_range=np.asarray(cfg.main.imag_dstb.range),
    )

  print("\n== Sets up policies ==")
  # task policy
  ilqr_policy = ILQRSpline(
      'ego',
      cfg.ilqr,
      dyn=Bicycle5D(
          cfg.agent, np.asarray(cfg.agent.action_range.ctrl, dtype=float)
      ),
      cost=copy.deepcopy(env.cost),
      track=copy.deepcopy(env.track),
  )

  # safety policy
  print("Constructs safety policy.")
  sac_adv.ctrl.restore(cfg.main.safety.ctrl_step, model_folder, verbose=True)
  safety_policy = NeuralNetworkControlSystem(
      id='ego', actor=sac_adv.ctrl.net,
      cfg=SimpleNamespace(device=sac_adv.device)
  )

  # safety filter
  venv = VecEnvBase([copy.deepcopy(env) for _ in range(cfg.main.num_envs)])
  venv.seed(cfg.environment.seed)
  agent_copy_list = [
      copy.deepcopy(env.agent) for _ in range(cfg.main.num_envs)
  ]

  # endregion

  # region: Evaluates
  print("\n== Evaluation starts ==")
  # Tests safety policy.
  if cfg.main.test_safety:
    start_time = time.time()
    for agent in agent_copy_list:
      agent.policy = safety_policy
    venv.set_attr("agent", agent_copy_list, value_batch=True)

    trajs, results, _ = venv.simulate_trajectories_zs(
        num_trajectories=num_trajs, T_rollout=cfg.main.real_timeout,
        end_criterion='failure', adversary=real_dstb_fn_list,
        reset_kwargs_list=reset_kwargs_list, use_tqdm=True
    )
    t_exec = time.time() - start_time
    f_rate = np.mean(results == -1)
    print(
        f'\nSafety policy has failure rate: {f_rate:.3f} (exec: {t_exec:.1f}s)'
    )
    print(results)
    results_dict_safety = {'trajs': trajs, 'results': results}
    save_obj(results_dict_safety, os.path.join(out_folder, 'results_safety'))

  # Tests value-based filter.
  if cfg.main.test_value:
    results_dict_value = {}
    sac_adv.critic.restore(
        cfg.main.safety.critic_step, model_folder, verbose=True
    )
    for thr in cfg.main.safety.filter.value_thr_list:
      print(f"\nConstructs value-based filter with thr {thr}.")
      for agent, imag_adv in zip(agent_copy_list, imag_dstb_fn_list):
        value_monitor = ValueMonitor(
            adversary=imag_adv,  # memory-heavy when using odp -> do not copy.
            critic=copy.deepcopy(sac_adv.critic),
            value_threshold=thr,
            value_to_be_max=False  # Our critic encodes the cost-to-go.
        )
        agent.policy = SafetyFilter(
            base_policy=copy.deepcopy(ilqr_policy),
            safety_policy=copy.deepcopy(safety_policy), monitor=value_monitor
        )
      venv.set_attr("agent", agent_copy_list, value_batch=True)

      start_time = time.time()
      trajs, results, _, infos = venv.simulate_trajectories_zs(
          num_trajectories=num_trajs, T_rollout=cfg.main.real_timeout,
          end_criterion='failure', adversary=real_dstb_fn_list,
          reset_kwargs_list=reset_kwargs_list, return_info=True, use_tqdm=True
      )
      t_exec = time.time() - start_time
      f_rate = np.mean(results == -1)
      print(f'{thr:.2f} thr: failure rate: {f_rate:.3f} (exec: {t_exec:.1f}s)')
      shield_rate = [
          np.sum(info['shield_ind']) / (len(traj) - 1)
          for traj, info in zip(trajs, infos)
      ]
      results_dict_value[f'{thr}'] = {
          'trajs': trajs,
          'results': results,
          'shield_ind': [info['shield_ind'] for info in infos],
          'shield_rate': shield_rate,
          'infos': infos
      }
      print(f"sf rate: {np.mean(np.array(shield_rate))}")
      print(results)
      with np.printoptions(precision=3, suppress=False):
        print(np.array(shield_rate))
      save_obj(results_dict_value, os.path.join(out_folder, 'results_value'))

  # Tests rollout-based filter.
  if cfg.main.test_rollout:
    results_dict_rollout = {}
    env_dup = copy.deepcopy(env_im)
    env_dup.agent.policy = safety_policy
    for steps in cfg.main.safety.filter.imag_timeout_list:
      print(f"\nConstructs rollout-based filter with {steps} steps.")
      for agent, imag_adv in zip(agent_copy_list, imag_dstb_fn_list):
        rollout_monitor = RolloutMonitor(
            adversary=imag_adv,  # memory-heavy when using odp -> do not copy.
            env=copy.deepcopy(env_dup),
            imag_end_criterion='failure',
            imag_steps=steps,
        )
        agent.policy = SafetyFilter(
            base_policy=ilqr_policy, safety_policy=safety_policy,
            monitor=rollout_monitor
        )
      venv.set_attr("agent", agent_copy_list, value_batch=True)

      start_time = time.time()
      trajs, results, _, infos = venv.simulate_trajectories_zs(
          num_trajectories=num_trajs, T_rollout=cfg.main.real_timeout,
          end_criterion='failure', adversary=real_dstb_fn_list,
          reset_kwargs_list=reset_kwargs_list, return_info=True, use_tqdm=True
      )
      t_exec = time.time() - start_time
      f_rate = np.mean(results == -1)
      print(f'{steps} steps: failure rate: {f_rate:.3f} (exec: {t_exec:.1f}s)')
      shield_rate = [
          np.sum(info['shield_ind']) / (len(traj) - 1)
          for traj, info in zip(trajs, infos)
      ]
      results_dict_rollout[f'{steps:d}'] = {
          'trajs': trajs,
          'results': results,
          'shield_ind': [info['shield_ind'] for info in infos],
          'shield_rate': shield_rate,
          'infos': infos
      }
      print(f"sf rate: {np.mean(np.array(shield_rate))}")
      print(results)
      with np.printoptions(precision=3, suppress=False):
        print(np.array(shield_rate))
      save_obj(
          results_dict_rollout, os.path.join(out_folder, 'results_rollout')
      )

  # Tests FRS-based filter.
  if cfg.main.test_frs:
    if cfg.main.imag_dstb.type != 'dummy':
      if cfg.main.safety.filter.override_dummy:
        warnings.warn("\nSuggest using dummy for FRS but overridden!")
        frs_dstb_fn_list = imag_dstb_fn_list
      else:
        warnings.warn("\nUsing dummy for FRS!")
        frs_dstb_fn_list = get_disturbance(
            dstb_type='dummy', num_envs=cfg.main.num_envs,
            key='dummy disturbance', odp_folder=cfg.main.imag_dstb.odp_folder,
            dstb_actor=sac_adv.dstb, dstb_step=cfg.main.imag_dstb.step,
            model_folder=model_folder, action_dim_dstb=sac_adv.dstb.action_dim,
            rng=rng, dstb_range=np.asarray(cfg.main.imag_dstb.range)
        )
    else:
      frs_dstb_fn_list = imag_dstb_fn_list

    env_dup = copy.deepcopy(env_im)
    env_dup.agent.policy = safety_policy
    cfg_track_policy = copy.deepcopy(cfg.ilqr)
    cfg.ref_cost.state_box_limit = cfg.agent.state_box_limit
    cfg.ref_cost.wheelbase = cfg.agent.wheelbase
    cfg.ref_cost.track_width_left = cfg.environment.track_width_left
    cfg.ref_cost.track_width_right = cfg.environment.track_width_right
    cfg.ref_cost.obs_spec = cfg.environment.obs_spec

    results_dict_rollout = {}
    for steps in cfg.main.safety.filter.imag_timeout_list:
      print(f"\nConstructs FRS-based filter with {steps} steps.")
      cfg_track_policy.plan_horizon = steps + 2
      track_policy = ILQRSpline(
          id='track', cfg=cfg_track_policy, dyn=Bicycle5D(
              cfg.agent, np.asarray(cfg.agent.action_range.ctrl, dtype=float)
          ), cost=None, track=env_dup.track
      )
      for agent, frs_adv in zip(agent_copy_list, frs_dstb_fn_list):
        robust_monitor = RobustFRSMonitor(
            adversary=frs_adv,
            env=copy.deepcopy(env_dup),
            imag_end_criterion='failure',
            imag_steps=steps,
            track_policy=copy.deepcopy(track_policy),
            cfg_ref_cost=cfg.ref_cost,
            dstb_bound=np.array(cfg.main.safety.dstb_bound),
            ctrl_bound=np.array(cfg.agent.action_range.ctrl),
            buffer=cfg.main.safety.filter.buffer,
        )
        agent.policy = SafetyFilter(
            base_policy=copy.deepcopy(ilqr_policy),
            safety_policy=copy.deepcopy(safety_policy), monitor=robust_monitor,
            override_type='linear', concise=cfg.main.concise
        )
      venv.set_attr("agent", agent_copy_list, value_batch=True)

      start_time = time.time()
      trajs, results, _, infos = venv.simulate_trajectories_zs(
          num_trajectories=num_trajs, T_rollout=cfg.main.real_timeout,
          end_criterion='failure', adversary=real_dstb_fn_list,
          reset_kwargs_list=reset_kwargs_list, return_info=True, use_tqdm=True,
          rollout_episode_callback=rollout_episode_callback
      )
      t_exec = time.time() - start_time
      f_rate = np.mean(results == -1)
      print(f'{steps} steps: failure rate: {f_rate:.3f} (exec: {t_exec:.1f}s)')
      shield_reason_all = []
      shield_rate_wo_ctrl = []
      shield_rate = []
      for traj, info in zip(trajs, infos):
        shield_rate.append(np.sum(info['shield_ind']) / (len(traj) - 1))
        shield_reason = []
        cnt_shield_wo_ctrl = 0
        for t in range(len(traj) - 1):
          solver_info = info['plan_hist'][t]
          if solver_info['shield']:
            shield_reason.append(solver_info['raised_reason'])
            if solver_info['raised_reason'] != 'ctrl':
              cnt_shield_wo_ctrl += 1
          else:
            shield_reason.append(None)
        shield_reason_all.append(shield_reason)
        shield_rate_wo_ctrl.append(cnt_shield_wo_ctrl / (len(traj) - 1))
      results_dict_rollout[f'{steps:d}'] = {
          'trajs': trajs,
          'results': results,
          'shield_ind': [info['shield_ind'] for info in infos],
          'shield_reason_all': shield_reason_all,
          'shield_rate': shield_rate,
          'shield_rate_wo_ctrl': shield_rate_wo_ctrl,
          'infos': infos
      }
      with np.printoptions(precision=2, suppress=False):
        print(f"sf rate w/ ctrl: {np.mean(np.array(shield_rate))}")
        print(f"sf rate w/o ctrl: {np.mean(np.array(shield_rate_wo_ctrl))}")
        # for traj, result in zip(trajs, results):
        #   print(traj[0], result)
        print(results)
        print(np.array(shield_rate))
        print(np.array(shield_rate_wo_ctrl))
      save_obj(results_dict_rollout, os.path.join(out_folder, 'results_frs'))
  # endregion


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-cf", "--config_file", help="config file path", type=str,
      default=os.path.join("config", "safety_filter.yaml")
  )
  args = parser.parse_args()
  main(args.config_file)
