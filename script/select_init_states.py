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
import jax
from omegaconf import OmegaConf

from simulators import (
    RaceCarDstb5DEnv, VecEnvBase, Bicycle5D, ILQRSpline, PrintLogger, save_obj
)
from optimized_dp.odp_policy import ODPPolicy

jax.config.update('jax_platform_name', 'cpu')


def main(config_file: str):
  cfg = OmegaConf.load(config_file)
  out_folder = cfg.main.out_folder
  os.makedirs(out_folder, exist_ok=True)
  copyfile(config_file, os.path.join(out_folder, 'config.yaml'))
  sys.stderr = PrintLogger(os.path.join(out_folder, 'log.txt'))
  sys.stdout = PrintLogger(os.path.join(out_folder, 'log.txt'))

  env = RaceCarDstb5DEnv(cfg.environment, cfg.agent, cfg.cost)

  # region: policy
  # Task policy: ILQR
  ilqr_policy = ILQRSpline(
      'ego',
      cfg.ilqr,
      dyn=Bicycle5D(
          cfg.agent, np.asarray(cfg.agent.action_range.ctrl, dtype=float)
      ),
      cost=env.cost,
      track=env.track,
  )

  # optimized DP
  odp_folder = cfg.main.odp_folder
  copyfile(
      os.path.join(odp_folder, 'config.yaml'),
      os.path.join(out_folder, 'odp.yaml')
  )
  with open(os.path.join(odp_folder, "results_lw.pkl"), "rb") as f:
    result_dict = pickle.load(f)
  cfg_odp_policy = SimpleNamespace(device='cpu', interp_method='linear')
  odp_policy = ODPPolicy(
      id='dstb', car=result_dict['my_car'], grid=result_dict['grid'],
      odp_values=result_dict['values'], cfg=cfg_odp_policy
  )
  # grid: Grid = copy.deepcopy(result_dict['grid'])
  # car: BicycleDstb5D = copy.deepcopy(result_dict['my_car'])
  del result_dict
  print("Finish loading ODP policy.")
  # endregion

  # region: Sets initial states.
  env.agent.policy = ilqr_policy
  venv = VecEnvBase([copy.deepcopy(env) for _ in range(cfg.main.num_envs)],
                    device=cfg.main.venv_device)
  venv.seed(cfg.environment.seed)

  # Collects initial states that ILQR fails to the optimal disturbance, but
  # optimal control can keep safety.
  n_trajs = cfg.main.num_eval_traj
  reset_kwargs_list = []
  cnt_trials = 0
  adv_fn_list = [odp_policy.get_opt_dstb for _ in range(cfg.main.num_envs)]
  agent_copy_list = [
      copy.deepcopy(env.agent) for _ in range(cfg.main.num_envs)
  ]

  while len(reset_kwargs_list) < n_trajs:
    trajs_to_collect = max(n_trajs - len(reset_kwargs_list), cfg.main.num_envs)
    # Collects initial states that ILQR fails to the optimal disturbance.
    for agent in agent_copy_list:
      agent.policy = ilqr_policy
    venv.set_attr("agent", agent_copy_list, value_batch=True)
    trajs1, results1, _ = venv.simulate_trajectories_zs(
        num_trajectories=trajs_to_collect, T_rollout=cfg.main.real_timeout,
        end_criterion='failure', adversary=adv_fn_list, use_tqdm=True
    )
    reset_kwargs_candidates = []
    for traj, result in zip(trajs1, results1):
      cnt_trials += 1
      if result == -1:
        reset_kwargs_candidates.append({"state": traj[0]})

    # Checks if the optimal control can keep safety among those candidates.
    for agent in agent_copy_list:
      agent.policy = odp_policy
    venv.set_attr("agent", agent_copy_list, value_batch=True)
    _, results2, _ = venv.simulate_trajectories_zs(
        num_trajectories=len(reset_kwargs_candidates),
        T_rollout=cfg.main.real_timeout, end_criterion='failure',
        adversary=adv_fn_list, reset_kwargs_list=reset_kwargs_candidates,
        use_tqdm=True
    )
    for result, reset_kwargs in zip(results2, reset_kwargs_candidates):
      if result != -1:
        reset_kwargs_list.append(reset_kwargs)
      if len(reset_kwargs_list) == n_trajs:
        break
    print(f"{cnt_trials} trials -> {len(reset_kwargs_list)} states.")

  save_obj(
      reset_kwargs_list, os.path.join(out_folder, f"init_states_{n_trajs}")
  )

  # endregion


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-cf", "--config_file", help="config file path", type=str,
      default=os.path.join("config", "init_states_safety_filter.yaml")
  )
  args = parser.parse_args()
  main(args.config_file)
