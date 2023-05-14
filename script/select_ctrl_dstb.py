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
from tqdm import tqdm
import jax
from omegaconf import OmegaConf
from functools import partial

from simulators import (
    RaceCarDstb5DEnv, VecEnvBase, Bicycle5D, NeuralNetworkControlSystem,
    ILQRSpline, PrintLogger, save_obj
)
from agent import SACAdv
from utils.dstb import adv_dstb, dummy_dstb

jax.config.update('jax_platform_name', 'cpu')


def main(config_file: str):
  cfg = OmegaConf.load(config_file)
  out_folder = cfg.main.out_folder
  isaacs_folder = cfg.main.isaacs_folder
  os.makedirs(out_folder, exist_ok=True)
  copyfile(config_file, os.path.join(out_folder, 'config.yaml'))
  sys.stderr = PrintLogger(os.path.join(out_folder, 'log.txt'))
  sys.stdout = PrintLogger(os.path.join(out_folder, 'log.txt'))

  env = RaceCarDstb5DEnv(cfg.environment, cfg.agent, cfg.cost)
  rng = np.random.default_rng(seed=cfg.environment.seed)

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

  # ISAACS.
  cfg.train.device = cfg.main.device
  sac_adv = SACAdv(cfg.train, cfg.arch, rng)
  print("Using device:", sac_adv.device)
  sac_adv.build_network(verbose=False)
  ctrl_candidates = []
  dstb_candidates = []
  model_folder = os.path.join(isaacs_folder, "model")
  ctrl_folder = os.path.join(model_folder, "ctrl")
  dstb_folder = os.path.join(model_folder, "dstb")

  for x in sorted(os.listdir(ctrl_folder), key=lambda x: int(x[5:-4])):
    if x.endswith(".pth"):
      ctrl_candidates.append(x[5:-4])
  for x in sorted(os.listdir(dstb_folder), key=lambda x: int(x[5:-4])):
    if x.endswith(".pth"):
      dstb_candidates.append(x[5:-4])
  print("CTRL candidates", ctrl_candidates)
  print("DSTB candidates", dstb_candidates)
  n_ctrls = len(ctrl_candidates)
  n_dstbs = len(dstb_candidates)
  safety_results_array = np.empty(shape=(n_dstbs, n_ctrls))
  dstb_results_list = np.empty(shape=(n_dstbs))

  # endregion

  # region: Sets initial states.
  env.agent.policy = ilqr_policy
  venv = VecEnvBase([copy.deepcopy(env) for _ in range(cfg.main.num_envs)],
                    device=cfg.main.venv_device)
  venv.seed(cfg.environment.seed)

  n_trajs = cfg.main.num_eval_traj
  if cfg.main.reset_kwargs_file is not None:
    print("Loading reset kwargs from", cfg.main.reset_kwargs_file)
    with open(cfg.main.reset_kwargs_file, "rb") as f:
      reset_kwargs_list = pickle.load(f)
    if len(reset_kwargs_list) >= n_trajs:
      reset_kwargs_list = reset_kwargs_list[:n_trajs]
    else:
      raise ValueError(
          f"Only {len(reset_kwargs_list)} reset kwargs are available."
      )
  else:
    reset_kwargs_list = []
    # Collects initial states that ILQR policy keeps safety when there is no
    # disturbance --> Testing false safe rate (w.r.t. trained disturbance).
    cnt_trials = 0
    dummy_adv_fn_list = [
        partial(dummy_dstb, dim=5) for _ in range(cfg.main.num_envs)
    ]
    while len(reset_kwargs_list) < n_trajs:
      trajs_to_collect = n_trajs - len(reset_kwargs_list)
      if cfg.main.test_false_neg:
        trajs, results, _ = venv.simulate_trajectories_zs(
            num_trajectories=trajs_to_collect, T_rollout=cfg.main.real_timeout,
            end_criterion='failure', adversary=dummy_adv_fn_list,
            use_tqdm=False
        )
        for traj, result in zip(trajs, results):
          cnt_trials += 1
          if result != -1:
            reset_kwargs_list.append({"state": traj[0]})
          if len(reset_kwargs_list) == n_trajs:
            break
        print(f"{cnt_trials} trials -> {len(reset_kwargs_list)} states.")
      else:
        venv.reset()
        for state in venv.get_attr('state'):
          reset_kwargs_list.append({"state": state})
          if len(reset_kwargs_list) == n_trajs:
            break

    if cfg.main.test_false_neg:
      save_obj(reset_kwargs_list, os.path.join(out_folder, "init_states"))
    else:
      save_obj(reset_kwargs_list, os.path.join(out_folder, "init_states_unif"))

  # endregion

  # region: Evaluates
  agent_copy_list = [
      copy.deepcopy(env.agent) for _ in range(cfg.main.num_envs)
  ]
  for dstb_idx, dstb_step in tqdm(
      enumerate(dstb_candidates), total=n_dstbs, desc="Matrix Game"
  ):
    # Loads disturbance checkpoints.
    sac_adv.dstb.restore(dstb_step, model_folder, verbose=False)
    adv_fn_list = []
    for _ in range(cfg.main.num_envs):
      dstb_policy = copy.deepcopy(sac_adv.dstb.net)
      dstb_policy.device = "cpu"
      dstb_policy.to("cpu")
      adv_fn_list.append(
          partial(
              adv_dstb, dstb_policy=dstb_policy, use_ctrl=sac_adv.dstb_use_ctrl
          )
      )

    # Evaluates task policy vs. dstb policy.
    for agent in agent_copy_list:
      agent.policy = ilqr_policy
    venv.set_attr("agent", agent_copy_list, value_batch=True)
    _, results1, _ = venv.simulate_trajectories_zs(
        num_trajectories=n_trajs, T_rollout=cfg.main.real_timeout,
        end_criterion='failure', adversary=adv_fn_list,
        reset_kwargs_list=reset_kwargs_list
    )
    dstb_results_list[dstb_idx] = np.mean(results1 == -1)

    # Evaluates safety policies vs. dstb policy.
    for ctrl_idx, ctrl_step in tqdm(
        enumerate(ctrl_candidates), total=n_ctrls, desc="vs. controls",
        leave=False
    ):
      sac_adv.ctrl.restore(ctrl_step, model_folder, verbose=False)
      for agent in agent_copy_list:
        agent.policy = NeuralNetworkControlSystem(
            id='ego', actor=sac_adv.ctrl.net,
            cfg=SimpleNamespace(device=sac_adv.device)
        )
      venv.set_attr("agent", agent_copy_list, value_batch=True)

      _, results2, _ = venv.simulate_trajectories_zs(
          num_trajectories=n_trajs, T_rollout=cfg.main.real_timeout,
          end_criterion='failure', adversary=adv_fn_list,
          reset_kwargs_list=reset_kwargs_list
      )
      safety_results_array[dstb_idx, ctrl_idx] = np.mean(results2 == -1)
  # endregion

  # region: Reports
  for dstb_idx, dstb_step in enumerate(dstb_candidates):
    print(f"Dstb ({dstb_step}):")
    print(f"  - task policy failure rate: {dstb_results_list[dstb_idx]}")
    print("  - safety policy failure rate: ", end='')
    for ctrl_idx in range(n_ctrls):
      print(f"{safety_results_array[dstb_idx, ctrl_idx]:.2f}", end='')
      if ctrl_idx != n_ctrls - 1:
        print(" | ", end='')
      else:
        print()

  eval_results = {
      "safety_results_array": safety_results_array,
      "dstb_results_list": dstb_results_list
  }
  idx1 = np.argmax(dstb_results_list)
  print(f"The worst dstb against task policy is {dstb_candidates[idx1]}.")
  avg = np.mean(safety_results_array, axis=1)
  idx2 = np.argmax(avg)
  idx3 = np.argmin(safety_results_array[idx2])
  print(
      f"The worst dstb against safety policies is {dstb_candidates[idx2]},"
      f" and the best safety policy against it is {ctrl_candidates[idx3]}."
  )

  save_obj(eval_results, os.path.join(out_folder, "eval_results"))
  # endregion


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-cf", "--config_file", help="config file path", type=str,
      default=os.path.join("config", "select.yaml")
  )
  args = parser.parse_args()
  main(args.config_file)
