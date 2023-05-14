# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from typing import List
import os
import sys
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
import wandb
import argparse
from functools import partial
from shutil import copyfile
import jax
from omegaconf import OmegaConf

from agent.naive_rl_dstb import NaiveRLDisturbance
from agent.sac_br import SACBestResponse
from simulators import RaceCarDstb5DEnv, PrintLogger, save_obj
from utils.dstb import adv_dstb
from utils.visualization import plot_traj, get_values, get_trajectories_zs

jax.config.update('jax_platform_name', 'cpu')


# region: local functions
def visualize(
    env: RaceCarDstb5DEnv, policy: SACBestResponse, fig_path: str,
    vel_list: List[float], yaw_list: List[float], num_pts: int = 5,
    T_rollout: int = 150, end_criterion: str = "failure",
    subfigsz_x: float = 4., subfigsz_y: float = 4., nx: int = 100,
    ny: int = 100, batch_size: int = 512, cmap: str = 'seismic',
    vmin: float = -.25, vmax: float = .25, alpha: float = 0.5,
    fontsize: int = 16, vel_scatter: bool = False, markersz: int = 40
):
  n_row = len(yaw_list)
  n_col = len(vel_list)
  figsize = (subfigsz_x * n_col, subfigsz_y * n_row)
  fig, axes = plt.subplots(
      n_row, n_col, figsize=figsize, sharex=True, sharey=True
  )
  vmin_label = vmin
  vmax_label = vmax
  vmean_label = 0

  xs, ys = env.get_samples(nx, ny)
  dstb_policy = policy.dstb.net
  dstb_policy.device = "cpu"
  dstb_policy.to("cpu")
  trajectories, results = get_trajectories_zs(
      env=env, adversary=partial(
          adv_dstb, dstb_policy=dstb_policy, use_ctrl=policy.dstb_use_ctrl
      ), vel_list=vel_list, yaw_list=yaw_list, num_pts=num_pts,
      T_rollout=T_rollout, end_criterion=end_criterion
  )

  for i, vel in enumerate(vel_list):
    axes[0][i].set_title(f"Vel: {vel:.2f}", fontsize=fontsize)
    for j, yaw in enumerate(yaw_list):
      ax = axes[j][i]
      if i == 0:
        ax.set_ylabel(f"Yaw: {yaw/np.pi*180:.0f}", fontsize=fontsize)
      # plots value function
      values = get_values(
          env, policy.value, xs, ys, vel, yaw, batch_size=batch_size,
          fail_value=vmax
      )
      im = ax.imshow(
          values.T, interpolation='none', extent=env.visual_extent,
          origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, zorder=-1,
          alpha=alpha
      )

      # Plots trajectories
      for k in range(num_pts):
        idx = int(k * len(yaw_list) * len(vel_list) + i * len(yaw_list) + j)
        trajectory = trajectories[idx]
        result = results[idx]
        plot_traj(
            ax, trajectory, result, c='g', lw=2., vel_scatter=vel_scatter,
            zorder=1, s=markersz
        )

      env.track.plot_track(ax, c='k')
      env.render_obs(ax=ax, c='r')
      ax.axis(env.visual_extent)
      ax.set_xticks(np.around(env.visual_bounds[0], 1))
      ax.set_yticks(np.around(env.visual_bounds[1], 1))
      ax.tick_params(axis='both', which='major', labelsize=fontsize - 4)

  # one color bar
  fig.subplots_adjust(
      left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.08, hspace=0.01
  )
  cbar_ax = fig.add_axes([0.96, 0.05, 0.02, 0.85])
  cbar = fig.colorbar(im, cax=cbar_ax, ax=ax, ticks=[vmin, 0, vmax])
  v_ticklabels = np.around(np.array([vmin_label, vmean_label, vmax_label]), 2)
  cbar.ax.set_yticklabels(labels=v_ticklabels, fontsize=fontsize - 4)
  fig.savefig(fig_path, dpi=400)
  plt.close('all')


# endregion


def main(config_file):
  # Loads config.
  cfg = OmegaConf.load(config_file)
  cfg.train.device = cfg.solver.device

  os.makedirs(cfg.solver.out_folder, exist_ok=True)
  copyfile(config_file, os.path.join(cfg.solver.out_folder, 'config.yaml'))
  log_path = os.path.join(cfg.solver.out_folder, 'log.txt')
  if os.path.exists(log_path):
    os.remove(log_path)
  sys.stdout = PrintLogger(log_path)
  sys.stderr = PrintLogger(log_path)

  if cfg.solver.use_wandb:
    wandb.init(
        entity='safe-princeton', project=cfg.solver.project_name,
        name=cfg.solver.name
    )
    tmp_cfg = {
        'environment': OmegaConf.to_container(cfg.environment),
        'solver': OmegaConf.to_container(cfg.solver),
        'arch': OmegaConf.to_container(cfg.arch),
        'train': OmegaConf.to_container(cfg.train)
    }
    wandb.config.update(tmp_cfg)

  if cfg.agent.dyn == "BicycleDstb5D":
    env_class = RaceCarDstb5DEnv
  else:
    raise ValueError("Dynamics type not supported!")

  # Constructs environment.
  print("\n== Environment information ==")
  env = env_class(cfg.environment, cfg.agent, cfg.cost)
  env.step_keep_constraints = False
  env.report()

  # Constructs solver.
  print("\n== Solver information ==")
  solver = NaiveRLDisturbance(cfg.solver, cfg.train, cfg.arch, cfg.environment)
  policy = solver.policy
  env.agent.init_policy(
      policy_type="NNCS", cfg=SimpleNamespace(device=policy.device),
      actor=policy.ctrl.net
  )
  n_params_ctrl = sum(
      p.numel() for p in policy.ctrl.net.parameters() if p.requires_grad
  )
  print(f'\nTotal parameters in ctrl: {n_params_ctrl}')
  print(f"We want to use: {cfg.train.device}, and Agent uses: {policy.device}")
  print("Critic is using cuda: ", next(policy.critic.net.parameters()).is_cuda)

  # Training starts.
  print("\n== Learning starts ==")
  vel_list = [0.5, 1., 1.5]
  yaw_list = [-np.pi / 3, -np.pi / 4, -np.pi / 8, 0., np.pi / 6, np.pi / 2]
  visualize_callback = partial(
      visualize, vel_list=vel_list, yaw_list=yaw_list,
      end_criterion=cfg.solver.rollout_end_criterion,
      T_rollout=cfg.solver.eval_timeout, nx=cfg.solver.cmap_res_x,
      ny=cfg.solver.cmap_res_y, subfigsz_x=cfg.solver.fig_size_x,
      subfigsz_y=cfg.solver.fig_size_y, vmin=-cfg.environment.g_x_fail,
      vmax=cfg.environment.g_x_fail, markersz=40
  )
  train_record, train_progress, violation_record, episode_record, pq_top_k = (
      solver.learn(env, visualize_callback=visualize_callback)
  )
  train_dict = {}
  train_dict['train_record'] = train_record
  train_dict['train_progress'] = train_progress
  train_dict['violation_record'] = violation_record
  train_dict['episode_record'] = episode_record
  train_dict['pq_top_k'] = list(pq_top_k.queue)
  save_obj(train_dict, os.path.join(cfg.solver.out_folder, 'train'))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-cf", "--config_file", help="config file path", type=str,
      default=os.path.join("config", "pretrain_dstb.yaml")
  )
  args = parser.parse_args()
  main(args.config_file)
