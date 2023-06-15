# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from typing import Callable, Optional, Union, List, Any, Tuple
import matplotlib as mpl
from matplotlib import cm
import numpy as np
import torch
import jax.numpy as jnp
from quickzonoreach.quickzonoreach.zono import Zonotope

from simulators import RaceCarDstb5DEnv, RaceCarSingle5DEnv
from shapely.geometry import Polygon, MultiPoint
from matplotlib import pyplot as plt


def get_values(
    env: Union[RaceCarSingle5DEnv, RaceCarDstb5DEnv],
    critic: Callable[[np.ndarray, Optional[Union[np.ndarray, torch.Tensor]]],
                     np.ndarray], xs: np.ndarray, ys: np.ndarray, v: float,
    yaw: float, batch_size: int, delta: float = 0., fail_value: float = 1.
):
  values = np.full((xs.shape[0], ys.shape[0]), fill_value=fail_value)
  it = np.nditer(values, flags=['multi_index'])
  while not it.finished:
    idx_all = []
    obs_all = np.empty((0, env.obs_dim), dtype=float)
    while len(idx_all) < batch_size:
      idx = it.multi_index
      x = xs[idx[0]]
      y = ys[idx[1]]
      if env.state_dim == 4:
        state = np.array([x, y, v, yaw]).reshape(-1, 1)
      elif env.state_dim == 5:
        state = np.array([x, y, v, yaw, delta]).reshape(-1, 1)

      closest_pt, slope, theta = env.track.get_closest_pts(state[:2, :])
      state_jnp = jnp.array(state)
      control_jnp = jnp.zeros((2, 1))
      closest_pt = jnp.array(closest_pt)
      slope = jnp.array(slope)
      theta = jnp.array(theta)
      dummy_time_indices = jnp.zeros((1, state_jnp.shape[1]), dtype=int)
      g_x = env.constraint.get_cost(
          state_jnp, control_jnp, closest_pt, slope, theta,
          time_indices=dummy_time_indices
      )
      g_x = np.asarray(g_x)[0]
      if g_x < 0.:
        obs = env.get_obs(state.reshape(-1))
        obs_all = np.concatenate((obs_all, obs.reshape(1, -1)))
        idx_all += [idx]
      it.iternext()
      if it.finished:
        break
    v_all = critic(obs_all, append=None)
    for v_s, idx in zip(v_all, idx_all):
      values[idx] = v_s
  return values


def plot_traj(
    ax, trajectory: np.ndarray, result: int, c: str = 'b', lw: float = 2.,
    zorder: int = 1, vel_scatter: bool = False, s: int = 40
):
  traj_x = trajectory[:, 0]
  traj_y = trajectory[:, 1]

  if vel_scatter:
    vel = trajectory[:, 2]
    ax.scatter(
        traj_x[0], traj_y[0], s=s, c=vel[0], cmap=cm.copper, vmin=0, vmax=2.,
        edgecolor='none', marker='s', zorder=zorder
    )
    ax.scatter(
        traj_x[1:-1], traj_y[1:-1], s=s - 12, c=vel[1:-1], cmap=cm.copper,
        vmin=0, vmax=2., edgecolor='none', marker='o', zorder=zorder
    )
    if result == -1:
      marker_final = 'X'
      edgecolor_final = 'r'
    elif result == 1:
      marker_final = '*'
      edgecolor_final = 'g'
    else:
      marker_final = '^'
      edgecolor_final = 'y'
    ax.scatter(
        traj_x[-1], traj_y[-1], s=s, c=vel[-1], cmap=cm.copper, vmin=0,
        vmax=2., edgecolor=edgecolor_final, marker=marker_final, zorder=zorder
    )
  else:
    ax.scatter(traj_x[0], traj_y[0], s=s, c=c, zorder=zorder)
    ax.plot(traj_x, traj_y, c=c, ls='-', lw=lw, zorder=zorder)

    if result == -1:
      ax.scatter(traj_x[-1], traj_y[-1], s=s, marker='x', c='r', zorder=zorder)
    elif result == 1:
      ax.scatter(traj_x[-1], traj_y[-1], s=s, marker='*', c='g', zorder=zorder)
    else:
      ax.scatter(traj_x[-1], traj_y[-1], s=s, marker='^', c='y', zorder=zorder)


def get_trajectories_zs(
    env: RaceCarDstb5DEnv, adversary: Callable[[np.ndarray, np.ndarray, Any],
                                               np.ndarray],
    vel_list: List[float], yaw_list: List[float], num_pts: int = 5,
    T_rollout: int = 150, end_criterion: str = "failure"
):
  num_traj = len(vel_list) * len(yaw_list) * num_pts
  reset_kwargs_list = []
  for _ in range(num_pts):
    far_failure = False
    cnt = 0
    a_dummy = {'ctrl': np.zeros((2, 1)), 'dstb': np.zeros((5, 1))}
    while (not far_failure) and (cnt <= 10):
      env.reset()
      state = env.state.copy()
      cons_dict = env.get_constraints(state, a_dummy, state)
      constraint_values = None
      for key, value in cons_dict.items():
        if constraint_values is None:
          num_pts = value.shape[1]
          constraint_values = value
        else:
          assert num_pts == value.shape[1], (
              "The length of constraint ({}) do not match".format(key)
          )
          constraint_values = np.concatenate((constraint_values, value),
                                             axis=0)
      g_x = np.max(constraint_values[:, -1], axis=0)
      far_failure = g_x <= -0.1
      cnt += 1
    for vel in vel_list:
      for yaw in yaw_list:
        state[2] = vel
        state[3] = yaw
        reset_kwargs_list.append(dict(state=state.copy()))

  trajectories, results, _ = env.simulate_trajectories(
      num_traj, T_rollout=T_rollout, end_criterion=end_criterion,
      reset_kwargs_list=reset_kwargs_list, adversary=adversary
  )
  return trajectories, results


def plot_poly(
    ax, poly: Polygon, c: str = 'g', alpha: float = 1., lw: float = 1.,
    plot_box: bool = False, c_box: str = 'm', alpha_box: float = 1.,
    lw_box: float = 1., zorder: int = 1
):
  x, y = poly.exterior.xy
  ax.plot(x, y, c=c, alpha=alpha, lw=lw, zorder=zorder)
  if plot_box:
    xmin, ymin, xmax, ymax = poly.bounds
    ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin],
            c=c_box, alpha=alpha_box, lw=lw_box, zorder=zorder)


def plot_poly_with_fp_single_yaw(
    ax, frs_poly: Polygon, env: RaceCarDstb5DEnv, state: np.ndarray,
    c: str = 'orange', alpha: float = 1., lw: float = 1
) -> Polygon:
  ego = env.agent.footprint.move2state(np.array([0., 0., state[3]]))
  vertices = ego.center[[0, 1]] + ego.offset
  footprint_poly = Polygon(vertices)

  xs1, ys1 = np.asarray(frs_poly.exterior.xy)
  xs2, ys2 = np.asarray(footprint_poly.exterior.xy)
  xs1: np.ndarray = xs1[:-1]
  xs2: np.ndarray = xs2[:-1]
  ys1: np.ndarray = ys1[:-1]
  ys2: np.ndarray = ys2[:-1]

  xs2 = xs2.reshape(-1, 1)
  ys2 = ys2.reshape(-1, 1)
  x_all = (xs1 + xs2).reshape(-1)
  y_all = (ys1 + ys2).reshape(-1)

  pts = MultiPoint([(x, y) for x, y in zip(x_all, y_all)])
  frs_aug_poly: Polygon = pts.convex_hull  # should be a Polygon
  plot_poly(ax, frs_aug_poly, c=c, alpha=alpha, lw=lw)
  return frs_aug_poly


def plot_poly_with_fp(
    ax, frs_verts: np.ndarray, env: RaceCarDstb5DEnv, state: np.ndarray,
    c: str = 'orange', alpha: float = 1., lw: float = 1, zorder: int = 1
) -> Polygon:
  x_all = np.empty((0,))
  y_all = np.empty((0,))
  for i in range(frs_verts.shape[0]):
    state_aug = state[[0, 1, 3]] + frs_verts[i, :]
    ego = env.agent.footprint.move2state(state_aug)
    fp_verts = ego.center[[0, 1]] + ego.offset
    fp_poly = Polygon(fp_verts)

    xs, ys = np.array(fp_poly.exterior.xy)
    x_all = np.concatenate((x_all, xs))
    y_all = np.concatenate((y_all, ys))

  pts = MultiPoint([(x, y) for x, y in zip(x_all, y_all)])
  frs_aug_poly: Polygon = pts.convex_hull  # should be a Polygon
  plot_poly(ax, frs_aug_poly, c=c, alpha=alpha, lw=lw, zorder=zorder)
  return frs_aug_poly


def plot_step(
    env: RaceCarDstb5DEnv,
    solver_info: dict,
    test_adversary: Callable[[np.ndarray, np.ndarray, Any], np.ndarray],
    extent: Optional[np.ndarray] = None,
    s=32,
    savefig: bool = True,
    fig_path: Optional[str] = None,
    plot_footprint: bool = False,
    color_dict: Optional[dict] = None,
    fontsize: float = 10,
    shift_extent: Optional[int] = None,
    transparent: bool = True,
    plot_linear: bool = True,
    plot_even_failed: bool = True,
    only_check: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
):
  if figsize is None:
    figsize = (4, 2)
  fig, ax = plt.subplots(1, 1, figsize=figsize)
  mpl.rcParams['font.family'] = 'monospace'
  if color_dict is None:
    color_dict = {
        'nominal': 'b',
        'linear': '#E77500',
        'frs': '#E77500',
        'fp': '#1DB102',
    }

  env.render_obs(ax=ax, c='k')
  env.track.plot_track(ax, c='k')

  # Retrieves information.
  monitor_info = solver_info['monitor']
  traj_n = monitor_info['nominal']['nom_states'].T
  traj_l = None

  if 'frs' in solver_info['monitor']:
    zonotopes: List[Zonotope] = monitor_info['frs']['zonotopes']
    stop = len(zonotopes)
    ax.plot(traj_n[:2, 0], traj_n[:2, 1], linewidth=2., c='k')
    ax.plot(traj_n[1:stop, 0], traj_n[1:stop, 1], linewidth=2., c='b')
    ax.scatter(traj_n[0, 0], traj_n[0, 1], c='k', s=s, zorder=2)
    ax.scatter(
        traj_n[1, 0], traj_n[1, 1], c=color_dict['nominal'], s=s, zorder=2
    )

    if plot_even_failed:
      ax.plot(
          traj_n[stop - 1:, 0], traj_n[stop - 1:, 1], linewidth=2., c='b',
          alpha=0.5
      )

    if plot_linear:
      env.agent.policy = monitor_info['frs']['linear_policy']
      traj_l, _, _ = env.simulate_one_trajectory(
          T_rollout=traj_n.shape[0] - 1, end_criterion='timeout',
          adversary=test_adversary, reset_kwargs={'state': traj_n[0].copy()}
      )
      ax.plot(
          traj_l[:stop, 0], traj_l[:stop, 1], linewidth=2,
          c=color_dict['linear'], linestyle='--'
      )
      if plot_even_failed:
        ax.plot(
            traj_l[stop - 1:, 0], traj_l[stop - 1:, 1], linewidth=2,
            c=color_dict['linear'], linestyle='--', alpha=0.5
        )

    plot_indices = np.linspace(1, len(zonotopes) - 1, 4, dtype=int)[1:]
    # plot_indices = np.linspace(1, len(zonotopes) - 1, 3, dtype=int)
    for i, zono in enumerate(zonotopes):
      verts = zono.verts_sel(dims_sel=[0, 1, 3], epsilon=0.01)
      xs = verts[:, 0].copy()
      ys = verts[:, 1].copy()
      xs += traj_n[i, 0]
      ys += traj_n[i, 1]
      # Prevents weird shape.
      pts = MultiPoint([(x, y) for x, y in zip(xs, ys)])
      frs_poly: Polygon = pts.convex_hull

      if i in plot_indices:
        ax.scatter(traj_n[i, 0], traj_n[i, 1], c=color_dict['nominal'], s=s)
        if plot_linear:
          ax.scatter(traj_l[i, 0], traj_l[i, 1], c=color_dict['linear'], s=s)
        plot_poly(ax, frs_poly, c=color_dict['linear'], lw=2., plot_box=False)

      if plot_footprint:
        alpha_footprint = .6 if i in plot_indices else 0.1
        plot_poly_with_fp(
            ax, verts, env, state=traj_n[i], c=color_dict['fp'],
            alpha=alpha_footprint, lw=1.
        )
        # plot_poly_with_fp_single_yaw(
        #     ax, frs_poly, env, state=traj_n[i], c=color_dict['fp'],
        #     alpha=alpha_footprint
        # )
  else:
    ax.plot(traj_n[:, 0], traj_n[:, 1], linewidth=2., c='b')
    env.render_footprint(
        ax, state=traj_n[-1], c=color_dict['nominal'], lw=1., alpha=1.
    )

  ax.plot([], [], c=color_dict['nominal'], linestyle='-', label='nominal')
  ax.plot([], [], c=color_dict['linear'], linestyle='--', label='linear')
  if plot_footprint:
    ax.plot([], [], c=color_dict['fp'], linestyle='-', label='fp')

  if extent is None:
    extent = env.visual_extent

  if shift_extent is not None:
    extent[0] = np.floor(traj_n[0, 0]) - 1
    extent[1] = extent[0] + shift_extent
    # if traj_l is None:
    #   extent[1] = np.ceil(traj_n[-1, 0]) + 1
    # else:
    #   extent[1] = np.ceil(max(traj_n[-1, 0], traj_l[-1, 0])) + 1

  ax.axis(extent)
  ax.set_aspect('equal')
  ax.tick_params(axis='both', which='major', labelsize=fontsize - 2)

  if only_check:
    ax.axis('off')
  else:
    ax.set_xticks([extent[0], extent[1]])
    ax.set_yticks([extent[2], extent[3]])

    if solver_info['shield']:
      if 'frs' in solver_info['monitor']:
        txt = f" at {monitor_info['frs']['failure_idx']}"
      else:
        txt = ''
      ax.set_title(
          f"Shields due to {monitor_info['raised_reason']}" + txt,
          fontsize=fontsize
      )
    else:
      ax.set_title("Uses task ctrl", fontsize=fontsize)
    ax.legend(
        ncol=3, framealpha=0., loc=10, bbox_to_anchor=(0.5, -0.2),
        fontsize=fontsize - 4
    )
  plt.tight_layout()
  if savefig:
    fig.savefig(fig_path, dpi=200, transparent=transparent)
  return fig
