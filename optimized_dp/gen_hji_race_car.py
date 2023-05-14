"""
This file is used to generate the optimal value function for the zero-sum game
between the race car controller and (additive) disturbance. The failure set is
defined by the simulator.
"""

import numpy as np
import sys
import os
from shutil import copyfile
import argparse
from tqdm import tqdm
from jax import numpy as jnp
import pickle
from omegaconf import OmegaConf

from odp.Grid import Grid
from odp.solver import HJSolver

from optimized_dp.odp_utils import plot_isosurface, PlotOptionsAug
from optimized_dp.odp_dyn import BicycleDstb5D

from simulators import save_obj, PrintLogger, Bicycle5DConstraint, Track


def get_state_cost_map(
    constraint: Bicycle5DConstraint, track: Track, nx: int, ny: int,
    vel: float, yaw: float, delta: float, xmin: float, xmax: float,
    ymin: float, ymax: float
) -> np.ndarray:

  state = np.zeros((5, nx * ny))
  offset_xs = np.linspace(xmin, xmax, nx)
  offset_ys = np.linspace(ymin, ymax, ny)
  offset_xv, offset_yv = np.meshgrid(offset_xs, offset_ys, indexing='ij')
  offset = np.concatenate(
      (offset_xv[..., np.newaxis], offset_yv[..., np.newaxis]), axis=-1
  )
  state[:2, :] = np.array(offset.reshape(-1, 2)).T
  state[2, :] = vel
  state[3, :] = yaw
  state[4, :] = delta
  closest_pt, slope, theta = track.get_closest_pts(
      state[:2, :], normalize_progress=True
  )
  ctrl = np.zeros((2, nx * ny))

  state = jnp.array(state)
  ctrl = jnp.array(ctrl)
  closest_pt = jnp.array(closest_pt)
  slope = jnp.array(slope)
  theta = jnp.array(theta)
  dummy_time_indices = jnp.zeros((1, state.shape[1]), dtype=int)
  v = constraint.get_cost(
      state, ctrl, closest_pt, slope, theta, time_indices=dummy_time_indices
  ).reshape(nx, ny)
  return v


def main(config_file):
  # region: config
  cfg = OmegaConf.load(config_file)

  uMode = cfg.dyn.uMode
  dMode = 'max'
  if uMode == 'max':
    dMode = 'min'
  plot_type = cfg.log.plot_type

  out_folder = cfg.log.out_folder
  if cfg.cost.has_vel_constr:
    out_folder = out_folder + "_vel"
  if cfg.cost.has_delta_constr:
    out_folder = out_folder + "_steer"

  os.makedirs(out_folder, exist_ok=True)
  copyfile(config_file, os.path.join(out_folder, 'config.yaml'))
  sys.stdout = PrintLogger(os.path.join(out_folder, 'log.txt'))
  # endregion

  # region: Grid and Dynamics.
  # periodicDims = [3, 4]
  periodicDims = []
  grid = Grid(
      minBounds=np.array(cfg.solver.grid_min),
      maxBounds=np.array(cfg.solver.grid_max),
      dims=len(cfg.solver.grid_min),
      pts_each_dim=np.array(cfg.solver.pts_each_dim),
      periodicDims=periodicDims,
  )
  with np.printoptions(precision=2, suppress=True):
    print(grid.grid_points[3] / np.pi * 180)
    print(grid.grid_points[4])
  slicesCut = cfg.log.slicesCut

  my_car = BicycleDstb5D(
      uMode=uMode, dMode=dMode, L=cfg.dyn.L, uMax=np.array(cfg.dyn.ctrl_bound),
      dMax=np.array(cfg.dyn.dstb_bound), v_max=cfg.dyn.v_max,
      v_min=cfg.dyn.v_min, delta_max=cfg.dyn.delta_max,
      delta_min=cfg.dyn.delta_min
  )
  if plot_type == '3d':
    plot_option = PlotOptionsAug(
        do_plot=False, plot_type="3d_plot", plotDims=[0, 1, 3],
        slicesCut=slicesCut
    )
    _delta = grid.grid_points[4][slicesCut[0]] * 180 / np.pi
    if cfg.log.plot_vf:
      print(f"Takes vel snapshot at {grid.grid_points[2][slicesCut[0]]:.2f}.")
      print(f"Takes delta snapshot at {_delta:.2f}.")
  else:
    plot_option = PlotOptionsAug(
        do_plot=False, plot_type="2d_plot", plotDims=[0, 1],
        slicesCut=slicesCut, vamp=0.2
    )
    _yaw = grid.grid_points[3][slicesCut[0]] * 180 / np.pi
    _delta = grid.grid_points[4][slicesCut[0]] * 180 / np.pi
    if cfg.log.plot_vf:
      print(f"Takes vel snapshot at {grid.grid_points[2][slicesCut[0]]:.2f}.")
      print(f"Takes yaw snapshot at {_yaw:.2f}.")
      print(f"Takes delta snapshot at {_delta:.2f}.")
  plot_option.title = out_folder
  # endregion

  # region: failure set
  # env = RaceCarDstb5DEnv(cfg.cost, cfg.agent, cfg.cost)
  if hasattr(cfg.solver, "init_value_file"):
    with open(cfg.solver.init_value_file, "rb") as f:
      Initial_value_f = pickle.load(f)
      print("loads initial values from", cfg.solver.init_value_file)
  else:
    constraint = Bicycle5DConstraint(cfg.cost)
    track_width_right = cfg.cost.track_width_right
    track_width_left = cfg.cost.track_width_left
    track_len = cfg.cost.track_len
    _center_line_x = np.linspace(
        start=0., stop=track_len, num=1000, endpoint=True
    ).reshape(1, -1)
    center_line = np.concatenate(
        (_center_line_x, np.zeros_like(_center_line_x)), axis=0
    )
    track = Track(
        center_line=center_line, width_left=track_width_left,
        width_right=track_width_right, loop=getattr(cfg.cost, 'loop', True)
    )
    obs = np.zeros(grid.pts_each_dim)

    # Optimized DP assumes the value in the set to be negative, while our
    # formulation considers a failure set. -> flips the sign.
    for i in tqdm(range(grid.pts_each_dim[2])):
      for j in range(grid.pts_each_dim[3]):
        for k in range(grid.pts_each_dim[4]):
          obs[..., i, j, k] = -get_state_cost_map(
              constraint=constraint, track=track, nx=grid.pts_each_dim[0],
              ny=grid.pts_each_dim[1], vel=grid.grid_points[2][i],
              yaw=grid.grid_points[3][j], delta=grid.grid_points[4][k],
              xmin=grid.min[0], xmax=grid.max[0], ymin=grid.min[1],
              ymax=grid.max[1]
          )
    Initial_value_f = obs
  if cfg.log.plot_vf:
    plot_isosurface(grid=grid, V=Initial_value_f, plot_option=plot_option)
  # endregion

  # region: HJI
  lookback_length = cfg.dyn.horizon
  t_step = cfg.dyn.dt
  small_number = 1e-5
  tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

  compMethods = {"TargetSetMode": "minVWithV0"}
  result = HJSolver(
      my_car, grid, Initial_value_f, tau, compMethods, plot_option,
      saveAllTimeSteps=True, accuracy="low", verbose=cfg.log.verbose
  )
  print(result.shape)
  result_dict = {"values": result, "grid": grid, "my_car": my_car}
  save_obj(result_dict, os.path.join(out_folder, 'results'))
  result_lw_dict = {
      "values": result[..., [0, -1]],
      "grid": grid,
      "my_car": my_car
  }
  save_obj(result_dict, os.path.join(out_folder, 'results'))
  save_obj(result_lw_dict, os.path.join(out_folder, 'results_lw'))
  if cfg.log.plot_vf:
    plot_isosurface(grid=grid, V=result[..., 0], plot_option=plot_option)
  # endregion


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-cf", "--config_file", help="config file path", type=str,
      default=os.path.join("optimized_dp", "bic5d.yaml")
  )
  args = parser.parse_args()
  main(args.config_file)
