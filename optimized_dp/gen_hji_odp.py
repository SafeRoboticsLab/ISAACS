"""
This file is used to generate the optimal value function for the zero-sum game
between the race car controller and (additive) disturbance. The failure set is
defined by the surface functions in `odp.Shapes` and `optimized_dp.odp_utils`.
"""

import numpy as np
import sys
import os
from shutil import copyfile
import argparse
from omegaconf import OmegaConf

from odp.Grid import Grid
from odp.Shapes import Lower_Half_Space, Upper_Half_Space, Union
from odp.solver import HJSolver

from optimized_dp.odp_utils import (
    plot_isosurface, PlotOptionsAug, box_to_half_space_footprint_shape,
    box_to_box_footprint_shape
)
from optimized_dp.odp_dyn import (
    BicycleDstb5D, DubinsCarDstb3D, DubinsCarDstb4D
)
from simulators import save_obj, PrintLogger


def main(config_file):
  # region: config
  cfg = OmegaConf.load(config_file)
  out_folder = cfg.log.out_folder

  dyn_class = cfg.dyn.dyn_class
  uMode = cfg.dyn.uMode
  dMode = 'max'
  if uMode == 'max':
    dMode = 'min'
  plot_type = cfg.log.plot_type
  road_constr = cfg.dyn.road_constr
  vel_constr = cfg.dyn.vel_constr
  steer_constr = cfg.dyn.steer_constr

  if dyn_class == 'BicycleDstb5D':
    yaw_dim = 3
    periodicDims = [3, 4]
    # ignore_dims = [2, 3, 4]
  elif dyn_class == 'DubinsCarDstb4D':
    yaw_dim = 3
    periodicDims = [3]
    steer_constr = False
    # ignore_dims = [2, 3]
  else:
    yaw_dim = 2
    periodicDims = [2]
    vel_constr = False
    steer_constr = False
    # ignore_dims = [2]

  if road_constr:
    out_folder = out_folder + "_road"
  if vel_constr:
    out_folder = out_folder + "_vel"
  if steer_constr:
    out_folder = out_folder + "_steer"

  os.makedirs(out_folder, exist_ok=True)
  copyfile(config_file, os.path.join(out_folder, 'config.yaml'))
  sys.stdout = PrintLogger(os.path.join(out_folder, 'log.txt'))
  # endregion

  # region: Grid and Dynamics.
  grid = Grid(
      minBounds=np.array(cfg.solver.grid_min),
      maxBounds=np.array(cfg.solver.grid_max),
      dims=len(cfg.solver.grid_min),
      pts_each_dim=np.array(cfg.solver.pts_each_dim),
      periodicDims=periodicDims,
  )
  print(grid.grid_points[4])
  slicesCut = cfg.log.slicesCut

  if dyn_class == 'BicycleDstb5D':
    my_car = BicycleDstb5D(
        uMode=uMode, dMode=dMode, L=cfg.dyn.L,
        uMax=np.array(cfg.dyn.ctrl_bound), dMax=np.array(cfg.dyn.dstb_bound),
        v_max=cfg.dyn.v_max, v_min=cfg.dyn.v_min, delta_max=cfg.dyn.delta_max,
        delta_min=cfg.dyn.delta_min
    )
    if plot_type == '3d':
      plot_option = PlotOptionsAug(
          do_plot=False, plot_type="3d_plot", plotDims=[0, 1, 3],
          slicesCut=slicesCut
      )
      _delta = grid.grid_points[4][slicesCut[0]] * 180 / np.pi
      print(f"Takes vel snapshot at {grid.grid_points[2][slicesCut[0]]:.2f}.")
      print(f"Takes delta snapshot at {_delta:.2f}.")
    else:
      plot_option = PlotOptionsAug(
          do_plot=False, plot_type="2d_plot", plotDims=[0, 1],
          slicesCut=slicesCut, vamp=0.2
      )
      _yaw = grid.grid_points[3][slicesCut[0]] * 180 / np.pi
      _delta = grid.grid_points[4][slicesCut[0]] * 180 / np.pi
      print(f"Takes vel snapshot at {grid.grid_points[2][slicesCut[0]]:.2f}.")
      print(f"Takes yaw snapshot at {_yaw:.2f}.")
      print(f"Takes delta snapshot at {_delta:.2f}.")
  elif dyn_class == 'DubinsCarDstb4D':
    my_car = DubinsCarDstb4D(
        uMode=uMode, dMode=dMode, uMax=np.array(cfg.dyn.ctrl_bound),
        dMax=np.array(cfg.dyn.dstb_bound)
    )
    if plot_type == '3d':
      plot_option = PlotOptionsAug(
          do_plot=False, plot_type="3d_plot", plotDims=[0, 1, 3],
          slicesCut=slicesCut
      )
      print(f"Takes vel snapshot at {grid.grid_points[2][slicesCut[0]]:.1f}.")
    else:
      plot_option = PlotOptionsAug(
          do_plot=False, plot_type="2d_plot", plotDims=[0, 1],
          slicesCut=slicesCut, vamp=0.2
      )
      _yaw = grid.grid_points[3][slicesCut[1]] * 180 / np.pi
      print(f"Takes vel snapshot at {grid.grid_points[2][slicesCut[0]]:.2f}.")
      print(f"Takes yaw snapshot at {_yaw:.2f}.")
  else:
    my_car = DubinsCarDstb3D(
        uMode=uMode, dMode=dMode, speed=1., uMax=np.array(cfg.dyn.ctrl_bound),
        dMax=np.array(cfg.dyn.dstb_bound)
    )
    if plot_type == '3d':
      plot_option = PlotOptionsAug(
          do_plot=False,
          plot_type="3d_plot",
          plotDims=[0, 1, 2],
          slicesCut=[],
          # min_isosurface=-0.1
      )
    else:
      plot_option = PlotOptionsAug(
          do_plot=False, plot_type="2d_plot", plotDims=[0, 1],
          slicesCut=slicesCut, vamp=0.2
      )
      _yaw = grid.grid_points[2][slicesCut[0]] * 180 / np.pi
      print(f"Takes yaw snapshot at {_yaw:.2f}.")

  # endregion

  # region: failure set
  obs_half_length = 0.5 / 2
  obs_half_width = 0.2 / 2
  width = 0.2
  length = 0.5
  pos_array = np.array([[2, -0.4], [5, 0.4], [8, 0.1], [11, -0.1], [14, -0.4],
                        [14, 0.4]])
  state_box_limit = [0, length, -width / 2, width / 2]
  for i, pos in enumerate(pos_array):
    # _obs = box_shape(
    #     grid=grid,
    #     target_min=[pos[0] - obs_half_length, pos[1] - obs_half_width],
    #     target_max=[pos[0] + obs_half_length,
    #                 pos[1] + obs_half_width], ignore_dims=ignore_dims
    # )
    _obs = box_to_box_footprint_shape(
        grid=grid, state_box_limit=state_box_limit,
        box_spec=[pos[0], pos[1], 0., obs_half_length,
                  obs_half_width], yaw_dim=yaw_dim
    )
    if i == 0:
      obs = _obs
    else:
      obs = Union(obs, _obs)

  if dyn_class == 'DubinsCarDstb3D':
    yaw_min = Lower_Half_Space(grid=grid, dim=2, value=cfg.constr.yaw_min)
    yaw_max = Upper_Half_Space(grid=grid, dim=2, value=cfg.constr.yaw_max)
  else:
    yaw_min = Lower_Half_Space(grid=grid, dim=3, value=cfg.constr.yaw_min)
    yaw_max = Upper_Half_Space(grid=grid, dim=3, value=cfg.constr.yaw_max)
  obs = Union(obs, yaw_min)
  obs = Union(obs, yaw_max)
  if road_constr:
    # road_left = Lower_Half_Space(grid=grid, dim=1, value=-1.)
    # road_right = Upper_Half_Space(grid=grid, dim=1, value=1.)
    # road = Union(road_left, road_right)
    road = box_to_half_space_footprint_shape(
        grid=grid, dim=1, state_box_limit=state_box_limit,
        values=[-cfg.constr.track_width_right,
                cfg.constr.track_width_left], yaw_dim=yaw_dim
    )
    obs = Union(obs, road)
  if vel_constr:
    vel_min = Lower_Half_Space(grid=grid, dim=2, value=cfg.constr.v_min)
    vel_max = Upper_Half_Space(grid=grid, dim=2, value=cfg.constr.v_max)
    obs = Union(obs, vel_min)
    obs = Union(obs, vel_max)
  if steer_constr:
    steer_min = Lower_Half_Space(grid=grid, dim=4, value=cfg.constr.delta_min)
    steer_max = Upper_Half_Space(grid=grid, dim=4, value=cfg.constr.delta_max)
    obs = Union(obs, steer_min)
    obs = Union(obs, steer_max)
  plot_option.title = out_folder
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
