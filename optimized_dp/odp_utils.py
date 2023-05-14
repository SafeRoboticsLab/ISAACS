from typing import List, Optional, Tuple, Callable
import scipy
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from odp.Grid import Grid
from odp.Plots.plot_options import PlotOptions


class PlotOptionsAug(PlotOptions):

  def __init__(
      self, do_plot=True, plot_type="3d_plot", plotDims=[], slicesCut=[],
      min_isosurface=0, max_isosurface=0, vamp: Optional[float] = None,
      title: Optional[str] = None
  ):
    super().__init__(
        do_plot=do_plot, plot_type=plot_type, plotDims=plotDims,
        slicesCut=slicesCut, min_isosurface=min_isosurface,
        max_isosurface=max_isosurface
    )
    self.vamp = vamp
    self.title = title


def lateral_accel_shape(
    grid: Grid, wheelbase: float, values: Tuple[float, float]
) -> np.ndarray:
  data = np.zeros(grid.pts_each_dim)
  vels = grid.vs[2]
  deltas = grid.vs[4]
  tmp = vels**2 * np.tan(deltas) / wheelbase
  # Union of two half spaces.
  data += np.minimum(tmp - values[0], values[1] - tmp)
  return data


def box_to_half_space_footprint_shape(
    grid: Grid, dim: int, state_box_limit: np.ndarray,
    values: Tuple[float, float], yaw_dim: int = 3
) -> np.ndarray:
  assert dim == 0 or dim == 1
  data = np.zeros(grid.pts_each_dim)
  xs = grid.grid_points[0]
  ys = grid.grid_points[1]
  yaws = grid.grid_points[yaw_dim]

  xv, yv = np.meshgrid(xs, ys, indexing='ij')
  pos = np.concatenate((xv[..., np.newaxis], yv[..., np.newaxis]), axis=-1)
  rot_matrices = np.array([[np.cos(yaws), -np.sin(yaws)],
                           [np.sin(yaws), np.cos(yaws)]])
  rot_matrices = rot_matrices.transpose(2, 0, 1)

  offset = np.array([[state_box_limit[0], state_box_limit[2]],
                     [state_box_limit[0], state_box_limit[3]],
                     [state_box_limit[1], state_box_limit[2]],
                     [state_box_limit[1], state_box_limit[3]]])
  rot_offset: np.ndarray = np.einsum("ijk,lk->ijl", rot_matrices, offset)
  rot_offset = rot_offset.transpose(0, 2, 1)
  pos_final = (
      pos[..., np.newaxis, np.newaxis, :] + rot_offset[np.newaxis, np.newaxis]
  )  # n_x, n_y, n_yaws, n_offset, 2

  lower_half = np.min(pos_final[..., dim] - values[0], axis=-1)
  upper_half = np.min(values[1] - pos_final[..., dim], axis=-1)
  _data = np.minimum(lower_half, upper_half)

  if grid.dims == 3:
    data += _data
  elif grid.dims == 4:
    if yaw_dim == 2:
      data += _data[..., np.newaxis]
    elif yaw_dim == 3:
      data += _data[..., np.newaxis, :]
  elif grid.dims == 5:
    if yaw_dim == 2:
      data += _data[..., np.newaxis, np.newaxis]
    elif yaw_dim == 3:
      data += _data[..., np.newaxis, :, np.newaxis]
    elif yaw_dim == 4:
      data += _data[..., np.newaxis, np.newaxis, :]
  return data


def box_to_box_footprint_shape(
    grid: Grid, state_box_limit: np.ndarray, box_spec: np.ndarray,
    yaw_dim: int = 3, offset_precision: Tuple[float, float] = (11, 31)
) -> np.ndarray:
  data = np.zeros(grid.pts_each_dim)
  xs = grid.grid_points[0]
  ys = grid.grid_points[1]
  yaws = grid.grid_points[yaw_dim]

  xv, yv = np.meshgrid(xs, ys, indexing='ij')
  pos = np.concatenate((xv[..., np.newaxis], yv[..., np.newaxis]), axis=-1)
  rot_matrices = np.array([[np.cos(yaws), -np.sin(yaws)],
                           [np.sin(yaws), np.cos(yaws)]])
  rot_matrices = rot_matrices.transpose(2, 0, 1)  # (n_yaws, 2, 2)

  offset_xs = np.linspace(
      state_box_limit[0], state_box_limit[1], offset_precision[0]
  )
  offset_ys = np.linspace(
      state_box_limit[2], state_box_limit[3], offset_precision[1]
  )
  offset_xv, offset_yv = np.meshgrid(offset_xs, offset_ys, indexing='ij')
  offset = np.concatenate(
      (offset_xv[..., np.newaxis], offset_yv[..., np.newaxis]), axis=-1
  )
  offset = offset.reshape(-1, 2)

  # box
  box_center = np.array([box_spec[0], box_spec[1]])
  box_yaw = box_spec[2]
  obs_rot_mat = np.array([[np.cos(box_yaw), -np.sin(box_yaw)],
                          [np.sin(box_yaw), np.cos(box_yaw)]])
  box_halflength = box_spec[3]
  box_halfwidth = box_spec[4]

  # rotation
  rot_offset: np.ndarray = np.einsum("ijk,lk->ilj", rot_matrices, offset)
  pos = (
      pos[..., np.newaxis, np.newaxis, :] + rot_offset[np.newaxis, np.newaxis]
  )  # n_x, n_y, n_yaws, n_offset, 2
  pos_final = np.einsum("ik,jlmnk->jlmni", obs_rot_mat, pos - box_center)

  diff_x = np.maximum(
      pos_final[..., 0] - box_halflength, -box_halflength - pos_final[..., 0]
  )
  diff_y = np.maximum(
      pos_final[..., 1] - box_halfwidth, -box_halfwidth - pos_final[..., 1]
  )
  diff = np.maximum(diff_x, diff_y)
  _data = np.min(diff, axis=-1)

  if grid.dims == 3:
    data += _data
  elif grid.dims == 4:
    if yaw_dim == 2:
      data += _data[..., np.newaxis]
    elif yaw_dim == 3:
      data += _data[..., np.newaxis, :]
  elif grid.dims == 5:
    if yaw_dim == 2:
      data += _data[..., np.newaxis, np.newaxis]
    elif yaw_dim == 3:
      data += _data[..., np.newaxis, :, np.newaxis]
    elif yaw_dim == 4:
      data += _data[..., np.newaxis, np.newaxis, :]
  return data


def box_shape(
    grid: Grid, target_min: List, target_max: List,
    ignore_dims: Optional[List] = None
) -> np.ndarray:
  """Computes the implicit surface function for a hyper-box.

  Args:
      grid (Grid): the value grids.
      target_min (List): the minimum of each dimension.
      target_max (List): the maximum of each dimension.

  Returns:
      np.ndarray: implicit surface values.
  """

  data = None
  if ignore_dims is None:
    ignore_dims = []
  idx = 0
  for i in range(grid.dims):
    # np.maximum automatically broadcasts to a common shape.
    if i in ignore_dims:
      ub = grid.max[i]
      lb = grid.min[i]
    else:
      ub = target_max[idx]
      lb = target_min[idx]
      idx += 1
    if data is None:
      data = np.maximum(-grid.vs[i] + lb, grid.vs[i] - ub)
    else:
      data = np.maximum(data, grid.vs[i] - ub)
      data = np.maximum(data, -grid.vs[i] + lb)

  return data


def plot_isosurface(grid: Grid, V: np.ndarray, plot_option: PlotOptionsAug):
  dims_plot = plot_option.dims_plot
  idx = [slice(None)] * grid.dims
  slice_idx = 0

  dims_list = list(range(grid.dims))
  for i in dims_list:
    if i not in dims_plot:
      idx[i] = plot_option.slices[slice_idx]
      slice_idx += 1

  if len(dims_plot) == 2:
    my_V = V[tuple(idx)]
    if (my_V > 0.0).all() or (my_V < 0.0).all():
      print(
          "Implicit surface is not shown since all values have the same sign."
      )
    if plot_option.vamp is None:
      vamp = min(np.max(my_V), np.abs(np.min(my_V)))
    else:
      vamp = plot_option.vamp
    fig = px.imshow(
        my_V.T, color_continuous_scale='RdBu',
        x=grid.grid_points[dims_plot[0]], y=grid.grid_points[dims_plot[1]],
        zmin=-vamp, zmax=vamp, labels={
            'x': 'x',
            'y': 'y'
        }, title=plot_option.title
    )
  elif len(dims_plot) == 3:
    dim1, dim2, dim3 = dims_plot[0], dims_plot[1], dims_plot[2]
    complex_x = complex(0, grid.pts_each_dim[dim1])
    complex_y = complex(0, grid.pts_each_dim[dim2])
    complex_z = complex(0, grid.pts_each_dim[dim3])
    mg_X, mg_Y, mg_Z = np.mgrid[grid.min[dim1]:grid.max[dim1]:complex_x,
                                grid.min[dim2]:grid.max[dim2]:complex_y,
                                grid.min[dim3]:grid.max[dim3]:complex_z]

    my_V = V[tuple(idx)]

    if (V > 0.0).all() or (V < 0.0).all():
      print(
          "Implicit surface is not shown since all values have the same sign."
      )
    fig = go.Figure(
        data=go.Isosurface(
            x=mg_X.flatten(),
            y=mg_Y.flatten(),
            z=mg_Z.flatten(),
            value=my_V.flatten(),
            colorscale='RdBu',
            isomin=plot_option.min_isosurface,
            isomax=plot_option.max_isosurface,
            caps=dict(x_show=True, y_show=True),
        )
    )
  else:
    raise Exception('dims_plot length should be equal to 3\n')
  fig.show()
