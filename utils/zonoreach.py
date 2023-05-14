# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
from typing import List, Optional, Tuple

from quickzonoreach.quickzonoreach.zono import zono_from_box, Zonotope


def get_box_samples(bound: np.ndarray, n_pts_per_axis: List[Optional[int]]):
  assert bound.shape[1] == 2
  if len(n_pts_per_axis) != bound.shape[0]:
    raise ValueError(
        f"The length of n_pts_per_axis should be {bound.shape[0]}."
    )
  gridpoints = []
  for _bound, n_pts in zip(bound, n_pts_per_axis):
    if n_pts is None:
      gridpoints.append(_bound)
    else:
      gridpoints.append(np.linspace(_bound[0], _bound[1], n_pts))

  n_axes = bound.shape[0]
  matrices = np.meshgrid(*gridpoints, indexing='ij')
  matrices = [mat[np.newaxis, ...] for mat in matrices]
  pts = np.concatenate(matrices, axis=0)
  return pts.reshape(n_axes, -1)


def get_box_vertices(bound: np.ndarray):
  """ Gets the vertices of a box.

  Args:
      bound (np.ndarray): each row consists of lower and upper bound of each
          axis.

  Returns:
      np.ndarray: the vertices of the box.
  """
  n_pts_per_axis = [None for _ in range(bound.shape[0])]
  return get_box_samples(bound, n_pts_per_axis)


def get_zono_samples_mixed(
    zono: Zonotope, vert_axes: List[int], box_axes: List[int], eps: float
) -> Tuple[np.ndarray, int]:
  _pts = zono.verts_sel(dims_sel=vert_axes, epsilon=eps).T  # (#axes, #pts)
  n_pts_base = _pts.shape[1]
  zono_box = zono.box_bounds()

  gridpoints = [zono_box[axis] for axis in box_axes]
  matrices = np.meshgrid(*gridpoints, indexing='ij')
  matrices = [mat[np.newaxis, ...] for mat in matrices]
  grid_f = np.concatenate(matrices, axis=0).reshape(len(box_axes), -1)
  n_repeats = grid_f.shape[1]

  pts = np.zeros((zono.center.size, n_pts_base))
  pts[vert_axes] = _pts
  pts = np.tile(pts, (1, n_repeats))
  for j in range(n_repeats):
    start = j * n_pts_base
    stop = (j+1) * n_pts_base
    pts[box_axes, start:stop] = grid_f[:, j].reshape(-1, 1)
  return pts, n_pts_base


def get_zonotope_reachset(
    init_box: np.ndarray, a_mat_list: List[np.ndarray],
    b_mat_list: List[np.ndarray], input_box_list: List[np.ndarray],
    bias_list: Optional[List[np.ndarray]] = None,
    save_list: Optional[List[bool]] = None
) -> List[Zonotope]:
  """ Computes the FRSs based on zonotopes.

  Args:
      init_box (np.ndarray): the initial set, R_0.
      a_mat_list (List[np.ndarray]): the jacobian w.r.t x, F_x.
      b_mat_list (List[np.ndarray]): the jacobian w.r.t control input, F_u/F_d.
      input_box_list (List[np.ndarray]): the input set, U/D.
      bias_list (Optional[List[np.ndarray]], optional): the bias term to add to
          the FRS computation. Defaults to None.
      save_list (Optional[List[bool]], optional): indices to save the FRSs.
          Defaults to None.

  Returns:
      List[Zonotope]: a list of FRSs.
  """

  assert len(a_mat_list) == len(b_mat_list) == len(input_box_list), (
      "all lists should be same length"
  )

  if bias_list is None:
    bias_list = [None for _ in range(len(b_mat_list))]
  else:
    assert len(b_mat_list) == len(bias_list), "all lists should be same length"

  # save everything by default
  if save_list is None:
    save_list = [True] * (len(a_mat_list) + 1)

  assert len(save_list) == len(a_mat_list) + 1, (
      "Save mat list should be one longer than the other lists"
  )

  rv = []

  def custom_func(index, zonotope):
    if save_list[index]:
      rv.append(zonotope.clone())

  iterate_zonotope_reachset(
      init_box, a_mat_list, b_mat_list, input_box_list, bias_list, custom_func
  )

  return rv


def iterate_zonotope_reachset(
    init_box: np.ndarray, a_mat_list: List[np.ndarray],
    b_mat_list: List[np.ndarray], input_box_list: List[np.ndarray],
    bias_list: Optional[List[np.ndarray]], custom_func
):
  """ Computes the FRSs based on zonotopes.

  Args:
      init_box (np.ndarray): the initial set, R_0.
      a_mat_list (List[np.ndarray]): the jacobian w.r.t x, F_x.
      b_mat_list (List[np.ndarray]): the jacobian w.r.t control input, F_u/F_d.
      input_box_list (List[np.ndarray]): the input set, U/D.
      bias_list (Optional[List[np.ndarray]], optional): the bias term to add to
          the FRS computation. Defaults to None.
      custom_func(Callable): a callback to operate FRS at every timestep.

  Returns:
      List[Zonotope]: a list of FRSs.
  """
  z: Zonotope
  z = zono_from_box(init_box)

  index = 0
  custom_func(index, z)
  index += 1

  for a_mat, b_mat, input_box, bias in zip(
      a_mat_list, b_mat_list, input_box_list, bias_list
  ):

    z.center = np.dot(a_mat, z.center)
    if bias is not None:
      z.center += bias
    z.mat_t = np.dot(a_mat, z.mat_t)

    # add new generators for inputs
    if b_mat is not None:
      z.mat_t = np.concatenate((z.mat_t, b_mat), axis=1)

      if isinstance(input_box, np.ndarray):
        input_box = input_box.tolist()

      z.init_bounds += input_box

      num_gens = z.mat_t.shape[1]
      assert len(z.init_bounds) == num_gens, (
          f"Zonotope had {num_gens} generators, "
          + f"but only {len(z.init_bounds)} bounds were there."
      )

    custom_func(index, z)
    index += 1


# == CHECK ==
def check_verts_in_box(
    verts: np.ndarray, center: float, box: np.ndarray, axis: int
) -> bool:
  """
  Args:
      verts (np.ndarray): _description_
      box (np.ndarray): _description_
      axis (int): which axis to check.

  Returns:
      bool: True if any of the vertices is outside the box.
  """
  verts_shift = center + verts[:, axis].copy()
  flag = np.any(verts_shift >= box[1]) or np.any(verts_shift <= box[0])
  return flag
