# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from typing import Tuple, Optional, Any, Dict, Callable
import copy
import time
import numpy as np
from scipy import interpolate

from odp.Grid import Grid
from odp.solver import computeSpatDerivArray

from optimized_dp.odp_dyn import BicycleDstb5D
from simulators.policy.base_policy import BasePolicy

# ! Only for BicycleDstb5D for now.


def get_deriv_value_interp(
    grid: Grid, odp_values: np.ndarray, interp_method: str = 'linear',
    accuracy: str = "low", get_init_value: bool = False,
    get_terminal_value: bool = False
) -> None:
  assert accuracy == "low" or accuracy == "medium"
  x_derivative: np.ndarray = computeSpatDerivArray(
      grid, odp_values[..., 0], deriv_dim=1, accuracy=accuracy
  )[..., np.newaxis]
  y_derivative: np.ndarray = computeSpatDerivArray(
      grid, odp_values[..., 0], deriv_dim=2, accuracy=accuracy
  )[..., np.newaxis]
  v_derivative: np.ndarray = computeSpatDerivArray(
      grid, odp_values[..., 0], deriv_dim=3, accuracy=accuracy
  )[..., np.newaxis]
  yaw_derivative: np.ndarray = computeSpatDerivArray(
      grid, odp_values[..., 0], deriv_dim=4, accuracy=accuracy
  )[..., np.newaxis]
  delta_derivative: np.ndarray = computeSpatDerivArray(
      grid, odp_values[..., 0], deriv_dim=5, accuracy=accuracy
  )[..., np.newaxis]

  derivatives_values = np.concatenate((
      x_derivative, y_derivative, v_derivative, yaw_derivative,
      delta_derivative
  ), axis=-1)
  if get_terminal_value:
    derivatives_values = np.concatenate(
        (derivatives_values, odp_values[..., 0, np.newaxis]), axis=-1
    )
  if get_init_value:
    derivatives_values = np.concatenate(
        (derivatives_values, odp_values[..., -1, np.newaxis]), axis=-1
    )

  return interpolate.RegularGridInterpolator(
      tuple(grid.grid_points), derivatives_values, method=interp_method,
      bounds_error=False, fill_value=None
  )


def get_opt_ctrl_dstb(
    obs: np.ndarray, interp_fn: Callable[[np.ndarray], np.ndarray],
    my_car: BicycleDstb5D, interp_method: str = 'linear', verbose: bool = False
) -> Tuple[float, np.ndarray, np.ndarray]:
  if obs.shape[0] == 6:  # ! hacky
    state = np.zeros(5)
    state[:3] = obs[:3].copy()
    state[3] = np.arctan(obs[4] / obs[3])
    state[4] = obs[5]
  else:
    state = obs.copy()
  deriv_value = interp_fn(obs, method=interp_method).reshape(-1)
  spat_deriv = deriv_value[:5]
  if verbose:
    with np.printoptions(precision=2):
      print("spatial deriv: ", end='')
      print(spat_deriv)

  return (
      my_car.opt_ctrl_np(None, state, spat_deriv),
      my_car.opt_dstb_np(None, state, spat_deriv)
  )


class ODPPolicy(BasePolicy):

  def __init__(
      self, id: str, car: BicycleDstb5D, grid: Grid, odp_values: np.ndarray,
      cfg: Any
  ):
    super().__init__(id, cfg)
    self.policy_type = "ODP"

    self.car = copy.deepcopy(car)
    self.interp_fn = get_deriv_value_interp(
        grid, odp_values, accuracy=getattr(cfg, "accuracy", "low"),
        get_init_value=getattr(cfg, "get_init_value", "false"),
        get_terminal_value=getattr(cfg, "get_terminal_value", "false")
    )
    self.interp_method = getattr(cfg, "interp_method", 'linear')

  def get_action(
      self, obs: np.ndarray,
      agents_action: Optional[Dict[str, np.ndarray]] = None, **kwargs
  ) -> Tuple[np.ndarray, dict]:
    """Gets the action to execute.

    Args:
        state (np.ndarray): current state.

    Returns:
        np.ndarray: the action to be executed.
        dict: info for the solver, e.g., processing time, status, etc.
    """
    time0 = time.time()
    action = get_opt_ctrl_dstb(
        kwargs.get('state'), self.interp_fn, self.car, self.interp_method,
        verbose=False
    )[0]
    t_process = time.time() - time0
    status = 1
    return action, dict(t_process=t_process, status=status)

  def get_opt_dstb(
      self, obs: np.ndarray, ctrl: np.ndarray, **kwargs
  ) -> np.ndarray:
    return get_opt_ctrl_dstb(
        kwargs.get('state'), self.interp_fn, self.car, self.interp_method,
        verbose=False
    )[1]


class ODPDstbPolicy(BasePolicy):

  def __init__(
      self, id: str, car: BicycleDstb5D, grid: Grid, odp_values: np.ndarray,
      cfg: Any
  ):
    super().__init__(id, cfg)
    self.policy_type = "ODP"

    self.car = copy.deepcopy(car)
    self.interp_fn = get_deriv_value_interp(
        grid, odp_values, accuracy=getattr(cfg, "accuracy", "low"),
        get_init_value=getattr(cfg, "get_init_value", "false"),
        get_terminal_value=getattr(cfg, "get_terminal_value", "false")
    )
    self.interp_method = getattr(cfg, "interp_method", 'linear')

  def get_action(
      self, obs: np.ndarray,
      agents_action: Optional[Dict[str, np.ndarray]] = None, **kwargs
  ) -> Tuple[np.ndarray, dict]:
    """Gets the action to execute.

    Args:
        state (np.ndarray): current state.

    Returns:
        np.ndarray: the action to be executed.
        dict: info for the solver, e.g., processing time, status, etc.
    """
    time0 = time.time()
    action = self.get_opt_dstb(obs=None, ctrl=None, **kwargs)
    t_process = time.time() - time0
    status = 1
    return action, dict(t_process=t_process, status=status)

  def get_opt_dstb(
      self, obs: np.ndarray, ctrl: np.ndarray, **kwargs
  ) -> np.ndarray:
    return get_opt_ctrl_dstb(
        kwargs.get('state'), self.interp_fn, self.car, self.interp_method,
        verbose=False
    )[1]
