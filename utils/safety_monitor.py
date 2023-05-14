# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Classes for safety monitors in safety filter.
"""

from typing import Optional, Dict, Tuple, Callable, Any, List
from abc import ABC, abstractmethod
import copy
import numpy as np
from jax import numpy as jnp
from types import SimpleNamespace

from quickzonoreach.quickzonoreach.zono import Zonotope, zono_from_box
from utils.zonoreach import get_zono_samples_mixed
from agent.base_block import Critic
from simulators import (
    Bicycle5DRefTrajCost, ILQRSpline, LinearPolicy, BaseZeroSumEnv
)

# TODO: turns adversary into agent's policy.


class BaseMonitor(ABC):

  def __init__(
      self, adversary: Callable[[np.ndarray, np.ndarray, Any], np.ndarray]
  ):
    super().__init__()
    self.adversary = adversary

  def add_imaginary_actions(
      self,
      obs: np.ndarray,
      agents_action: Optional[Dict[str, np.ndarray]] = None,
      **kwargs,
  ):
    if agents_action is None:
      action_dict = {}
    else:
      action_dict = copy.deepcopy(agents_action)
    action_dict['dstb'] = self.adversary(obs, action_dict['ctrl'], **kwargs)

    return action_dict

  @abstractmethod
  def check(
      self, obs: np.ndarray,
      agents_action: Optional[Dict[str, np.ndarray]] = None, **kwargs
  ) -> bool:
    raise NotImplementedError


class ValueMonitor(BaseMonitor):

  def __init__(
      self, adversary: Callable[[np.ndarray, np.ndarray, Any], np.ndarray],
      critic: Critic, value_threshold: float, value_to_be_max: bool
  ):
    super().__init__(adversary=adversary)
    self.critic = critic
    self.value_threshold = value_threshold
    self.value_to_be_max = value_to_be_max  # if True, bigger the better.

  def check(
      self, obs: np.ndarray,
      agents_action: Optional[Dict[str, np.ndarray]] = None, **kwargs
  ) -> Tuple[bool, dict]:
    """
    Args:
        obs (np.ndarray): _description_
        agents_action (Dict[str, np.ndarray], optional): . Defaults to None.

    Returns:
        bool: overrides the proposed action (if True).
    """
    if 'ctrl' not in agents_action:
      raise ValueError("ctrl must be provided.")

    action_dict_all = self.add_imaginary_actions(
        obs=obs, agents_action=agents_action, **kwargs
    )

    if isinstance(self.critic.action_src, list):
      action = np.concatenate([
          action_dict_all[key] for key in self.critic.action_src
      ], axis=-1)
    else:
      action = action_dict_all[self.critic.action_src]

    append = kwargs.get('append', None)
    latent = kwargs.get('latent', None)
    q_val = self.critic.value(obs, action, append=append, latent=latent)

    monitor_info = {'q_val': q_val}
    if self.value_to_be_max:
      override_flag = q_val < self.value_threshold
    else:
      override_flag = q_val > self.value_threshold

    if override_flag:
      monitor_info['raised_reason'] = 'value'
    else:
      monitor_info['raised_reason'] = None
    return override_flag, monitor_info


class RolloutMonitor(BaseMonitor):

  def __init__(
      self, adversary: Callable[[np.ndarray, np.ndarray, Any], np.ndarray],
      env: BaseZeroSumEnv, imag_end_criterion: str, imag_steps: int
  ):
    super().__init__(adversary=adversary)
    # Assumes agents' policies have been set up before constructing the monitor.
    self.env = copy.deepcopy(env)
    self.imag_end_criterion = imag_end_criterion
    self.imag_steps = imag_steps
    self.env.end_criterion = self.imag_end_criterion

  def get_nominal_traj(
      self, obs: np.ndarray, state: np.ndarray,
      agents_action: Optional[Dict[str, np.ndarray]] = None, **kwargs
  ) -> Tuple[bool, dict]:
    """
    Args:
        obs (np.ndarray): _description_
        agents_action (Dict[str, np.ndarray], optional): . Defaults to None.

    Returns:
        bool: overrides the proposed action (if True).
    """
    # Placeholders.
    nom_states = np.zeros((self.env.state_dim, self.imag_steps + 2))
    # The last ctrl/dstb are dummy values (set to zero).
    nom_ctrls = np.zeros((self.env.action_dim_ctrl, self.imag_steps + 2))
    nom_dstbs = np.zeros((self.env.action_dim_dstb, self.imag_steps + 2))

    # Applies the first action.
    self.env.reset(state=state.copy())

    action_dict_all = self.add_imaginary_actions(
        obs=obs, agents_action=agents_action, state=state, **kwargs
    )
    _, _, done, step_info = self.env.step(action_dict_all)
    nom_states[:, 0] = state.copy()
    nom_ctrls[:, 0] = step_info['ctrl_clip'].copy()
    nom_dstbs[:, 0] = step_info['dstb_clip'].copy()

    if done:  # The episode is terminated.
      result = 0
      if step_info["done_type"] == "success":
        result = 1
      elif step_info["done_type"] == "failure":
        result = -1
      nom_states[:, 1] = self.env.state.copy()
      valid_length = 2
    else:
      _traj, result, rollout_info = self.env.simulate_one_trajectory(
          T_rollout=self.imag_steps, end_criterion=self.env.end_criterion,
          adversary=self.adversary,
          reset_kwargs=dict(state=self.env.state.copy())
      )
      nom_states[:, 1:1 + len(_traj)] = _traj.T
      nom_ctrls[:, 1:len(_traj)] = rollout_info['action_hist']['ctrl'].copy().T
      nom_dstbs[:, 1:len(_traj)] = rollout_info['action_hist']['dstb'].copy().T
      valid_length = 1 + len(_traj)

    info = {
        'nom_states': nom_states[:, :valid_length],
        'nom_ctrls': nom_ctrls[:, :valid_length],
        'nom_dstbs': nom_dstbs[:, :valid_length],
    }
    if self.env.end_criterion == "reach-avoid" and result != 1:
      info['raised_reason'] = 'reach-avoid'
      return True, info
    elif self.env.end_criterion == "failure" and result == -1:
      info['raised_reason'] = 'avoid'
      return True, info
    info['raised_reason'] = None
    return False, info

  def check(
      self, obs: np.ndarray, agents_action: Dict[str, np.ndarray], **kwargs
  ) -> Tuple[bool, dict]:
    """
    Args:
        obs (np.ndarray): _description_
        agents_action (Dict[str, np.ndarray], optional): . Defaults to None.

    Returns:
        bool: overrides the proposed action (if True).
    """
    if 'state' not in kwargs:
      raise ValueError("state must be provided.")
    if 'ctrl' not in agents_action:
      raise ValueError("ctrl must be provided.")

    override_flag, info = self.get_nominal_traj(
        obs=obs, agents_action=agents_action, **kwargs
    )
    return override_flag, info


class RobustFRSMonitor(RolloutMonitor):

  def __init__(
      self,
      adversary: Callable[[np.ndarray, np.ndarray, Any], np.ndarray],
      env: BaseZeroSumEnv,
      imag_end_criterion: str,
      imag_steps: int,
      track_policy: ILQRSpline,
      cfg_ref_cost,
      dstb_bound: np.ndarray,
      ctrl_bound: np.ndarray,
      buffer: float,
  ):
    super().__init__(
        adversary=adversary, env=env, imag_end_criterion=imag_end_criterion,
        imag_steps=imag_steps
    )
    self.track_policy = copy.deepcopy(track_policy)
    self.cfg_ref_cost = cfg_ref_cost
    self.dstb_bound = dstb_bound.copy()
    self.ctrl_bound = ctrl_bound.copy()
    self.buffer = buffer

  def get_linear_policy(
      self, nom_states: np.ndarray, nom_ctrls: np.ndarray,
      nom_dstbs: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    self.track_policy.cost = Bicycle5DRefTrajCost(
        self.cfg_ref_cost, jnp.asarray(nom_states), jnp.asarray(nom_ctrls)
    )
    closest_pt, slope, theta = self.track_policy.track.get_closest_pts(
        np.asarray(nom_states[:2, :])
    )
    closest_pt = jnp.array(closest_pt)
    slope = jnp.array(slope)
    theta = jnp.array(theta)

    c_x, c_u, c_xx, c_uu, c_ux = self.track_policy.cost.get_derivatives(
        nom_states, nom_ctrls, closest_pt, slope, theta,
        time_indices=jnp.arange(nom_states.shape[1]).reshape(1, -1)
    )
    fx, fu, fd = self.env.agent.dyn.get_jacobian(
        nom_states[:, :-1], nom_ctrls[:, :-1], nom_dstbs[:, :-1]
    )
    K_closed_loop, k_open_loop, _ = self.track_policy.backward_pass(
        c_x=c_x, c_u=c_u, c_xx=c_xx, c_uu=c_uu, c_ux=c_ux, fx=fx, fu=fu,
        reg=1e-8
    )

    fx = np.array(fx)
    fu = np.array(fu)
    fd = np.array(fd)
    K_closed_loop = np.array(K_closed_loop[..., :fx.shape[-1]])
    k_open_loop = np.array(k_open_loop[..., :fx.shape[-1]])
    return K_closed_loop, k_open_loop, fx, fu, fd

  def get_frs_step(
      self, zono: Zonotope, fx: np.ndarray, fu: np.ndarray, fd: np.ndarray,
      K_cl: np.ndarray, input_box: np.ndarray,
      bias: Optional[np.ndarray] = None
  ) -> Zonotope:
    zono_new = zono.clone()
    A = fx + fu@K_cl
    B = fd

    # c -> Ac + bias
    zono_new.center = np.dot(A, zono_new.center)
    if bias is not None:
      zono_new.center += bias

    # G -> AG
    zono_new.mat_t = np.dot(A, zono_new.mat_t)

    # G -> [G, B], x -> [x, y]
    zono_new.mat_t = np.concatenate((zono_new.mat_t, B), axis=1)
    if isinstance(input_box, np.ndarray):
      input_box = input_box.tolist()
    zono_new.init_bounds += input_box

    return zono_new

  def get_frs(
      self, nom_states: np.ndarray, nom_ctrls: np.ndarray,
      nom_dstbs: np.ndarray
  ) -> Tuple[List[Zonotope], np.ndarray, np.ndarray]:
    K_closed_loop, k_open_loop, fx, fu, fd = self.get_linear_policy(
        nom_states, nom_ctrls, nom_dstbs
    )

    init_box = np.empty((fx.shape[0], 2))
    init_box[:, 0] = -1e-6
    init_box[:, 1] = 1e-6

    # Gets forward reachable sets via zonotope representation.
    zono = zono_from_box(init_box)
    zonotopes = [zono]
    for i in range(K_closed_loop.shape[-1]):
      input_box = self.dstb_bound.copy() - nom_dstbs[:, [i]]
      bias = fu[..., i] @ k_open_loop[..., i]
      zonotopes.append(
          self.get_frs_step(
              zonotopes[-1], fx=fx[..., i], fu=fu[..., i], fd=fd[..., i],
              K_cl=K_closed_loop[..., i], input_box=input_box, bias=bias
          )
      )

    return zonotopes, K_closed_loop, k_open_loop

  def check_single_step_with_verts_sel(
      self, zono: Zonotope, nom_state: np.ndarray, nom_ctrl: np.ndarray,
      check_ctrl: bool, K_closed_loop: Optional[np.ndarray], eps: float = 0.01
  ) -> Tuple[bool, Optional[str]]:
    zono_pts, n_pts_base = get_zono_samples_mixed(
        zono, vert_axes=[0, 1, 3], box_axes=[2, 4], eps=eps
    )

    # Checks footprint constrains, we only use first `n_pts_base` as we do not
    # need information from the `box_axes`.
    states = jnp.array(zono_pts[:, :n_pts_base] + nom_state.reshape(-1, 1))
    dummy_ctrls = jnp.zeros((nom_ctrl.shape[0], states.shape[1]))
    dummy_time_indices = jnp.arange(states.shape[1]).reshape(1, -1)
    cons_dict: Dict[str, np.ndarray] = self.env.get_constraints_all(
        states, dummy_ctrls, dummy_time_indices
    )
    for k, v in cons_dict.items():
      if np.any(v >= -self.buffer):
        return True, k

    if check_ctrl:  # Checks if controls are in the control bound.
      d_ctrls = np.einsum("in,nj->ij", K_closed_loop, zono_pts)  # (2, n_pts)
      d_ctrls_max = np.max(d_ctrls, axis=1)
      d_ctrls_min = np.min(d_ctrls, axis=1)
      if (
          np.any(d_ctrls_max > self.ctrl_bound[:, 1] - nom_ctrl)
          or np.any(d_ctrls_min < self.ctrl_bound[:, 0] - nom_ctrl)
      ):
        return True, 'ctrl'

    return False, None

  def check(
      self, obs: np.ndarray, agents_action: Dict[str, np.ndarray], **kwargs
  ) -> Tuple[bool, dict]:
    """
    Args:
        obs (np.ndarray): _description_
        agents_action (Dict[str, np.ndarray], optional): . Defaults to None.

    Returns:
        bool: overrides the proposed action (if True).
    """
    override_flag, nom_info = super().check(
        obs=obs, agents_action=agents_action, **kwargs
    )
    info = {'nominal': nom_info, 'raised_reason': 'nominal'}
    if override_flag:
      return override_flag, info  # returns early.

    # Computes forward reachable sets (FRSs).
    nom_states = nom_info['nom_states']
    nom_ctrls = nom_info['nom_ctrls']
    nom_dstbs = nom_info['nom_dstbs']
    K_closed_loop, k_open_loop, fx, fu, fd = self.get_linear_policy(
        nom_states, nom_ctrls, nom_dstbs
    )

    if nom_states.shape[-1] - 1 != K_closed_loop.shape[-1]:
      raise ValueError(
          f"{K_closed_loop.shape[-1]} policies but get "
          + f"({nom_states.shape[-1]}) states."
      )
    cfg_linear_policy = SimpleNamespace(device='cpu')
    linear_policy = LinearPolicy(
        id='linear', nominal_states=nom_states[..., :-1],
        nominal_controls=nom_ctrls[..., :-1], K_closed_loop=K_closed_loop,
        k_open_loop=k_open_loop, cfg=cfg_linear_policy
    )
    frs_info = {'failure_idx': None, 'linear_policy': linear_policy}

    init_box = np.empty((fx.shape[0], 2))
    init_box[:, 0] = -1e-6
    init_box[:, 1] = 1e-6
    zono = zono_from_box(init_box)
    zonotopes = [zono]

    # The number of FRSs is equal to the length of control sequence plus 1.
    # Note that we add a dummy control to `nom_ctrls`.
    for i in range(K_closed_loop.shape[-1] + 1):
      if i == 0:  # don't monitor the initial state.
        override_flag = False
      else:
        if i == K_closed_loop.shape[-1]:  # don't check the last ctrl.
          check_ctrl = False
          K = None
        else:
          check_ctrl = True
          K = K_closed_loop[..., i]

        override_flag, reason = self.check_single_step_with_verts_sel(
            zono=zonotopes[-1], nom_state=nom_states[..., i],
            nom_ctrl=nom_ctrls[..., i], check_ctrl=check_ctrl, K_closed_loop=K
        )

      if override_flag:
        frs_info['failure_idx'] = i
        info['raised_reason'] = reason
        break
      elif i != K_closed_loop.shape[-1]:
        input_box = self.dstb_bound.copy() - nom_dstbs[:, [i]]
        bias = fu[..., i] @ k_open_loop[..., i]
        zonotopes.append(
            self.get_frs_step(
                zonotopes[-1], fx=fx[..., i], fu=fu[..., i], fd=fd[..., i],
                K_cl=K_closed_loop[..., i], input_box=input_box, bias=bias
            )
        )

    frs_info['zonotopes'] = zonotopes
    info['frs'] = frs_info
    return override_flag, info
