# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for safety filter.
"""

from typing import Optional, Dict, Tuple
import copy
import numpy as np

from simulators import BasePolicy, LinearPolicy
from utils.safety_monitor import BaseMonitor


class SafetyFilter(BasePolicy):

  def __init__(
      self, base_policy: BasePolicy, safety_policy: BasePolicy,
      monitor: BaseMonitor, override_type: str = 'naive', concise: bool = False
  ):
    super().__init__(base_policy.id, base_policy.cfg)
    self.base_policy = base_policy
    self.safety_policy = safety_policy
    self.monitor = monitor
    self.override_type = override_type
    self.concise = concise
    self.time_idx = None
    self.linear_policy = None

  def clear_cache(self):
    # print("clear cache!")
    self.time_idx = None
    self.linear_policy = None

  def override(
      self, obs: np.ndarray, agents_action: Dict[str, np.ndarray],
      monitor_info: dict, **kwargs
  ):
    if self.override_type == 'naive':
      a_s, safety_info = self.safety_policy.get_action(
          obs=obs, agents_action=agents_action, monitor_info=monitor_info,
          **kwargs
      )
    elif self.override_type == 'linear':
      has_linear_policy = (
          (self.time_idx is not None)
          and (self.time_idx < self.linear_policy.K_closed_loop.shape[-1])
      )
      if has_linear_policy:
        kwargs_dup = copy.deepcopy(kwargs)
        kwargs_dup['time_idx'] = self.time_idx
        a_s, safety_info = self.linear_policy.get_action(
            obs=obs, agents_action=agents_action, monitor_info=monitor_info,
            **kwargs_dup
        )
        self.time_idx += 1
      else:
        a_s, safety_info = self.safety_policy.get_action(
            obs=obs, agents_action=agents_action, monitor_info=monitor_info,
            **kwargs
        )
    else:
      raise ValueError(f"Unknown override type {self.override_type}.")

    return a_s, safety_info

  def get_action(
      self, obs: np.ndarray,
      agents_action: Optional[Dict[str, np.ndarray]] = None, **kwargs
  ) -> Tuple[np.ndarray, dict]:
    """Gets the action to execute.

    Args:
        obs (np.ndarray): current observation.
        agents_action (Optional[Dict]): other agents' actions that are
            observable to the ego agent.

    Returns:
        np.ndarray: the action to be executed.
        dict: info for the solver, e.g., processing time, status, etc.
    """
    if agents_action is None:
      action_dict = {}
    else:
      action_dict = copy.deepcopy(agents_action)

    a_p, base_info = self.base_policy.get_action(  # Proposed action.
        obs=obs, agents_action=agents_action, **kwargs
    )
    action_dict['ctrl'] = a_p  # ! hacky

    override_flag, monitor_info = self.monitor.check(
        obs=obs, agents_action=action_dict, **kwargs
    )

    info = {'raised_reason': monitor_info['raised_reason']}
    if not self.concise:
      info['base'] = base_info
      info['monitor'] = monitor_info
    if 'control' in base_info:
      info['controls'] = base_info['control']

    if override_flag:
      a_s, safety_info = self.override(
          obs, action_dict, monitor_info, **kwargs
      )
      info['shield'] = True
      info['task_action'] = a_p
      if not self.concise:
        info['safety'] = safety_info
      if 'controls' in safety_info:  # Overrides.
        info['controls'] = safety_info['controls']
      return a_s, info
    else:
      info['shield'] = False
      if 'frs' in monitor_info:  # ! hacky, has linear policy
        self.linear_policy: LinearPolicy = monitor_info['frs']['linear_policy']
        self.time_idx = 1
      return a_p, info
