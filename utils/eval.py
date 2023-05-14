# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from typing import List, Callable, Any
import os
import copy
import pickle
import numpy as np
from functools import partial
from types import SimpleNamespace

from utils.dstb import dummy_dstb, random_dstb, adv_dstb
from optimized_dp.odp_policy import ODPPolicy
from agent.base_block import Actor


def get_disturbance(
    dstb_type: str, num_envs: int, key: str, **kwargs
) -> List[Callable[[np.ndarray, np.ndarray, Any], np.ndarray]]:
  print(f"Constructs {key} with {dstb_type}.")
  if dstb_type == 'odp':
    print("  => Loading pretrained HJI values from", kwargs['odp_folder'])
    with open(os.path.join(kwargs['odp_folder'], "results_lw.pkl"), "rb") as f:
      result_dict = pickle.load(f)
    odp_policy = ODPPolicy(
        id='dstb', car=result_dict['my_car'], grid=result_dict['grid'],
        odp_values=result_dict['values'],
        cfg=SimpleNamespace(device='cpu', interp_method='linear')
    )
    del result_dict
    dstb_fn_list = [odp_policy.get_opt_dstb for _ in range(num_envs)]
    # deepcopy gets slower and consumes more memory.
    # dstb_fn_list = []
    # for i in range(num_envs):
    #   if i == 0:
    #     odp_policy_dup = odp_policy
    #   else:
    #     odp_policy_dup = copy.deepcopy(odp_policy)
    #   dstb_fn_list.append(odp_policy_dup.get_opt_dstb)

  elif dstb_type == 'isaacs':
    dstb_fn_list = []
    dstb_actor: Actor = kwargs['dstb_actor']
    dstb_actor.restore(
        kwargs['dstb_step'], kwargs['model_folder'], verbose=True
    )
    dstb_actor.net.device = "cpu"
    dstb_actor.net.to("cpu")
    for _ in range(num_envs):
      dstb_policy = copy.deepcopy(dstb_actor.net)
      dstb_fn_list.append(
          partial(
              adv_dstb, dstb_policy=dstb_policy,
              use_ctrl=dstb_actor.obs_other_list is not None
          )
      )
  elif dstb_type == 'dummy':
    dstb_fn_list = [
        partial(dummy_dstb, dim=kwargs['action_dim_dstb'])
        for _ in range(num_envs)
    ]
  elif dstb_type == 'random':
    dstb_fn_list = [
        partial(
            random_dstb, rng=kwargs['rng'], dstb_range=kwargs['dstb_range']
        ) for _ in range(num_envs)
    ]
  else:
    raise ValueError(f"Unsupported imaginary disturbance type {dstb_type }.")
  return dstb_fn_list
