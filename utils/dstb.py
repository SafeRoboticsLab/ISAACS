# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from typing import Optional
import numpy as np


def dummy_dstb(obs, ctrl, append=None, dim: int = 0, **kwargs):
  return np.zeros(dim)


def random_dstb(obs, ctrl, **kwargs):
  rng: np.random.Generator = kwargs.get("rng")
  dstb_range: np.ndarray = kwargs.get("dstb_range")
  return rng.uniform(low=dstb_range[:, 0], high=dstb_range[:, 1])


def adv_dstb(
    obs: np.ndarray, ctrl: np.ndarray, dstb_policy,
    append: Optional[np.ndarray] = None, use_ctrl: bool = True, **kwargs
) -> np.ndarray:
  if use_ctrl:
    obs_dstb = np.concatenate((obs, ctrl), axis=-1)
  else:
    obs_dstb = obs
  dstb = dstb_policy(obs_dstb, append=append)
  assert isinstance(dstb, np.ndarray)
  return dstb
