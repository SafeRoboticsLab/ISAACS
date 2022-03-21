
"""Mutation operator implementations"""


import pdb

import numpy as np


def _iso_line(x, y):
    """ISO_DD helper function."""
    assert x.shape == y.shape
    # Algorithm parameters
    _use_distance = True
    _sigma_iso = 0.01
    _sigma_line = 0.2
    # Search components
    gauss_iso = np.random.normal(0, _sigma_iso, size=len(x))
    gauss_line = np.random.normal(0, _sigma_line)
    direction = (x - y)
    if not _use_distance:
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 0 else 1
        new_sample = x.copy() + gauss_iso + gauss_line * direction
    else:
        new_sample = x.copy() + gauss_iso + gauss_line * direction
    # return np.clip(new_sample, 0, 1)
    return new_sample


def mutate_iso_line(sampled_solutions, **kwargs):
    """
    Apply a mutation in the original parameter space, using Hypervolumes.

    Reference:
    Vassiliades V, Mouret JB.
    Discovering the elite hypervolume by leveraging interspecies correlation.
    GECCO 2018

    """
    nsmp = len(sampled_solutions)
    if nsmp == 0:
        return []
    # Extract parameters
    x_list = np.array([sln.ctrl_parameters for sln in sampled_solutions])
    # Generate #nsmp different pairs for ISO_DD
    y_idx = np.arange(nsmp)
    np.random.shuffle(y_idx)
    cnt = 0
    while (y_idx == np.arange(nsmp)).any():
        same_idx = np.where(y_idx == np.arange(nsmp))[0]
        swap_idx = (
            same_idx + np.random.randint(1, nsmp, size=len(same_idx)))
        swap_idx = swap_idx % nsmp
        tmp = y_idx[same_idx]
        y_idx[same_idx] = y_idx[swap_idx]
        y_idx[swap_idx] = tmp
        cnt += 1
        if cnt > nsmp:
            break

    # pdb.set_trace()

    y_list = x_list[y_idx]
    # Generate new mutated samples by applying ISO_DD
    mutated_parameters = np.stack(list(map(_iso_line, x_list, y_list)), axis=0)
    return np.clip(mutated_parameters, 0, 1)


def mutate_gaussian(sampled_solutions, sigma_iso=0.1, fraction=1, **kwargs):
    """
    Apply a perturbation in the original (high-dim) space.

    Args:
        sampled_solutions (list): sampled solutions to mutate
        sigma_iso (float): scale of the Gaussian noise
        fraction (float): fraction of parameter dimensions to mutate

    """
    nsmp = len(sampled_solutions)
    if nsmp == 0:
        return []
    new_param = np.array([sln.ctrl_parameters for sln in sampled_solutions])
    # Generate mutation noise
    if fraction < 1:
        # Mutate only along some of the dimensions
        dim_param = new_param.shape[1]
        mutation = np.zeros_like(new_param)
        sample_sz = max(1, int(fraction * dim_param))
        mutate_idxs = np.random.choice(
            np.arange(dim_param), size=sample_sz, replace=False)
        mutation_mean = np.zeros(dim_param)
        mutation_var = sigma_iso * np.ones(dim_param)
        mutation[:, mutate_idxs] = np.random.normal(
            loc=mutation_mean[mutate_idxs],
            scale=mutation_var[mutate_idxs],
            size=(nsmp, len(mutate_idxs)))
    else:
        # Mutate all the dimensions
        mutation = np.random.normal(
            loc=0, scale=sigma_iso, size=new_param.shape)
        # mutation = np.random.normal(loc=0, scale=param_stats['std'])
        # mutation = np.random.multivariate_normal(
        #                         mean=np.zeros(dim_param),
        #                         cov=sigma_iso*param_stats['cov'],
        #                         size=nsmp)
    mutated_parameters = new_param + mutation
    return np.clip(mutated_parameters, 0, 1)
