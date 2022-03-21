
"""Behaviour descriptor implementations."""

import pdb

import numpy as np


def hexapod_grid_xy(bd_dimensions, trajectory):
    """
    Behaviour descriptor based on the final (x, y) resting position of the
    hexapod robot.
    """
    assert len(bd_dimensions) == 2
    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5
    # Final x,y position
    x_final, y_final = trajectory[-1][1:3]
    coef_0 = (x_final - x_min) / (x_max - x_min)
    coef_1 = (y_final - y_min) / (y_max - y_min)
    index_0 = int(coef_0 * (bd_dimensions[0] - 1))
    index_1 = int(coef_1 * (bd_dimensions[1] - 1))

    return (index_0, index_1)


def hexapod_grid_leg(bd_dimensions, trajectory):
    """
    Behaviour descriptor based on the proportion of time that each leg was
    in contact with the ground. (6D, 5^6 grid)
    """
    assert len(bd_dimensions) == 6
    # Leg-ground contacts
    leg_contacts = np.mean(np.array(trajectory)[1:, -6:], axis=0)
    indices = (leg_contacts * (np.array(bd_dimensions) - 1)).astype(int)
    return tuple(indices)
