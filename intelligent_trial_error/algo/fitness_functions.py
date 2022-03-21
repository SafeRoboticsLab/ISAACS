
"""Behaviour descriptor implementations"""


import pdb

import math

import numpy as np
import pybullet as pb
import pybullet_utils.bullet_client as bc
import pybullet_data

def hexapod_fit_speed(trajectory):
    """Calculate fitness based on average speed from the trajectory."""
    distance = 0
    time = trajectory[-1][0]
    init_location = trajectory[0][1:3]
    #location_prev = trajectory[0][1:3]
    # Collect total distance traversed
    #for step in trajectory[1:]:
    #    location_current = step[1:3]
    #    distance += np.linalg.norm(location_current - location_prev)
    #    location_prev = location_current

    final_location = trajectory[-1][1:3]
    distance = final_location[0] - init_location[0]
    return distance / time


def hexapod_fit_orientation(trajectory):
    """
    Calculate fitness based on average speed from the trajectory
    Reference:
    https://hal.archives-ouvertes.fr/file/index/docid/841958/filename/t02pap489-cully.pdf
    """
    final_rot = pb.getEulerFromQuaternion(trajectory[-1][4:8])
    arrival_angle = round(final_rot[2] * 100) / 100.0;


    # Performance - Angle Difference (desrird angle and obtained angle fomr simulation)
    # Change of orientation of axis in counting the desried angle to account for frontal axis of the newer robot (x-axis:frontal axis)
    x = trajectory[-1][1] 
    y = trajectory[-1][2] 
      
    # Computation of desired angle (yaxis-north x-axis(postive))
    B = math.sqrt((0.25 * x * x) + ( 0.25 * y * y))
    alpha = math.atan2(y, x)
    A = B / math.cos(alpha)
    beta = math.atan2(y, x - A)
      
    if (x < 0):
        beta = beta - math.pi
    while (beta < -math.pi):
        beta += 2 * math.pi
    while (beta > math.pi):
        beta -= 2 * math.pi
      
    angle_diff = abs(angle_dist(beta, arrival_angle)); #angle dist was a finction made earlier up in the script
      

    
    fitness = - angle_diff
    return fitness


def angle_dist( a,  b):
    theta = b - a
    while (theta < -math.pi):
      theta += 2 * math.pi
    while (theta > math.pi):
      theta -= 2 * math.pi
    return theta
  
