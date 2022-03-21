
"""
Author:         Nemanja Rakicevic
Date:           September 2020
Description:
                Useful classes and functions for PyBullet environments.
"""

import os

from pybullet_envs.robot_bases import BodyPart
from pybullet_envs.scene_abstract import Scene


""" Constants """


REWARD_THRSH = 20
_VEL_THRSH = .0005


""" Functions for adding objects """


def get_cube(_p, x, y, z):
    body = _p.loadURDF(
        os.path.join(
            os.path.dirname(__file__),
            "assets/objects/wall.urdf"),
        [x, y, z])
    _p.changeDynamics(body, -1, mass=1.2)  # match Roboschool
    part_name, _ = _p.getBodyInfo(body)
    part_name = part_name.decode("utf8")
    bodies = [body]
    return BodyPart(_p, part_name, bodies, 0, -1)


def get_sphere(_p, x, y, z):
    body = _p.loadURDF(
        os.path.join(
            os.path.dirname(__file__),
            "assets/objects/ball_blue.urdf"),
        [x, y, z])
    part_name, _ = _p.getBodyInfo(body)
    part_name = part_name.decode("utf8")
    bodies = [body]
    return BodyPart(_p, part_name, bodies, 0, -1)


""" Camera classes """


class TopCamera:
    """Overwriting the visualisation angle to make it birds-eye view."""

    def __init__(self, env):
        self.env = env

    def move_and_look_at(self, i, j, k, x, y, z):
        lookat = self.env.camera_info['lookat']  # [x, y, z]
        distance = self.env.camera_info['camera']['distance'] - 4.5
        yaw = self.env.camera_info['camera']['yaw']
        pitch = self.env.camera_info['camera']['pitch']
        # distance, yaw, pitch = 3, -90., -45.
        self.env._p.resetDebugVisualizerCamera(distance, yaw, pitch, lookat)


class FollowCamera:
    """ Overwriting the visualisation angle to to follow (x, y, z)."""

    def __init__(self, env):
        self.env = env

    def move_and_look_at(self, i, j, k, x, y, z):
        lookat = [x, y, z]
        distance = 4.5
        yaw = 0
        pitch = -30
        # distance, yaw, pitch = 3, -90., -45.
        self.env._p.resetDebugVisualizerCamera(distance, yaw, pitch, lookat)
