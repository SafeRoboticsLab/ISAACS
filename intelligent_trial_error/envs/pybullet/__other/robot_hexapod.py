
"""Robot definition"""

import numpy as np

from pybullet_envs.gym_locomotion_envs import Ant
from pybullet_envs.robot_locomotors import WalkerBase


class HexapodRobot(Ant):
    """
    Wrapper around the original Ant environment.
    Allows setting the initial robot pose.
    Observation set to np.float16 to facilitate reproducibility.
    """

    def __init__(self, init_robot, scale=1, action_dim=8, obs_dim=28, **kwargs):
        WalkerBase.__init__(
            self,
            fn="ant.xml",
            robot_name="torso",
            action_dim=action_dim,
            obs_dim=obs_dim,
            power=2.5)
        self.init_robot_pos = init_robot  # [0, -1.5, .5]
        self.init_robot_orient = [0, 0, 0, 1]
        self.walk_target_x = self.init_robot_pos[0]
        self.walk_target_y = self.init_robot_pos[1]
        self.SCALE = scale
        self.power /= 2   # to account for max torque j.power_coeff

    def robot_specific_reset(self, bullet_client):
        """Put robot to a fixed initial position."""
        WalkerBase.robot_specific_reset(self, bullet_client)
        self.robot_body.reset_position(self.init_robot_pos)

    def calc_state(self):
        """Convert state to np.float16, and body_xyz based on hull."""
        j = np.array(
            [j.current_relative_position() for j in self.ordered_joints],
            dtype=np.float32).flatten()
        self.joint_speeds = j[1::2]
        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)
        body_pose = self.robot_body.pose()
        self.body_real_xyz = body_pose.xyz()
        self.body_xyz = body_pose.xyz()  # more stable
        self.body_rpy = body_pose.rpy()
        z = self.body_xyz[2]
        if self.initial_z == None:
            self.initial_z = z
        r, p, yaw = self.body_rpy
        self.walk_target_theta = np.arctan2(
            self.walk_target_y - self.body_xyz[1],
            self.walk_target_x - self.body_xyz[0])
        self.walk_target_dist = np.linalg.norm(
            [self.walk_target_y - self.body_xyz[1],
             self.walk_target_x - self.body_xyz[0]])
        angle_to_target = self.walk_target_theta - yaw
        # rotate speed back to body point of view
        rot_speed = np.array([[np.cos(-yaw), -np.sin(-yaw), 0],
                              [np.sin(-yaw), np.cos(-yaw), 0], [0, 0, 1]])
        vx, vy, vz = np.dot(rot_speed, self.robot_body.speed())
        more = np.array([
            z - self.initial_z,
            np.sin(angle_to_target),
            np.cos(angle_to_target),
            0.3 * vx,
            0.3 * vy,
            0.3 * vz,
            r,
            p],
            dtype=np.float32)
        standard_state = np.clip(
            np.concatenate([more] + [j] + [self.feet_contact]),
            -5, +5)
        return standard_state.astype(np.float16)

    def calc_potential(self):
        """
        Incentivise the agent to move away from the initial position.
        Or nothing.
        """
        # return self.walk_target_dist / self.scene.dt
        return 0
