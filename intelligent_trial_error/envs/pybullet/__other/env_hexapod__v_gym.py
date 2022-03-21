
"""Environment definition"""

import numpy as np
import pybullet

from pybullet_envs.env_bases import MJCFBaseBulletEnv
from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv

from linc.envs.pybullet.utils import TopCamera
from linc.envs.pybullet.scenes import RoughTerrainScene
from linc.envs.pybullet.robot_hexapod import HexapodRobot


TIME_STEP_FIXED = 0.0165
FRAME_SKIP = 4

REWARD_THRSH = 20
VEL_THRSH = .0005


class HexapodBaseEnv(WalkerBaseBulletEnv):
    """
    Ant walking agent.
    Basic environment with initialisation and finalisation methods added.
    """

    robot = None

    def __init__(self, render=False):
        WalkerBaseBulletEnv.__init__(self, self.robot, render)
        self.walk_target_x = 0
        self.walk_target_y = 0
        # Camera info
        self.camera_info = {'camera': {'distance': 12, 'yaw': -0, 'pitch': -89},
                            'lookat': [0, 0, 0]}
        self.camera = TopCamera(self)
        self._render_width = 640
        self._render_height = 480
        # Additional environment info
        self.param_ranges = np.vstack([
            self.action_space.low,
            self.action_space.high]).T
        self.env_info = dict(
            num_targets=1,
            num_obstacles=0,
            wall_geoms=None,
            ball_geom=None,
            target_info=[{'xy': (REWARD_THRSH, 0)}],
            # agent_ranges= # [[-5, 5], [-5, 5]], # [-10, 10], [-10, 10]
            agent_ranges=[[-3, 3], [-3, 3]],
            ball_ranges=None)

    def _get_info_dict(self, state=None):
        hull_pos = self.robot.body_xyz
        hull_angles = self.robot.body_rpy
        contact_info = state[np.array([24, 25, 26, 27])] \
            if state is not None else []
        # contact_info = self.robot.feet_contact
        velocity_info = self.robot_body.speed()
        b_angle = self.jdict['hip_1'].current_relative_position()[0]
        f_angle = self.jdict['hip_4'].current_relative_position()[0]
        rel_angle = b_angle - f_angle
        info_dict = dict(
            position=np.hstack([hull_pos, hull_angles]),
            position_aux=np.hstack([contact_info, rel_angle, velocity_info]),
            velocity=velocity_info,
            angle=hull_angles)
        return info_dict

    def initialize(self, seed_task, **kwargs):
        # Restart seed
        self.seed(seed_task)
        self.action_space.seed(seed_task)
        # Restart environment
        state = self.reset()
        info_dict = self._get_info_dict(state)
        return state, info_dict['position'], info_dict['position_aux']

    def finalize(self, rew_list, **kwargs):
        outcome = -1  # 0 if reward_len >= REWARD_THRSH else -1
        return np.array([outcome, np.sum(rew_list)])

    def reset(self):
        if (self.stateId >= 0):
            # print("restoreState self.stateId:", self.stateId)
            self._p.restoreState(self.stateId)
        state = MJCFBaseBulletEnv.reset(self)
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
        (self.parts, self.jdict,
            self.ordered_joints, self.robot_body) = self.robot.addToScene(
                self._p, self.stadium_scene.ground_plane_mjcf)
        # Properly add all terrain blocks
        terrain_obj = [tk for tk in self.parts.keys() if 'terrain' in tk]
        for tk in terrain_obj:
            del self.parts[tk]
        self.ground_ids = set()
        for f in self.foot_ground_object_names:
            for b in self.parts[f].bodies:
                new_id = (b, self.parts[f].bodyPartIndex)
                if new_id not in self.ground_ids:
                    self.ground_ids.add(new_id)
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
        if (self.stateId < 0):
            self.stateId = self._p.saveState()
            # print("saving state self.stateId:", self.stateId)
        self.prev_action = np.zeros(self.action_space.shape[0])
        self.robot.ground_ids = self.ground_ids
        return state

    def step(self, action):
        state, rew, done, _ = super().step(action)
        info_dict = self._get_info_dict(state)
        done = done or np.linalg.norm(info_dict['velocity']) <= VEL_THRSH
        return state, rew, done, info_dict

    def render(self, mode='human', close=False):
        if mode == "human":
            self.isRender = True
        if self.physicsClientId >= 0:
            self.camera_adjust()
        if mode != "rgb_array":
            return np.array([])
        if (self.physicsClientId >= 0):
            view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self.camera_info['lookat'],
                roll=0,
                upAxisIndex=2,
                **self.camera_info['camera'])
            proj_matrix = self._p.computeProjectionMatrixFOV(
                fov=60,
                aspect=float(self._render_width) / self._render_height,
                nearVal=0.1,
                farVal=100.0)
            (_, _, px, _, _) = self._p.getCameraImage(
                width=self._render_width,
                height=self._render_height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
            self._p.configureDebugVisualizer(
                self._p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
        else:
            px = np.array(
                [[[255, 255, 255, 255]] * self._render_width] *
                self._render_height,
                dtype=np.uint8)
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(
            np.array(px),
            (self._render_height, self._render_width, -1))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array


class HexapodEnv(HexapodBaseEnv):
    """Hexapod walking agent on flat terrain."""

    robot = HexapodRobot(init_robot=[0, 0, .5], scale=1)


class HexapodTerrainEnv(HexapodBaseEnv):
    """Hexapod walking environment agent on rough terrain."""

    robot = HexapodRobot(init_robot=[0, 0, .5], scale=1)

    def create_single_player_scene(self, bullet_client):
        """Override standard terrain"""
        self.stadium_scene = RoughTerrainScene(
            bullet_client=bullet_client,
            gravity=9.8,
            timestep=TIME_STEP_FIXED / FRAME_SKIP,
            frame_skip=FRAME_SKIP)
        return self.stadium_scene
