
"""
Author:         Nemanja Rakicevic
Date:           September 2020
Description:
                Useful Scene classes for PyBullet environments.
                - BallScene
                - BoundedBallScene
                - TableScene
                - SurgeryScene

                TODO - fix masses of the objects in SurgeryScene
"""

import os
import numpy as np

from pybullet_envs.scene_abstract import Scene



class StadiumScene_V2(object):
    """
    Custom-made field with rough terrain.
    Used for walking on uneven terrain.
    """

    zero_at_running_strip_start_line = True
    # if False, the center of coordinates (0,0,0) will be in the middle
    # of the stadium
    stadium_halflen = 105 * 0.25  # FOOBALL_FIELD_HALFLEN
    stadium_halfwidth = 50 * 0.25  # FOOBALL_FIELD_HALFWID
    stadiumLoaded = 0
    multiplayer = False

    def __init__(self,
                 bullet_client,
                 gravity,
                 timestep,
                 frame_skip,
                 **kwargs):
        self._p = bullet_client
        self.gravity = gravity
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.num_solver_iterations = 5
        self.dt = self.timestep * self.frame_skip
        self.clean_everything()

    def clean_everything(self):
        self._p.resetSimulation()
        self._p.setPhysicsEngineParameter(
            fixedTimeStep=self.timestep * self.frame_skip,
            numSolverIterations=self.num_solver_iterations,
            numSubSteps=self.frame_skip)
        self._p.setGravity(0, 0, -self.gravity)
        self.stadiumLoaded = 0

    def episode_restart(self, bullet_client):
        self._p = bullet_client
        self.clean_everything()

        if (self.stadiumLoaded == 0):
            self.stadiumLoaded = 1
            # Add stadium with walls
            filename = os.path.join(
                os.path.dirname(__file__),
                "assets/objects/plane_normal.sdf")
            self.ground_plane_mjcf = self._p.loadSDF(filename)
            # Adjust the dynamics properties of the terrain
            for obj in self.ground_plane_mjcf:
                self._p.changeDynamics(
                    obj, -1,
                    lateralFriction=0.8,
                    restitution=0.5)

    def actor_introduce(self, robot):
        return

    def global_step(self):
        self._p.stepSimulation()


class LightTerrainScene_V2(object):
    """
    Custom-made field with rough terrain. Includes restart.
    Used for walking on uneven terrain.
    """

    zero_at_running_strip_start_line = True
    # if False, the center of coordinates (0,0,0) will be in the middle
    # of the stadium
    stadium_halflen = 105 * 0.25  # FOOBALL_FIELD_HALFLEN
    stadium_halfwidth = 50 * 0.25  # FOOBALL_FIELD_HALFWID
    stadiumLoaded = 0
    multiplayer = False

    def __init__(self,
                 bullet_client,
                 gravity,
                 timestep,
                 frame_skip,
                 **kwargs):
        self._p = bullet_client
        self.gravity = gravity
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.num_solver_iterations = 5
        self.dt = self.timestep * self.frame_skip
        self.clean_everything()

    def actor_introduce(self, robot):
        return

    def clean_everything(self):
        self._p.resetSimulation()
        self._p.setPhysicsEngineParameter(
            fixedTimeStep=self.timestep * self.frame_skip,
            numSolverIterations=self.num_solver_iterations,
            numSubSteps=self.frame_skip)
        self._p.setGravity(0, 0, -self.gravity)
        self.stadiumLoaded = 0

    def episode_restart(self, bullet_client):
        self._p = bullet_client
        self.clean_everything()

        if (self.stadiumLoaded == 0):
            self.stadiumLoaded = 1
            # Add stadium with walls
            filename = os.path.join(
                os.path.dirname(__file__),
                "assets/objects/plane_normal.sdf")
            self.ground_plane_mjcf = self._p.loadSDF(filename)
            self._p.changeDynamics(
                self.ground_plane_mjcf[0], -1,
                lateralFriction=0.8,
                restitution=0.5)
            # Add terrain blocks
            file_block_1 = os.path.join(
                os.path.dirname(__file__),
                "assets/objects/terrain_block_1.sdf")
            file_block_2 = os.path.join(
                os.path.dirname(__file__),
                "assets/objects/terrain_block_05.sdf")
            orient_block = self._p.getQuaternionFromEuler([0, 0, np.pi / 4])
            for i in range(-4, 5):
                for j in range(-4, 5):
                    tb1_id = self._p.loadSDF(file_block_1)
                    self._p.resetBasePositionAndOrientation(
                        tb1_id[0],
                        [2 * i + 0.5, 2 * j + 0.5, 0.05], [0, 0, 0, 1])
                    tb2_id = self._p.loadSDF(file_block_2)
                    self._p.resetBasePositionAndOrientation(
                        tb2_id[0],
                        [2 * i - 0.5, 2 * j - 0.5, 0.025], orient_block)
                    # Adjust the dynamics properties of the terrain
                    self._p.changeDynamics(
                        tb1_id[0], -1,
                        lateralFriction=0.8,
                        restitution=0.5)
                    self._p.changeDynamics(
                        tb2_id[0], -1,
                        lateralFriction=0.8,
                        restitution=0.5)
                    self.ground_plane_mjcf += (tb1_id[0], tb2_id[0])

    def global_step(self):
        self._p.stepSimulation()


class RoughTerrainScene(Scene):
    """
    Custom-made field with rough terrain.
    Used for walking on uneven terrain.
    """

    zero_at_running_strip_start_line = True
    # if False, the center of coordinates (0,0,0) will be in the middle
    # of the stadium
    stadium_halflen = 105 * 0.25  # FOOBALL_FIELD_HALFLEN
    stadium_halfwidth = 50 * 0.25  # FOOBALL_FIELD_HALFWID
    stadiumLoaded = 0
    multiplayer = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def episode_restart(self, bullet_client):
        self._p = bullet_client
        # cpp_world.clean_everything()
        Scene.episode_restart(self, bullet_client)
        if (self.stadiumLoaded == 0):
            self.stadiumLoaded = 1
            # Add stadium with walls
            filename = os.path.join(
                os.path.dirname(__file__),
                "assets/objects/plane_normal.sdf")
            self.ground_plane_mjcf = self._p.loadSDF(filename)
            self._p.changeDynamics(
                self.ground_plane_mjcf[0], -1,
                lateralFriction=0.8,
                restitution=0.5)
            # Add terrain blocks
            file_block_1 = os.path.join(
                os.path.dirname(__file__),
                "assets/objects/terrain_block_1.sdf")
            # file_block_2 = os.path.join(
            #     os.path.dirname(__file__),
            #     "assets/objects/terrain_block_2.sdf")
            file_block_05 = os.path.join(
                os.path.dirname(__file__),
                "assets/objects/terrain_block_05.sdf")
            orient_block = self._p.getQuaternionFromEuler([0, 0, np.pi / 4])
            for i in range(-4, 5):
                for j in range(-4, 5):
                    tb1_id = self._p.loadSDF(file_block_1)
                    self._p.resetBasePositionAndOrientation(
                        tb1_id[0],
                        [2 * i + 0.5, 2 * j + 0.5, 0.05], [0, 0, 0, 1])
                    # tb2_id = self._p.loadSDF(file_block_2)
                    # self._p.resetBasePositionAndOrientation(
                    #     tb2_id[0],
                    #     [2 * i - 0.5, 2 * j - 0.5, 0.1], orient_block)
                    tb05_id = self._p.loadSDF(file_block_05)
                    self._p.resetBasePositionAndOrientation(
                        tb05_id[0],
                        [2 * i - 0.5, 2 * j - 0.5, 0.025], orient_block)
                    # Adjust the dynamics properties of the terrain
                    self._p.changeDynamics(
                        tb1_id[0], -1,
                        lateralFriction=0.8,
                        restitution=0.5)
                    self._p.changeDynamics(
                        tb05_id[0], -1,
                        lateralFriction=0.8,
                        restitution=0.5)
                    self.ground_plane_mjcf += (tb1_id[0], tb05_id[0])


class RoughTerrain2DScene(Scene):
    """
    Custom-made field with rough terrain, only along 1 dimension.
    Used for walking on uneven terrain.
    """

    zero_at_running_strip_start_line = True
    # if False, the center of coordinates (0,0,0) will be in the middle
    # of the stadium
    stadium_halflen = 105 * 0.25  # FOOBALL_FIELD_HALFLEN
    stadium_halfwidth = 50 * 0.25  # FOOBALL_FIELD_HALFWID
    stadiumLoaded = 0
    multiplayer = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def episode_restart(self, bullet_client):
        self._p = bullet_client
        # cpp_world.clean_everything()
        Scene.episode_restart(self, bullet_client)
        if (self.stadiumLoaded == 0):
            self.stadiumLoaded = 1
            # Add stadium with walls
            filename = os.path.join(
                os.path.dirname(__file__),
                "assets/objects/plane_normal.sdf")
            self.ground_plane_mjcf = self._p.loadSDF(filename)
            self._p.changeDynamics(
                self.ground_plane_mjcf[0], -1,
                lateralFriction=0.8,
                restitution=0.5)
            # Add terrain blocks
            file_block_1 = os.path.join(
                os.path.dirname(__file__),
                "assets/objects/terrain_block_1.sdf")
            for i in range(-9, 10):
                tb_id = self._p.loadSDF(file_block_1)
                self._p.resetBasePositionAndOrientation(
                    tb_id[0],
                    [2 * i + 0.45, 0, 0.05], [0, 0, 0, 1])
                # Adjust the dynamics properties of the terrain
                self._p.changeDynamics(
                    tb_id[0], -1,
                    lateralFriction=0.8,
                    restitution=0.5)
                self.ground_plane_mjcf += (tb_id[0], )

