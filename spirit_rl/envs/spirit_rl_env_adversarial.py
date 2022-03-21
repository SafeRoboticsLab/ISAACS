import gym
import numpy as np 
import pybullet as p
import math
from spirit_rl.resources.spirit import Spirit
from spirit_rl.resources.plane import Plane
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time
from scipy.stats import truncnorm

from pybullet_debugger import pybulletDebug
import subprocess

class SpiritRLEnvAdversarial(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, device="cpu", mode='normal', doneType='toEnd', 
                seed=0, gui = False, verbose = False, force = 0.0, forceResetTime = 20, forceRandom = False, rotateReset = False, heightReset = False, payload = 0.0, envtype = None, control_dt = 0.05, 
                episode_len = 3., dt = 1./240., video_output_file = None, terrain = "normal", terrainHeight = 0.05, terrainFriction = 1.0):
        
        self.np_random, _ = gym.utils.seeding.np_random()

        # verbose information of training process
        self.verbose = verbose
        # set this to True if want to see animated render
        self.video_output_file = video_output_file
        self.gui = gui or self.video_output_file is not None

        # magnitude of the force to be applied to the robots
        self.force = float(force)
        # this variable ensures that the force applied will be reset every FORCE_APPLIED_RESET steps
        self.elapsedForceApplied = 0
        self.FORCE_APPLIED_RESET = forceResetTime

        self.rotateReset = rotateReset
        self.heightReset = heightReset

        self.robot = None
        
        self.rendered_img = None

        self.force_applied_force_vector = None
        self.force_applied_position_vector = None
        self.forceRandom = forceRandom

        self.doneType = doneType
        
        # Cost Params
        self.targetScaling = 1.
        self.safetyScaling = 1.
        self.penalty = 1.
        self.reward = -1.
        self.costType = 'sparse'

        # linc
        self.payload = payload
        self.payload_max = 10
        self.envtype = envtype     

        self.GRAVITY = -9.81
        self.dt = 1./240.
        self.i = 0
        self.dt = dt
        self.control_dt = control_dt
        self.n_timesteps = int(episode_len / self.dt)
        # Controller settings
        self.control_period = int(control_dt / dt)

        self.angles = np.zeros(12)

        self.ffmpeg_pipe = None
        if self.gui:
            # Setup the GUI (disable the useless windows)
            self.camera_info = {
                'camera': {'distance': 12, 'yaw': -0, 'pitch': -89},
                'lookat': [0, 0, 0]}
            self._render_width = 640
            self._render_height = 480
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(
                p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(
                p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            p.configureDebugVisualizer(
                p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(
                p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.resetDebugVisualizerCamera(
                cameraDistance=1,
                cameraYaw=20,
                cameraPitch=-20,
                cameraTargetPosition=[1, -0.5, 0.8])
        else:
            self.client = p.connect(p.DIRECT)

        p.setTimeStep(self.dt, self.client)
        p.setPhysicsEngineParameter(fixedTimeStep = self.dt)
        p.setRealTimeSimulation(0)
        self.debugger = pybulletDebug()

        # rollout
        self.rollout_client = None
        self.rollout_robot = None
        self.rolloutElapsedForceApplied = 0
        self.rollout_force_applied_force_vector = None
        self.rollout_force_applied_position_vector = None

        # test
        self.test_client = None
        self.test_robot = None
        self.testElapsedForceApplied = 0
        self.test_force_applied_force_vector = None
        self.test_force_applied_position_vector = None

        # terrain
        self.terrain = terrain
        self.terrain_height = terrainHeight
        self.terrain_friction = terrainFriction

        self.reset()
    
    def _gen_terrain(self, client):
        heightPerturbationRange = self.terrain_height
        numHeightfieldRows = 256
        numHeightfieldColumns = 256

        terrainShape = 0
        heightfieldData = [0] * numHeightfieldRows * numHeightfieldColumns

        heightPerturbationRange = heightPerturbationRange
        for j in range(int(numHeightfieldColumns / 2)):
            for i in range(int(numHeightfieldRows / 2)):
                height = np.random.uniform(0, heightPerturbationRange)
                heightfieldData[2 * i +
                                        2 * j * numHeightfieldRows] = height
                heightfieldData[2 * i + 1 +
                                        2 * j * numHeightfieldRows] = height
                heightfieldData[2 * i + (2 * j + 1) *
                                        numHeightfieldRows] = height
                heightfieldData[2 * i + 1 + (2 * j + 1) *
                                        numHeightfieldRows] = height

        terrainShape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[.1, .1, 1.0],
            heightfieldTextureScaling=(numHeightfieldRows - 1) / 2,
            heightfieldData = heightfieldData,
            numHeightfieldRows=numHeightfieldRows,
            numHeightfieldColumns=numHeightfieldColumns,
            physicsClientId = client)
        
        terrain = p.createMultiBody(0, terrainShape, physicsClientId = client)

        p.resetBasePositionAndOrientation(
            terrain, [0, 0, 0.0], [0, 0, 0, 1], physicsClientId = client)
        
        p.changeDynamics(terrain, -1, lateralFriction=self.terrain_friction, physicsClientId = client)
        p.changeVisualShape(terrain, -1, rgbaColor=[0.2, 0.8, 0.8, 1], physicsClientId = client)

    def test_reset(self):
        if self.test_client is None:
            self.test_client = p.connect(p.DIRECT)
        p.resetSimulation(physicsClientId = self.test_client)

        # p.configureDebugVisualizer(
        #     p.COV_ENABLE_GUI, 0, physicsClientId = self.test_client)
        # p.configureDebugVisualizer(
        #     p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId = self.test_client)
        # p.configureDebugVisualizer(
        #     p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId = self.test_client)
        # p.configureDebugVisualizer(
        #     p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId = self.test_client)
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        # p.resetDebugVisualizerCamera(
        #     cameraDistance=1,
        #     cameraYaw=20,
        #     cameraPitch=-20,
        #     cameraTargetPosition=[1, -0.5, 0.8], physicsClientId = self.test_client)
        
        p.setGravity(0, 0, self.GRAVITY, physicsClientId = self.test_client)
        p.setTimeStep(self.dt, physicsClientId = self.test_client)
        p.setPhysicsEngineParameter(fixedTimeStep = self.dt, physicsClientId = self.test_client)
        Plane(self.test_client)

        if self.heightReset:
            height = 0.3 + np.random.rand()
        else:
            height = 0.3

        if self.rotateReset:
            rotate = p.getQuaternionFromEuler(np.random.rand(3) * np.pi * 2.0)
        else:
            rotate = p.getQuaternionFromEuler([0.0, 0.0, 0.0])

        self.test_robot = Spirit(self.test_client, height, rotate, self.envtype, self.payload, self.payload_max)
        
        random_joint_value = self.get_random_joint_value()

        # jointFrictionForce = 1
        # # reset joint state
        for joint in range(p.getNumJoints(self.test_robot.robot, physicsClientId = self.test_client)):
            current_joint_position, current_joint_velocity, _, _ = p.getJointState(self.robot.robot, joint, physicsClientId = self.client)
            p.resetJointState(self.test_robot.robot, joint, current_joint_position, targetVelocity = current_joint_velocity, physicsClientId = self.test_client)
        #     p.setJointMotorControl2(
        #         self.test_robot.robot,
        #         joint,
        #         p.POSITION_CONTROL,
        #         force=jointFrictionForce,
        #         physicsClientId = self.test_client)

        self.test_robot.reset(random_joint_value)
        self.test_robot.apply_position(random_joint_value)

        # for t in range(0, 100):
        #     p.stepSimulation(physicsClientId = self.test_client)
        #     p.setGravity(0, 0, self.GRAVITY, physicsClientId = self.test_client)

        spirit_observation = self.test_robot.get_observation()

        # create a random force applied on the robot
        # self.testElapsedForceApplied = 0
        # if self.forceRandom:
        #     self.test_force_applied_force_vector = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)]) * self.force
        # else:
        #     self.test_force_applied_force_vector = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-50, 5)]) * self.force
        # self.test_force_applied_position_vector = np.array([np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(0, 0.5)])

        initial_value = np.concatenate((np.array(spirit_observation, dtype=np.float32), np.array(spirit_observation, dtype=np.float32), random_joint_value, random_joint_value), axis = 0)

        return initial_value

    def test_step(self, action):
        disturbance_action = action[:6]
        robot_action = action[6:]

        # Feed action to robot and get observation of robot's state
        spirit_old_observation = self.test_robot.get_observation()
        self.test_robot.apply_position(robot_action)

        # apply random force to robot
        # if self.testElapsedForceApplied > self.FORCE_APPLIED_RESET:
        #     self.testElapsedForceApplied = 0
        #     if self.forceRandom:
        #         self.test_force_applied_force_vector = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)]) * self.force
        #     else:
        #         self.test_force_applied_force_vector = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-50, 5)]) * self.force
        #     self.test_force_applied_position_vector = np.array([np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(0, 0.5)])
        # else:
        #     self.testElapsedForceApplied += 1
        # p.applyExternalForce(self.test_robot.robot, -1, self.test_force_applied_force_vector, self.test_force_applied_position_vector, p.LINK_FRAME, physicsClientId = self.test_client)

        p.applyExternalForce(self.test_robot.robot, -1, disturbance_action[:3], disturbance_action[3:], p.LINK_FRAME, physicsClientId = self.test_client)

        p.setGravity(0, 0, self.GRAVITY, physicsClientId = self.test_client)
        p.stepSimulation(physicsClientId = self.test_client)

        l_x = self.target_margin(self.test_robot)
        g_x = self.safety_margin(self.test_robot)

        fail = g_x > 0
        success = l_x <= 0

        if self.gui:
            time.sleep(self.dt)

        spirit_observation = self.test_robot.get_observation()

        if fail:
            cost = self.penalty
        elif success:
            cost = self.reward
        else:
            if self.costType == 'dense_ell':
                cost = l_x
            elif self.costType == 'dense_ell_g':
                cost = l_x + g_x
            elif self.costType == 'sparse':
                cost = 0.
            elif self.costType == 'max_ell_g':
                cost = max(l_x, g_x)
            else:
                cost = 0.

        #= `done` signal
        # done = fail
        if self.doneType == 'toEnd':
            print("WARNING: Function has not been implemented")
            pass
        elif self.doneType == 'fail':
            done = fail
        elif self.doneType == 'TF':
            done = fail or success
        else:
            raise ValueError("invalid doneType")

        ob = np.array(spirit_observation, dtype = np.float32)
        ob_old = np.array(spirit_old_observation, dtype = np.float32)
        joint_position = np.array(self.test_robot.get_joint_position(), dtype = np.float32)

        ob = np.concatenate((ob, ob_old, robot_action, joint_position), axis=0)

        #= `info`
        if done and self.doneType == 'fail':
            info = {"g_x": self.penalty, "l_x": l_x}
        else:
            info = {"g_x": g_x, "l_x": l_x}

        if self.verbose:
            print("\rg_x:\t{:.2f}\tl_x:\t{:.2f}".format(g_x, l_x), end = "")
        
        return ob, cost, done, info

    def rollout_reset(self):
        if self.rollout_client is None:
            self.rollout_client = p.connect(p.DIRECT)
        p.resetSimulation(physicsClientId = self.rollout_client)

        # p.configureDebugVisualizer(
        #     p.COV_ENABLE_GUI, 0, physicsClientId = self.rollout_client)
        # p.configureDebugVisualizer(
        #     p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId = self.rollout_client)
        # p.configureDebugVisualizer(
        #     p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId = self.rollout_client)
        # p.configureDebugVisualizer(
        #     p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId = self.rollout_client)
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        # p.resetDebugVisualizerCamera(
        #     cameraDistance=1,
        #     cameraYaw=20,
        #     cameraPitch=-20,
        #     cameraTargetPosition=[1, -0.5, 0.8], physicsClientId = self.rollout_client)
        
        p.setGravity(0, 0, self.GRAVITY, physicsClientId = self.rollout_client)
        p.setTimeStep(self.dt, physicsClientId = self.rollout_client)
        p.setPhysicsEngineParameter(fixedTimeStep = self.dt, physicsClientId = self.rollout_client)
        Plane(self.rollout_client)

        if self.terrain == "rough":
            self._gen_terrain(self.rollout_client)

        current_pos, current_ang = p.getBasePositionAndOrientation(self.robot.robot, physicsClientId = self.client)
        self.rollout_robot = Spirit(self.rollout_client, current_pos[2], current_ang, self.envtype, self.payload, self.payload_max)
        
        # jointFrictionForce = 1
        # reset joint state
        for joint in range(p.getNumJoints(self.rollout_robot.robot, physicsClientId = self.rollout_client)):
            current_joint_position, current_joint_velocity, _, _ = p.getJointState(self.robot.robot, joint, physicsClientId = self.client)
            p.resetJointState(self.rollout_robot.robot, joint, current_joint_position, targetVelocity = current_joint_velocity, physicsClientId = self.rollout_client)
            # p.setJointMotorControl2(
            #     self.rollout_robot.robot,
            #     joint,
            #     p.POSITION_CONTROL,
            #     force=jointFrictionForce,
            #     physicsClientId = self.rollout_client)

        # reset velocity of the robot
        current_linear_velocity, current_angular_velocity = p.getBaseVelocity(self.robot.robot, physicsClientId = self.client)
        p.resetBaseVelocity(self.rollout_robot.robot, linearVelocity = current_linear_velocity, angularVelocity = current_angular_velocity, physicsClientId = self.rollout_client)

        current_joint_value = self.rollout_robot.get_joint_position()
        spirit_observation = self.rollout_robot.get_observation()

        # create a random force applied on the robot
        # self.rolloutElapsedForceApplied = 0
        # if self.forceRandom:
        #     self.rollout_force_applied_force_vector = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)]) * self.force
        # else:
        #     self.rollout_force_applied_force_vector = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-50, 5)]) * self.force
        # self.rollout_force_applied_position_vector = np.array([np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(0, 0.5)])

        initial_value = np.concatenate((np.array(spirit_observation, dtype=np.float32), np.array(spirit_observation, dtype=np.float32), current_joint_value, current_joint_value), axis = 0)

        return initial_value

    def rollout_step(self, action):
        disturbance_action = action[:6]
        robot_action = action[6:]
        # Feed action to robot and get observation of robot's state
        spirit_old_observation = self.rollout_robot.get_observation()
        self.rollout_robot.apply_position(robot_action)

        # apply random force to robot
        # if self.rolloutElapsedForceApplied > self.FORCE_APPLIED_RESET:
        #     self.rolloutElapsedForceApplied = 0
        #     if self.forceRandom:
        #         self.rollout_force_applied_force_vector = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)]) * self.force
        #     else:
        #         self.rollout_force_applied_force_vector = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-50, 5)]) * self.force
        #     self.rollout_force_applied_position_vector = np.array([np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(0, 0.5)])
        # else:
        #     self.rolloutElapsedForceApplied += 1
        # p.applyExternalForce(self.rollout_robot.robot, -1, self.rollout_force_applied_force_vector, self.rollout_force_applied_position_vector, p.LINK_FRAME, physicsClientId = self.rollout_client)

        p.applyExternalForce(self.test_robot.robot, -1, disturbance_action[:3], disturbance_action[3:], p.LINK_FRAME, physicsClientId = self.test_client)

        p.setGravity(0, 0, self.GRAVITY, physicsClientId = self.rollout_client)
        p.stepSimulation(physicsClientId = self.rollout_client)

        l_x = self.target_margin(self.rollout_robot)
        g_x = self.safety_margin(self.rollout_robot)

        fail = g_x > 0
        success = l_x <= 0

        if self.gui:
            time.sleep(self.dt)

        spirit_observation = self.rollout_robot.get_observation()

        if fail:
            cost = self.penalty
        elif success:
            cost = self.reward
        else:
            if self.costType == 'dense_ell':
                cost = l_x
            elif self.costType == 'dense_ell_g':
                cost = l_x + g_x
            elif self.costType == 'sparse':
                cost = 0.
            elif self.costType == 'max_ell_g':
                cost = max(l_x, g_x)
            else:
                cost = 0.

        #= `done` signal
        # done = fail
        if self.doneType == 'toEnd':
            print("WARNING: Function has not been implemented")
            pass
        elif self.doneType == 'fail':
            done = fail
        elif self.doneType == 'TF':
            done = fail or success
        else:
            raise ValueError("invalid doneType")

        ob = np.array(spirit_observation, dtype = np.float32)
        ob_old = np.array(spirit_old_observation, dtype = np.float32)
        joint_position = np.array(self.rollout_robot.get_joint_position(), dtype = np.float32)

        ob = np.concatenate((ob, ob_old, robot_action, joint_position), axis=0)

        #= `info`
        if done and self.doneType == 'fail':
            info = {"g_x": self.penalty, "l_x": l_x}
        else:
            info = {"g_x": g_x, "l_x": l_x}
        
        return ob, cost, done, info

    def step(self, action, shielding = False):
        # Feed action to robot and get observation of robot's state
        disturbance_action = action[:6]
        robot_action = action[6:]

        if shielding:
            self.angles = robot_action
        elif self.i % self.control_period == 0:
            self.angles = robot_action
        
        self.i += 1

        self._save_frames()

        spirit_old_observation = self.robot.get_observation()
        self.robot.apply_position(self.angles)

        # apply random force to robot
        # if self.elapsedForceApplied > self.FORCE_APPLIED_RESET:
        #     self.elapsedForceApplied = 0
        #     if self.forceRandom:
        #         self.force_applied_force_vector = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)]) * self.force
        #     else:
        #         self.force_applied_force_vector = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-50, 5)]) * self.force
        #     self.force_applied_position_vector = np.array([np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(0, 0.5)])
        # else:
        #     self.elapsedForceApplied += 1
        # p.applyExternalForce(self.robot.robot, -1, self.force_applied_force_vector, self.force_applied_position_vector, p.LINK_FRAME)

        # use adversarial agent to control force applied
        p.applyExternalForce(self.robot.robot, -1, disturbance_action[:3], disturbance_action[3:], p.LINK_FRAME)

        p.setGravity(0, 0, self.GRAVITY)
        p.stepSimulation(physicsClientId = self.client)

        l_x = self.target_margin(self.robot)
        g_x = self.safety_margin(self.robot)

        fail = g_x > 0
        success = l_x <= 0

        if self.gui:
            time.sleep(self.dt)

        spirit_observation = self.robot.get_observation()

        if fail:
            cost = self.penalty
        elif success:
            cost = self.reward
        else:
            if self.costType == 'dense_ell':
                cost = l_x
            elif self.costType == 'dense_ell_g':
                cost = l_x + g_x
            elif self.costType == 'sparse':
                cost = 0.
            elif self.costType == 'max_ell_g':
                cost = max(l_x, g_x)
            else:
                cost = 0.

        #= `done` signal
        # done = fail
        if self.doneType == 'toEnd':
            print("WARNING: Function has not been implemented")
            pass
        elif self.doneType == 'fail':
            done = fail
        elif self.doneType == 'TF':
            done = fail or success
        else:
            raise ValueError("invalid doneType")

        ob = np.array(spirit_observation, dtype = np.float32)
        ob_old = np.array(spirit_old_observation, dtype = np.float32)
        joint_position = np.array(self.robot.get_joint_position(), dtype = np.float32)

        ob = np.concatenate((ob, ob_old, robot_action, joint_position), axis=0)

        # set camera
        self.debugger.cam_and_robotstates(self.robot.robot)

        #= `info`
        if done and self.doneType == 'fail':
            info = {"g_x": self.penalty, "l_x": l_x}
        else:
            info = {"g_x": g_x, "l_x": l_x}
        
        if self.verbose:
            print("\rg_x:\t{:.2f}\tl_x:\t{:.2f}".format(g_x, l_x), end = "")
        
        return ob, cost, done, info

    def get_random_joint_value(self):
        # return (
        #     0.0 + np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
        #     0.75 + np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
        #     1.45 + np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
        #     0.0 + np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
        #     0.75 + np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
        #     1.45 + np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
        #     0.0 + np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
        #     0.75 + np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
        #     1.45 + np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
        #     0.0 + np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
        #     0.75 + np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
        #     1.45 + np.random.uniform(-math.pi * 0.5, math.pi * 0.5)
        # )
        return (
            np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
            np.random.uniform(0, math.pi),
            np.random.uniform(0, math.pi),
            np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
            np.random.uniform(0, math.pi),
            np.random.uniform(0, math.pi),
            np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
            np.random.uniform(0, math.pi),
            np.random.uniform(0, math.pi),
            np.random.uniform(-math.pi * 0.5, math.pi * 0.5),
            np.random.uniform(0, math.pi),
            np.random.uniform(0, math.pi)
        )

    def reset(self):
        self.i = 0
        p.resetSimulation(physicsClientId = self.client)
        p.setGravity(0, 0, self.GRAVITY, physicsClientId = self.client)
        p.setTimeStep(self.dt, physicsClientId = self.client)
        p.setPhysicsEngineParameter(fixedTimeStep = self.dt, physicsClientId = self.client)
        Plane(self.client)
        
        if self.terrain == "rough":
            self._gen_terrain(self.client)

        if self.heightReset:
            height = 0.3 + np.random.rand()
        else:
            height = 0.3

        if self.rotateReset:
            rotate = p.getQuaternionFromEuler(np.random.rand(3) * np.pi * 2.0)
        else:
            rotate = p.getQuaternionFromEuler([0.0, 0.0, 0.0])

        self.robot = Spirit(self.client, height, rotate, self.envtype, self.payload, self.payload_max)
        
        random_joint_value = self.get_random_joint_value()

        # Put the hexapod on the ground (gently)
        p.setRealTimeSimulation(0)
        # jointFrictionForce = 1
        # for joint in range(p.getNumJoints(self.robot.robot)):
        #     p.setJointMotorControl2(
        #         self.robot.robot,
        #         joint,
        #         p.POSITION_CONTROL,
        #         force=jointFrictionForce)

        self.robot.reset(random_joint_value)
        self.robot.apply_position(random_joint_value)

        # for t in range(0, 100):
        #     p.stepSimulation()
        #     p.setGravity(0, 0, self.GRAVITY)
                    
        spirit_observation = self.robot.get_observation()

        # create a random force applied on the robot
        # self.elapsedForceApplied = 0
        # if self.forceRandom:
        #     self.force_applied_force_vector = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)]) * self.force
        # else:
        #     self.force_applied_force_vector = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-50, 5)]) * self.force
        # self.force_applied_position_vector = np.array([np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(0, 0.5)])

        initial_value = np.concatenate((np.array(spirit_observation, dtype=np.float32), np.array(spirit_observation, dtype=np.float32), random_joint_value, random_joint_value), axis = 0)

        self._init_frames()

        return initial_value

    def render(self):
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))

        # Base information
        robot_id, client_id = self.robot.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in
                    p.getBasePositionAndOrientation(robot_id, client_id)]

        pos[0] += -5
        pos[2] += 3
        ori = (0.00011645204953981455, 0.14789470218498033, -0.0007823537532204507, 0.9890027964708397)

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Display image
        frame = p.getCameraImage(200, 200, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (200, 200, 4))
        self.rendered_img.set_data(frame)
        plt.draw()
        plt.pause(.00001)

    def close(self):
        p.disconnect(self.client)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def safety_margin(self, robot):
        return robot.safety_margin()

    def target_margin(self, robot):
        return robot.target_margin()

    def _init_frames(self):
        """
        Initialize the pipe for streaming frames to the video file.
        Warning: video slows down the simulation!
        """
        if self.ffmpeg_pipe is not None:
            try:
                if self.video_output_file is not None:
                    self.ffmpeg_pipe.stdin.close()
                    self.ffmpeg_pipe.stderr.close()
                    ret = self.ffmpeg_pipe.wait()
            except Exception as e:
                print(
                    "VideoRecorder encoder exited with status {}".format(ret))

        if self.video_output_file is not None:
            camera = p.getDebugVisualizerCamera()
            command = [
                'ffmpeg',
                '-y',
                '-r', str(24),
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', '{}x{}'.format(camera[0], camera[1]),
                '-pix_fmt', 'rgba',
                '-i', '-',
                '-an',
                '-vcodec', 'mpeg4',
                '-vb', '20M',
                self.video_output_file]
            #print(command)
            self.ffmpeg_pipe = subprocess.Popen(
                command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def _save_frames(self):
        """
        Write frame at each step.
        24 FPS dt =1/240 : every 10 frames
        """
        if self.video_output_file is not None and \
                self.i % (int(1. / (self.dt * 24))) == 0:
            camera = p.getDebugVisualizerCamera()
            img = p.getCameraImage(
                camera[0], camera[1],
                renderer=p.ER_BULLET_HARDWARE_OPENGL)
            self.ffmpeg_pipe.stdin.write(img[2].tobytes())

    def destroy(self):
        """Properly close the simulation."""
        try:
            p.disconnect()
        except p.error as e:
            print("Warning (destructor of simulator):", e)
        try:
            if self.video_output_file is not None:
                self.ffmpeg_pipe.stdin.close()
                self.ffmpeg_pipe.stderr.close()
                ret = self.ffmpeg_pipe.wait()
        except Exception as e:
            print(
                "VideoRecorder encoder exited with status {}".format(ret))