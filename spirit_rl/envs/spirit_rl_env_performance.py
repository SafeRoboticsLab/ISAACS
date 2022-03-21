import gym
import numpy as np
from numpy.lib.function_base import angle 
import pybullet as p
import math
from spirit_rl.resources.spirit import Spirit
from spirit_rl.resources.plane import Plane
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time
from scipy.stats import truncnorm

from pybullet_debugger import pybulletDebug

class SpiritRLEnvPerformance(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, gui = False, verbose = False):
        self.np_random, _ = gym.utils.seeding.np_random()
        # verbose information of training process
        self.verbose = verbose
        # set this to True if want to see animated render
        self.gui = gui
        # self.gui = True
        # change this value to change the delay between frames
        self.elapsedTimeVerbose = 0.01
        
        self.robot = None
        self.goal = None
        
        self.done = False
        self.prev_dist_from_origin = None
        self.elapsedWalkingTime = 0
        self.rendered_img = None

        self.jointDirection = None

        if self.gui:
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

        p.setTimeStep(1/30, self.client)
        self.debugger = pybulletDebug()
        self.reset()

    def step(self, action):
        # Feed action to robot and get observation of robot's state
        robot_old_observation = self.robot.get_observation()
        self.robot.apply_position(action)
        p.stepSimulation()

        if self.gui:
            time.sleep(self.elapsedTimeVerbose)

        # previous_position = self.robot.previous_position
        robot_observation = self.robot.get_observation()

        # When robot moves forward, distance of robot from origin is X value
        dist_from_origin = robot_observation[0]
        dist_error = dist_from_origin - self.prev_dist_from_origin
        self.prev_dist_from_origin = dist_from_origin

        # EPISODE TERMINATION
        # 1. The height of the body is too close to ground
        if robot_observation[2] < 0.1:
            self.done = True
            if self.verbose:
                print("TERMINATED: Height too close")

        # 2. Roll, Pitch, Yaw angles are outside bound (+/– 0.1745, +/– 0.1745, and +/– 0.3491 rad, respectively)
        if robot_observation[3] > 0.35 or robot_observation[3] < -0.35:
            if self.verbose:
                print("TERMINATED: Roll")
            self.done = True

        if robot_observation[4] > 0.35 or robot_observation[4] < -0.35:
            if self.verbose:
                print("TERMINATED: Pitch")
            self.done = True
        
        if robot_observation[5] > 0.7 or robot_observation[5] < -0.7:
            if self.verbose:
                print("TERMINATED: Yaw")
            self.done = True
        
        # REWARD
        reward = 0

        # 1. Velocity, the faster the better the reward
        velocity_reward = robot_observation[6]
        reward = reward + velocity_reward
        if self.verbose:
            print("{}: {} for velocity".format(reward, velocity_reward))

        # 2. Constant reward for staying alive
        alive_reward = 25.0 * 1./30.
        reward = reward + alive_reward
        if self.verbose:
            print("{}: {} for staying alive".format(reward, alive_reward))

        # 3. Negative reward for height of robot deviation from 0.3
        height_diff = (robot_observation[2] - 0.3) ** 2
        height_forfeit = 50.0 * height_diff
        reward = reward - height_forfeit
        if self.verbose:
            print("{}: {} for height diff".format(reward, -height_forfeit))
        
        # 4. Negative reward for pitch angle strayed from desired value (0) 
        pitch_error = robot_observation[4] ** 2
        reward = reward - 20.0 * pitch_error

        # 5. Difference in joint value between this step and previous step
        angle_diff = np.array(self.robot.get_joint_position()) - np.array(action)
        reward = reward - 0.02 * angle_diff @ angle_diff
        if self.verbose:
            print("{}: {} for angle diff".format(reward, -angle_diff))
        
        if self.verbose and self.done:
            print("\n\n")

        ob = np.array(robot_observation, dtype=np.float32)
        ob_old = np.array(robot_old_observation, dtype=np.float32)
        joint_position = np.array(self.robot.get_joint_position(), dtype = np.float32)

        ob = np.concatenate((ob, ob_old, action, joint_position), axis=0)

        # set camera
        self.debugger.cam_and_robotstates(self.robot.robot)

        return ob, reward, self.done, dict()

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8)
        Plane(self.client)
        self.robot = Spirit(self.client, 0.5, p.getQuaternionFromEuler([0.0, 0.0, 0.0]))

        self.done = False

        robot_observation = self.robot.get_observation()

        self.prev_dist_from_origin = robot_observation[0]
        self.elapsedWalkingTime = 0

        random_joint_value = (
            (np.random.rand() - 0.5) * 0.707,
            np.random.rand() * 3.14,
            np.random.rand() * 3.14,
            (np.random.rand() - 0.5) * 0.707,
            np.random.rand() * 3.14,
            np.random.rand() * 3.14,
            (np.random.rand() - 0.5) * 0.707,
            np.random.rand() * 3.14,
            np.random.rand() * 3.14,
            (np.random.rand() - 0.5) * 0.707,
            np.random.rand() * 3.14,
            np.random.rand() * 3.14
        )

        self.robot.reset(random_joint_value)

        initial_value = np.concatenate((np.array(robot_observation, dtype=np.float32), np.array(robot_observation, dtype=np.float32), random_joint_value, random_joint_value), axis = 0)

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