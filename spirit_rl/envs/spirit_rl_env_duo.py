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

class SpiritRLEnvDuo(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, device="cpu", mode='normal', doneType='toEnd', 
                seed=0, gui = False, verbose = False, force = 0.0, forceResetTime = 20, forceRandom = False, rotateReset = False, heightReset = False, payload = 0.0, envtype = None, control_dt = 0.05, 
                episode_len = 3., dt = 1./240., video_output_file = None):
        
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

        # dict to hold robot id to access robot
        self.robots = {
            "safety": None,
            "performance": None
        }
        
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

        self.angles = {
            "safety": np.zeros(12), 
            "performance": np.zeros(12)
        }

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

        self.reset()

    def step(self, action):
        # Feed action to robot and get observation of robot's state
        self.angles["safety"] = action["safety"]
        
        if self.i % self.control_period == 0:
            self.angles["performance"] = action["performance"]
        
        self.i += 1
        
        self._save_frames()

        spirit_old_observation = {
            "safety": self.robots["safety"].get_observation(),
            "performance": self.robots["performance"].get_observation()
        }

        self.robots["safety"].apply_position(self.angles["safety"])
        self.robots["performance"].apply_position(self.angles["performance"])

        # apply random force to robot
        if self.elapsedForceApplied > self.FORCE_APPLIED_RESET:
            self.elapsedForceApplied = 0
            self.force_applied_force_vector = np.random.rand(3) * self.force
            self.force_applied_position_vector = np.random.rand(3)
        else:
            self.elapsedForceApplied += 1
        p.applyExternalForce(self.robots["safety"].robot, -1, self.force_applied_force_vector, self.force_applied_position_vector, p.LINK_FRAME)
        p.applyExternalForce(self.robots["performance"].robot, -1, self.force_applied_force_vector, self.force_applied_position_vector, p.LINK_FRAME)

        p.setGravity(0, 0, self.GRAVITY)
        p.stepSimulation()

        l_x = self.target_margin()
        g_x = self.safety_margin()

        fail = {
            "safety": g_x["safety"] > 0,
            "performance": g_x["performance"] > 0
        }

        success = {
            "safety": l_x["safety"] <= 0,
            "performance": l_x["performance"] <= 0
        }

        if self.gui:
            time.sleep(self.dt)

        spirit_observation = {
            "safety": self.robots["safety"].get_observation(),
            "performance": self.robots["performance"].get_observation()
        }

        cost = dict()
        done = dict()
        ob = dict()
        ob_old = dict()
        info = dict()

        for robot in self.robots.keys():
            if fail[robot]:
                cost[robot] = self.penalty
            elif success[robot]:
                cost[robot] = self.reward
            else:
                if self.costType == 'dense_ell':
                    cost[robot] = l_x[robot]
                elif self.costType == 'dense_ell_g':
                    cost[robot] = l_x[robot] + g_x[robot]
                elif self.costType == 'sparse':
                    cost[robot] = 0.
                elif self.costType == 'max_ell_g':
                    cost[robot] = max(l_x[robot], g_x[robot])
                else:
                    cost[robot] = 0.

            #= `done` signal
            # done = fail
            if self.doneType == 'toEnd':
                print("WARNING: Function has not been implemented")
                pass
            elif self.doneType == 'fail':
                done[robot] = fail[robot]
            elif self.doneType == 'TF':
                done[robot] = fail[robot] or success[robot]
            else:
                raise ValueError("invalid doneType")

            ob[robot] = np.array(spirit_observation[robot], dtype=np.float32)
            ob_old[robot] = np.array(spirit_old_observation[robot], dtype=np.float32)
            joint_position = np.array(self.robots[robot].get_joint_position(), dtype = np.float32)

            ob[robot] = np.concatenate((ob[robot], ob_old[robot], action[robot], joint_position), axis=0)

            #= `info`
            if done[robot] and self.doneType == 'fail':
                info[robot] = {"g_x": self.penalty, "l_x": l_x[robot]}
            else:
                info[robot] = {"g_x": g_x[robot], "l_x": l_x[robot]}
        
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

    def reset(self):
        self.i = 0
        p.resetSimulation(self.client)
        p.setGravity(0, 0, self.GRAVITY)
        p.setTimeStep(self.dt)
        p.setPhysicsEngineParameter(fixedTimeStep = self.dt)
        Plane(self.client)

        if self.heightReset:
            height = 0.5 + np.random.rand()
        else:
            height = 0.5

        if self.rotateReset:
            rotate = p.getQuaternionFromEuler(np.random.rand(3) * np.pi * 2.0)
        else:
            rotate = p.getQuaternionFromEuler([0.0, 0.0, 0.0])

        self.robots["safety"] = Spirit(self.client, height, rotate, self.envtype, self.payload, self.payload_max, oy = 0.5)
        self.robots["performance"] = Spirit(self.client, height, rotate, self.envtype, self.payload, self.payload_max, oy = -0.5)
        
        random_joint_value = self.get_random_joint_value()

        # Put the hexapod on the ground (gently)
        p.setRealTimeSimulation(0)
        # jointFrictionForce = 1
        # for r in self.robots.keys():
        #     for joint in range(p.getNumJoints(self.robots[r].robot)):
        #         p.setJointMotorControl2(
        #             self.robots[r].robot,
        #             joint,
        #             p.POSITION_CONTROL,
        #             force=jointFrictionForce)

        for t in range(0, 100):
            p.stepSimulation()
            p.setGravity(0, 0, self.GRAVITY)

        self.robots["safety"].reset(random_joint_value)
        self.robots["performance"].reset(random_joint_value)

        spirit_observation = {
            "safety": self.robots["safety"].get_observation(),
            "performance": self.robots["performance"].get_observation()
        }

        # create a random force applied on the robot
        self.elapsedForceApplied = 0
        self.force_applied_force_vector = np.random.rand(3) * self.force
        self.force_applied_position_vector = np.random.rand(3)

        initial_value = {
            "safety": np.concatenate((np.array(spirit_observation["safety"], dtype=np.float32), np.array(spirit_observation["safety"], dtype=np.float32), random_joint_value, random_joint_value), axis = 0),
            "performance": np.concatenate((np.array(spirit_observation["performance"], dtype=np.float32), np.array(spirit_observation["performance"], dtype=np.float32), random_joint_value, random_joint_value), axis = 0)
        }

        self._init_frames()

        return initial_value

    def render(self):
        pass

    def close(self):
        p.disconnect(self.client)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def safety_margin(self):
        return {
            "safety": self.robots["safety"].safety_margin(),
            "performance": self.robots["performance"].safety_margin()
        }

    def target_margin(self):
        return {
            "safety": self.robots["safety"].target_margin(),
            "performance": self.robots["performance"].target_margin()
        }

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