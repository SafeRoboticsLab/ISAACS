
import pdb

import os
import time
import math
from timeit import default_timer as timer
import subprocess 
import time
import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data

from linc.algo.controllers import HexapodController


class HexapodSimulator:

    def __init__(
        self,
        gui=True,
        urdf=(
            os.path.dirname(os.path.abspath(__file__)) +
            '/urdf/pexod.urdf'),
        dt=1. / 240.,  # the default for pybullet (see doc)
        control_dt=0.05,
        video='./nemanja_test_video'
    ):

        self.GRAVITY = -9.81
        self.dt = dt
        self.control_dt = control_dt
        # we call the controller every control_period steps
        self.control_period = int(control_dt / dt)
        self.t = 0
        self.i = 0
        self.safety_turnover = True
        self.video_output_file = video

        # the final target velocity is computed using:
        # kp*(erp*(desiredPosition-currentPosition)/dt)+currentVelocity+kd*(m_desiredVelocity - currentVelocity).
        # here we set kp to be likely to reach the target position
        # in the time between two calls of the controller
        self.kp = 1./12.# * self.control_period
        self.kd = 0.4
        # the desired position for the joints
        self.angles = np.zeros(18)
        # setup the GUI (disable the useless windows)
        if gui:
            self.physics = bc.BulletClient(connection_mode=p.GUI)
            self.physics.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
            self.physics.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            self.physics.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            self.physics.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            self.physics.resetDebugVisualizerCamera(cameraDistance=1,
                                                    cameraYaw=20,
                                                    cameraPitch=-20,
                                                    cameraTargetPosition=[1, -0.5, 0.8])
        else:
            self.physics = bc.BulletClient(connection_mode=p.DIRECT)

        self.physics.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.physics.resetSimulation()
        self.physics.setGravity(0,0,self.GRAVITY)
        self.physics.setTimeStep(self.dt)
        self.physics.setPhysicsEngineParameter(fixedTimeStep=self.dt)
        self.planeId = self.physics.loadURDF("plane.urdf")

        start_pos = [0,0,0.15]
        start_orientation = self.physics.getQuaternionFromEuler([0.,0,0])
        self.botId = self.physics.loadURDF(urdf, start_pos, start_orientation)
        self.joint_list = self._make_joint_list(self.botId)

        # bullet links number corresponding to the legs
        self.leg_link_ids = [17, 14, 2, 5, 8, 11]
        self.descriptor = {17 : [], 14 : [], 2 : [], 5 : [], 8 : [], 11 : []}

        # video makes things much slower
        if (video != ''):
            self._stream_to_ffmpeg(self.video_output_file)

        # put the hexapod on the ground (gently)
        self.physics.setRealTimeSimulation(0)
        jointFrictionForce=1
        for joint in range (self.physics.getNumJoints(self.botId)):
            self.physics.setJointMotorControl2(self.botId, joint,
                p.POSITION_CONTROL,
                force=jointFrictionForce)
        for t in range(0, 100):
            self.physics.stepSimulation()
            self.physics.setGravity(0,0, self.GRAVITY)


    def destroy(self):
        try:
            self.physics.disconnect()
            if self.video_output_file != '':
                self.ffmpeg_pipe.stdin.close()
                self.ffmpeg_pipe.stderr.close()
                self.ffmpeg_pipe.wait()
        except p.error as e:
            print("Warning (destructor of simulator):", e)


    def reset(self):
        assert(0), "not working for now"
        self.t = 0
        self.physics.resetSimulation()
#       self.physics.restoreState(self._init_state)
        

    def get_pos(self):
        """
        Returns the position list of 3 floats and orientation as list of 4 floats in [x,y,z,w] order.
        Use p.getEulerFromQuaternion to convert the quaternion to Euler if needed.
        """
        return self.physics.getBasePositionAndOrientation(self.botId)

    def step(self, controller):
        if self.i % self.control_period == 0:
            # self.angles = controller.step(self)
            self.angles = controller.get_action([self.t])
        self.i += 1
        
        # 24 FPS dt =1/240 : every 10 frames
        if self.video_output_file != '' and self.i % (int(1. / (self.dt * 24))) == 0: 
            camera = self.physics.getDebugVisualizerCamera()
            img = p.getCameraImage(camera[0], camera[1], renderer=p.ER_BULLET_HARDWARE_OPENGL)
            self.ffmpeg_pipe.stdin.write(img[2].tobytes())

        #Check if roll pitch are not too high
        error = False
        self.euler = self.physics.getEulerFromQuaternion(self.get_pos()[1])
        if(self.safety_turnover):
            if((abs(self.euler[1]) >= math.pi/2) or (abs(self.euler[0]) >= math.pi/2)):
                error = True

        # move the joints
        missing_joint_count = 0
        j = 0
        for joint in self.joint_list:
            if(joint==1000):
                missing_joint_count += 1
            else:
                info = self.physics.getJointInfo(self.botId, joint)
                lower_limit = info[8]
                upper_limit = info[9]
                max_force = info[10]
                max_velocity = info[11]
                pos = min(max(lower_limit, self.angles[j]), upper_limit)
                self.physics.setJointMotorControl2(self.botId, joint,
                    p.POSITION_CONTROL,
                    positionGain=self.kp,
                    velocityGain=self.kd,
                    targetPosition=pos,
                    force=max_force,
                    maxVelocity=max_velocity)
            j += 1

        #Get contact points between robot and world plane
        contact_points = self.physics.getContactPoints(self.botId,self.planeId)
        link_ids = [] #list of links in contact with the ground plane
        if(len(contact_points) > 0):
            for cn in contact_points:
                linkid= cn[3] #robot link id in contact with world plane
                if linkid not in link_ids:
                    link_ids.append(linkid)
        for l in self.leg_link_ids:
            cns = self.descriptor[l]
            if l in link_ids:
                cns.append(1)
            else:
                cns.append(0)
            self.descriptor[l] = cns

        # don't forget to add the gravity force!
        self.physics.setGravity(0, 0, self.GRAVITY)

        # finally, step the simulation
        self.physics.stepSimulation()
        self.t += self.dt
        return error

    def get_joints_positions(self):
        ''' return the actual position in the physics engine'''
        p = np.zeros(len(self.joint_list))
        i = 0
        # be careful that the joint_list is not necessarily in the same order as 
        # in bullet (see make_joint_list)
        for joint in self.joint_list:
            p[i] = self.physics.getJointState(self.botId, joint)[0]
            i += 1
        return p


    def _stream_to_ffmpeg(self, fname):
        camera = self.physics.getDebugVisualizerCamera()
        command = ['ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec','rawvideo',
                '-s',  '{}x{}'.format(camera[0], camera[1]),
                '-pix_fmt', 'rgba',
                '-r', str(24),
                '-i', '-',
                '-an',
                '-vcodec', 'mpeg4',
                '-vb', '20M',
                fname]
        print(command)
        self.ffmpeg_pipe = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def _make_joint_list(self, botId):
        joint_names = [b'body_leg_0', b'leg_0_1_2', b'leg_0_2_3',
        b'body_leg_1', b'leg_1_1_2', b'leg_1_2_3',
        b'body_leg_2', b'leg_2_1_2', b'leg_2_2_3',
        b'body_leg_3', b'leg_3_1_2', b'leg_3_2_3',
        b'body_leg_4', b'leg_4_1_2', b'leg_4_2_3',
        b'body_leg_5', b'leg_5_1_2', b'leg_5_2_3',
        ]
        joint_list = []
        for n in joint_names:
            joint_found = False
            for joint in range (self.physics.getNumJoints(botId)):
                name = self.physics.getJointInfo(botId, joint)[1]
                if name == n:
                    joint_list += [joint]
                    joint_found = True
            if(joint_found==False):
                joint_list += [1000] #if the joint is not here (aka broken leg case) put 1000
        return joint_list


# for an unkwnon reason, connect/disconnect works only if this is a function
def test_ref_controller():
    # this the reference controller from Cully et al., 2015 (Nature)
    ctrl = [1, 0, 0.5, 0.25, 0.25, 0.5, 1, 0.5, 0.5, 0.25, 0.75, 0.5, 1, 0, 0.5, 0.25, 0.25, 0.5, 1, 0, 0.5, 0.25, 0.75, 0.5, 1, 0.5, 0.5, 0.25, 0.25, 0.5, 1, 0, 0.5, 0.25, 0.75, 0.5]
    simu = HexapodSimulator(gui=False)
    controller = HexapodController(ctrl)

    for i in range(0, int(3./simu.dt)): # seconds
        simu.step(controller)
    
    print("=>", simu.get_pos()[0])
    simu.destroy()


if __name__ == "__main__":
    # we do 10 simulations to get some statistics (perfs, reproducibility)
    for k in range(0, 1):
        t0 = time.perf_counter()
        test_ref_controller()# this needs to be in a sub-function...
        print(time.perf_counter() - t0, " ms")