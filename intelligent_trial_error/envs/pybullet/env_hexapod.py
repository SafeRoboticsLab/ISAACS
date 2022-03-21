
"""
Hexapod robot and environment definition.

References:
- https://github.com/resibots/pyhexapod

"""


import pdb
import logging

import os
import math
import subprocess
import numpy as np
import pybullet as pb
import pybullet_utils.bullet_client as bc
import pybullet_data
import time

logger = logging.getLogger(__name__)


class HexapodEnvironment(object):

    def __init__(
        self,
        envtype,
        payload=None,
        damaged_legs=(),
        dt=1. / 240.,  # the default for pybullet (see doc)
        control_dt=0.05,
        episode_len=3.,
        gui=False,
        rough_terrain = False,
        terrain_height = 0.05,
        terrain_friction = 1.0,
        video_output_file=None
    ):
        # Simulator setup
        self.envtype = envtype
        self.payload = payload
        self.payload_max = 10
        # Impose damaged_legs string encoding
        self.damaged_legs = [
            dl.encode('UTF-8') if isinstance(dl, str) else dl
            for dl in damaged_legs]
        if self.envtype == 'normal':
            self.payload = None
            self.damaged_legs = []

        self._gen_urdf(envtype)

        self.GRAVITY = -9.81
        self.dt = dt
        self.control_dt = control_dt
        self.n_timesteps = int(episode_len / self.dt)
        # Controller settings
        self.control_period = int(control_dt / dt)
        self.safety_turnover = True
        self.video_output_file = video_output_file
        self.use_gui = gui or self.video_output_file is not None
        """
        the final target velocity is computed using:
        kp * ( erp * (desiredPosition - currentPosition) / dt)
         + currentVelocity + kd * (m_desiredVelocity - currentVelocity)
        Here we set kp to be likely to reach the target position
        in the time between two calls of the controller
        """
        self.kp = 1. / 12.  # * self.control_period
        self.kd = 0.4
        # The initial desired position for the joints
        self.angles = np.zeros(12)

        # Setup the GUI (disable the useless windows)
        if self.use_gui:
            self.camera_info = {
                'camera': {'distance': 12, 'yaw': -0, 'pitch': -89},
                'lookat': [0, 0, 0]}
            self._render_width = 640
            self._render_height = 480
            self.physics = bc.BulletClient(connection_mode=pb.GUI)
            self.physics.configureDebugVisualizer(
                pb.COV_ENABLE_GUI, 0)
            self.physics.configureDebugVisualizer(
                pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            self.physics.configureDebugVisualizer(
                pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            self.physics.configureDebugVisualizer(
                pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            # self.physics.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)
            self.physics.resetDebugVisualizerCamera(
                cameraDistance=1,
                cameraYaw=20,
                cameraPitch=-20,
                cameraTargetPosition=[1, -0.5, 0.8])
        else:
            self.physics = bc.BulletClient(connection_mode=pb.DIRECT)
        self.ffmpeg_pipe = None

        # Pybullet settings
        self.physics.setAdditionalSearchPath(pybullet_data.getDataPath())
        # self.reset()
        self.rough_terrain = rough_terrain
        self.terrain_height = terrain_height
        self.terrain_friction = terrain_friction

        if self.rough_terrain:
            self._gen_terrain()

    def _gen_terrain(self):
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

        terrainShape = self.physics.createCollisionShape(
            shapeType=self.physics.GEOM_HEIGHTFIELD,
            meshScale=[.1, .1, 1.0],
            heightfieldTextureScaling=(numHeightfieldRows - 1) / 2,
            heightfieldData = heightfieldData,
            numHeightfieldRows=numHeightfieldRows,
            numHeightfieldColumns=numHeightfieldColumns)
        terrain = self.physics.createMultiBody(0, terrainShape)
        self.physics.resetBasePositionAndOrientation(
            terrain, [0, 0, 0.0], [0, 0, 0, 1])
        self.physics.changeDynamics(terrain, -1, lateralFriction=self.terrain_friction)
        self.physics.changeVisualShape(terrain, -1, rgbaColor=[0.2, 0.8, 0.8, 1])

    def _gen_urdf(self, envtype):
        urdf_name = 'spirit40.urdf'
        urdf_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'urdf/{}'.format(urdf_name))
        fin = open(urdf_path)
        urdf = fin.read()
        fin.close()

        payload_definition = ""
        
        # Update payload scale
        if self.envtype == 'payload':
            payload_definition = """<!-- ADDING PAYLOAD -->
                <joint name="body_payload" type="fixed">
                    <parent link="base_link"/>
                    <child link="payload"/>
                </joint>
                <link name="payload">
                    <visual>
                    <origin rpy="0 0 0" xyz="0.0 0.0 @ZLOCATION"/>  <!-- z = height/2 + 0.02 -->
                    <geometry>
                        <box size="0.15 0.15 @HEIGHT"/>
                    </geometry>
                    <material name="Yellow"/>
                    </visual>
                    <collision>
                    <origin rpy="0 0 0" xyz="0.00 0.00 @ZLOCATION"/> <!-- z = height/2 + 0.02 -->
                    <geometry>
                        <box size="0.15 0.15 @HEIGHT"/>
                    </geometry>
                    </collision>
                    <inertial>
                    <!-- CENTER OF MASS -->
                    <origin rpy="0 0 0" xyz="0.0 0.0 @ZCOM"/>
                    <mass value="@MASS"/>
                    <!-- box inertia: 1/12*m(y^2+z^2), ... -->
                    <inertia ixx="@IXX" ixy="0" ixz="0" iyy="@IYY" iyz="0" izz="@IZZ"/>
                    </inertial>
                </link>
                <!-- END PAYLOAD --> 
            """

            payload_mass = self.payload * self.payload_max
            payload_height = self.payload * 0.45
            payload_definition = payload_definition \
                .replace('@MASS', str(payload_mass) ) \
                .replace('@HEIGHT', str(payload_height) ) \
                .replace('@ZLOCATION', str(payload_height/2+0.02) ) \
                .replace('@IXX', str(1/12 * payload_mass * (0.15*0.15 + payload_height * payload_height))) \
                .replace('@IYY', str(1/12 * payload_mass * (0.15*0.15 + payload_height * payload_height))) \
                .replace('@IZZ', str(1/12 * payload_mass * (2 * 0.15*0.15)))
        
        elif self.envtype == 'spring':
            payload_mass = self.payload * self.payload_max
            payload_blocks = 5 # 10 is too many
            payload_active = int(self.payload * payload_blocks)
            block_mass = self.payload_max / float(payload_blocks)

            remaining = 0
            if not payload_active * block_mass == payload_mass:
                remaining = payload_mass - payload_active * block_mass
                
            fixed_joint_definition = """
                <joint name="@JOINTNAME" type="fixed">
                    <parent link="@PARENTLINK"/>
                    <child link="@CHILDLINK"/>
                </joint>
            """

            revolute_joint_definition = """
                <joint name="@JOINTNAME" type="revolute">
                    <parent link="@PARENTLINK"/>
                    <child link="@CHILDLINK"/>
                    <limit effort="10.0" lower="-0.5" upper="0.5" velocity="10.0"/>
                    <origin rpy="0 0 0.0" xyz="0.0 0.0 @ZLOCATION"/>
                    <axis xyz="@AXIS"/>
                    <dynamics damping="0.05"/>
                </joint>
            """

            block_definition = """
                <link name="@LINKNAME">
                    <visual>
                    <origin rpy="0 0 0" xyz="0.0 0.0 @ZLOCATION"/>  <!-- z = height/2 + 0.02 -->
                    <geometry>
                        <box size="0.15 0.15 0.05"/>
                    </geometry>
                    <material name="@COLOR"/>
                    </visual>
                    <collision>
                    <origin rpy="0 0 0" xyz="0.00 0.00 @ZLOCATION"/> <!-- z = height/2 + 0.02 -->
                    <geometry>
                        <box size="0.15 0.15 0.05"/>
                    </geometry>
                    </collision>
                    <inertial>
                    <!-- CENTER OF MASS -->
                    <origin rpy="0 0 0" xyz="0.0 0.0 0.025"/>
                    <mass value="@MASS"/>
                    <!-- box inertia: 1/12*m(y^2+z^2), ... -->
                    <inertia ixx="@IXX" ixy="0" ixz="0" iyy="@IYY" iyz="0" izz="@IZZ"/>
                    </inertial>
                </link>
            """

            payload_definition = "<!-- ADDING PAYLOAD -->\n"
            if remaining > 0.01:
                total_range = payload_active +1
            else:
                total_range = payload_active
            for se_ in range(total_range):
                if not se_:
                    #the fixed joint should always be the first element of the definition
                    payload_definition += fixed_joint_definition\
                        .replace("@JOINTNAME","body_payload")\
                        .replace("@PARENTLINK","base_link")\
                        .replace("@CHILDLINK",f"payload_block_{se_}")
                else:
                    payload_definition += revolute_joint_definition\
                        .replace("@JOINTNAME",f"payload{se_-1}_payload{se_}")\
                        .replace("@PARENTLINK",f"payload_block_{se_-1}")\
                        .replace("@CHILDLINK",f"payload_block_{se_}")\
                        .replace("@ZLOCATION","0.05")\
                        .replace("@AXIS","0 1 0" if se_ % 2 else "1 0 0")
                current_block_mass = block_mass if se_ != payload_active else remaining
                payload_definition += block_definition\
                    .replace("@LINKNAME",f"payload_block_{se_}")\
                    .replace("@COLOR", "Yellow" if se_ % 2 else "Green" )\
                    .replace("@ZLOCATION","0.05")\
                    .replace("@MASS",str(block_mass))\
                    .replace('@IXX', str(1/12 * block_mass * (0.15*0.15 + 0.05 * 0.05))) \
                    .replace('@IYY', str(1/12 * block_mass * (0.15*0.15 + 0.05 * 0.05))) \
                    .replace('@IZZ', str(1/12 * block_mass * (2 * 0.15*0.15)))
            payload_definition += " \n<!-- END PAYLOAD -->"
                                                         
        self.urdf = urdf_path[:-5]+"_tmp.urdf"
        fout=open(self.urdf,"w")
        fout.write(urdf.replace("<!-- PAYLOAD_PLACEHOLDER -->",payload_definition))
        fout.close()

        
    def _make_joint_list(self, botId):
        # joint_names = [
        #     b'body_leg_0', b'leg_0_1_2', b'leg_0_2_3',
        #     b'body_leg_1', b'leg_1_1_2', b'leg_1_2_3',
        #     b'body_leg_2', b'leg_2_1_2', b'leg_2_2_3',
        #     b'body_leg_3', b'leg_3_1_2', b'leg_3_2_3',
        #     b'body_leg_4', b'leg_4_1_2', b'leg_4_2_3',
        #     b'body_leg_5', b'leg_5_1_2', b'leg_5_2_3',
        # ]
        joint_names = [
            b'hip0', b'upper0', b'lower0',
            b'hip1', b'upper1', b'lower1',
            b'hip2', b'upper2', b'lower2',
            b'hip3', b'upper3', b'lower3'
        ]
        joint_list = []
        for n in joint_names:
            joint_found = False
            for joint in range(self.physics.getNumJoints(botId)):
                name = self.physics.getJointInfo(botId, joint)[12]
                if name == n and name not in self.damaged_legs:
                    joint_list += [joint]
                    joint_found = True
                elif name == n and name in self.damaged_legs:
                    self.physics.changeVisualShape(
                        1, joint, rgbaColor=[0.5, 0.5, 0.5, 0.5])
            if joint_found is False:
                # if the joint is not here (aka broken leg case) put 1000
                # joint_list += [1000]
                continue
        return joint_list

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
                logger.error(
                    "VideoRecorder encoder exited with status {}".format(ret))

        if self.video_output_file is not None:
            camera = self.physics.getDebugVisualizerCamera()
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
            camera = self.physics.getDebugVisualizerCamera()
            img = pb.getCameraImage(
                camera[0], camera[1],
                renderer=pb.ER_BULLET_HARDWARE_OPENGL)
            self.ffmpeg_pipe.stdin.write(img[2].tobytes())

    def soft_reset(self):
        """Reset without killing the robot"""
        self.t = 0
        self.i = 0
        return self.get_state()

    def reset(self):
        """Reset the simulation and the environment variables."""
        self.t = 0
        self.i = 0
        self.physics.resetSimulation()
        self.physics.setGravity(0, 0, self.GRAVITY)
        self.physics.setTimeStep(self.dt)
        self.physics.setPhysicsEngineParameter(fixedTimeStep=self.dt)
        self.planeId = self.physics.loadURDF("plane.urdf")
        if self.rough_terrain:
            self._gen_terrain()

        start_pos = [0, 0, 0.4]
        start_orientation = self.physics.getQuaternionFromEuler([0., 0., 0.])
        self.botId = self.physics.loadURDF(
            self.urdf, start_pos, start_orientation)

        # Setup used joints and exclude controlling damaged ones
        self.joint_list = self._make_joint_list(self.botId)

        # Bulled links number corresponding to the legs
        # self.leg_link_ids = [17, 14, 2, 5, 8, 11]
        self.leg_link_ids = [3, 7, 11, 15]
        self.descriptor = {3: [], 7: [], 11: [], 15: []}
        self.ground_contacts = np.zeros_like(self.leg_link_ids)

        # Initialise the frame saver
        if self.video_output_file is not None:
            self._init_frames()

        initial_joint_value = [
            0, 0.75, 1.45,
            0, 0.75, 1.45,
            0, 0.75, 1.45,
            0, 0.75, 1.45
        ]

        # initial_joint_value = [
        #     0, 0.4, 1.0, 
        #     0, 0.4, 1.0, 
        #     0, 0.4, 1.0, 
        #     0, 0.4, 1.0
        # ]

        # Put the hexapod on the ground (gently)
        self.physics.setRealTimeSimulation(0)
        for joint in range(len(self.joint_list)):
            self.physics.resetJointState(
                self.botId, self.joint_list[joint], initial_joint_value[joint])
        
        for t in range(0, 100):
            self.physics.stepSimulation()
            self.physics.setGravity(0, 0, self.GRAVITY)
        
        jointFrictionForce = 1
        for joint in range(len(self.joint_list)):
            self.physics.setJointMotorControl2(
                self.botId,
                self.joint_list[joint],
                pb.POSITION_CONTROL,
                force=jointFrictionForce)

#       self.physics.restoreState(self._init_state)
        return self.get_state()

    def destroy(self):
        """Properly close the simulation."""
        try:
            self.physics.disconnect()
        except pb.error as e:
            logger.error("Warning (destructor of simulator):", e)
        try:
            if self.video_output_file is not None:
                self.ffmpeg_pipe.stdin.close()
                self.ffmpeg_pipe.stderr.close()
                ret = self.ffmpeg_pipe.wait()
        except Exception as e:
            logger.error(
                "VideoRecorder encoder exited with status {}".format(ret))

    def step(self, action):
        """Perform evnironment step based on the given action."""
        # Controller actions are only considered after control period
        if self.i % self.control_period == 0:
            # self.angles = controller.step(self)
            self.angles = action
        self.i += 1

        self.physics.resetDebugVisualizerCamera(cameraDistance = 2, cameraYaw = 45, cameraPitch = -20, cameraTargetPosition = self.physics.getBasePositionAndOrientation(self.botId)[0])
        # Save current simulation frame
        if self.video_output_file is not None:
            self._save_frames()

        # Check if roll pitch are not too high
        error = False
        self.euler = self.physics.getEulerFromQuaternion(self.get_pos()[1])
        if self.safety_turnover:
            if ((abs(self.euler[1]) >= math.pi / 2) or
                    (abs(self.euler[0]) >= math.pi / 2)):
                error = True
                # print("\nSAFETY TURNOVER ERROR!", error)

        # Execute the joint movements
        missing_joint_count = 0
        j = 0
        for joint in self.joint_list:
            if joint == 1000:
                missing_joint_count += 1
            else:
                info = self.physics.getJointInfo(self.botId, joint)
                lower_limit = info[8]
                upper_limit = info[9]
                max_force = info[10]
                max_velocity = info[11]
                pos = min(max(lower_limit, self.angles[j]), upper_limit)
                self.physics.setJointMotorControl2(
                    self.botId,
                    joint,
                    pb.POSITION_CONTROL,
                    positionGain=self.kp,
                    velocityGain=self.kd,
                    targetPosition=pos,
                    force=max_force,
                    maxVelocity=max_velocity)
            j += 1

        # Get contact points between robot and world plane
        contact_points = self.physics.getContactPoints(self.botId, self.planeId)
        link_ids = []  # list of links in contact with the ground plane
        if len(contact_points) > 0:
            for cn in contact_points:
                linkid = cn[3]  # robot link id in contact with world plane
                if linkid not in link_ids:
                    link_ids.append(linkid)
        for l in self.leg_link_ids:
            cns = self.descriptor[l]
            if l in link_ids:
                cns.append(1)
            else:
                cns.append(0)
            self.descriptor[l] = cns

        self.ground_contacts = np.zeros_like(self.leg_link_ids)
        for i, ll in enumerate(self.leg_link_ids):
            if ll in link_ids:
                self.ground_contacts[i] = 1

        # Add the gravity force!                                    ???
        self.physics.setGravity(0, 0, self.GRAVITY)

        # Run the step of the simulation
        self.physics.stepSimulation()
        self.t += self.dt
        
        if self.use_gui:
            time.sleep(self.dt)

        return error, self.get_state()

    def get_state(self):
        """Combine the elements of the state vector."""
        state = np.concatenate([
            [self.t],
            list(sum(self.get_pos(), ())),
            self.get_joints_positions(),
            self.ground_contacts])
        return state

    def get_pos(self):
        """
        Return the position list of 3 floats and orientation as list of
        4 floats in [x,y,z,w] order. Use pb.getEulerFromQuaternion to convert
        the quaternion to Euler if needed.
        """
        return self.physics.getBasePositionAndOrientation(self.botId)

    def get_joints_positions(self):
        """Return the actual position in the physics engine"""
        p = np.zeros(len(self.joint_list))
        i = 0
        # be careful that the joint_list is not necessarily in the same order as
        # in bullet (see make_joint_list)
        for joint in self.joint_list:
            if joint != 1000:
                p[i] = self.physics.getJointState(self.botId, joint)[0]
                i += 1
        return p
