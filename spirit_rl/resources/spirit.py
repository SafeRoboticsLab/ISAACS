import pybullet as p
import os
import math
import numpy as np

class Spirit:
    def __init__(self, client, height, orientation, 
        envtype = None, payload = 0, payload_max = 0, **kwargs):
        self.client = client
        self.urdf = "spirit40.urdf"
        self.height = height
        
        ox = 0
        oy = 0
        
        for key in kwargs.keys():
            if key == "ox":
                ox = kwargs["ox"]
            if key == "oy":
                oy = kwargs["oy"]
        
        if envtype != None:
            # TODO: create different env here
            # self._gen_urdf(envtype, payload, payload_max)
            pass
            
        f_name = os.path.join(os.path.dirname(__file__), self.urdf)

        self.robot = p.loadURDF(fileName = f_name, basePosition=[ox, oy, self.height], baseOrientation = orientation, physicsClientId = client)
        
        # self.joint_index = range(p.getNumJoints(self.robot))
        self.joint_index = self.make_joint_list()
        self.torque_gain = 10.0

    def get_ids(self):
        return self.robot, self.client

    def reset(self, position):
        for i in range(len(self.joint_index)):
            p.resetJointState(self.robot, self.joint_index[i], position[i], physicsClientId = self.client)

    def apply_position(self, action):
        for i in range(len(self.joint_index)):
            info = p.getJointInfo(self.robot, self.joint_index[i], physicsClientId = self.client)
            lower_limit = info[8]
            upper_limit = info[9]
            max_force = info[10]
            max_velocity = info[11]
            pos = min(max(lower_limit, action[i]), upper_limit)

            p.setJointMotorControl2(
                self.robot, 
                self.joint_index[i],
                p.POSITION_CONTROL, 
                targetPosition = pos, 
                positionGain=1./12.,
                velocityGain=0.4,
                force=max_force,
                maxVelocity=max_velocity, 
                physicsClientId = self.client
            )

    def get_observation(self):
        # Get the position and orientation of robot in the simulation
        pos, ang = p.getBasePositionAndOrientation(self.robot, self.client)
        ang = p.getEulerFromQuaternion(ang, physicsClientId = self.client)
        
        # ori = (math.cos(ang[0]), math.sin(ang[0]), math.cos(ang[1]), math.sin(ang[1]), math.cos(ang[2]), math.sin(ang[2]))
        
        # Get the velocity of robot
        vel = p.getBaseVelocity(self.robot, self.client)[0][:]

        # self.previous_position = pos[0:2]

        # Concatenate position, orientation, velocity
        # VALUE:
        # 0 - x value of body wrt to env
        # 1 - y value of body wrt to env
        # 2 - z value of body wrt to env
        # 3 - roll value, rad
        # 4 - pitch value, rad
        # 5 - yaw value, rad
        # 6 - x velocity of body wrt to env
        # 7 - y velocity of body wrt to env
        # 8 - z velocity of body wrt to env

        observation = (pos + ang + vel) # 3 + 3 + 3

        return observation # return observation size of 12

    def get_joint_position(self):
        joint_state = p.getJointStates(self.robot, jointIndices = self.joint_index, physicsClientId = self.client)
        position = [state[0] for state in joint_state]
        return position

    def get_joint_torque(self):
        joint_state = p.getJointStates(self.robot, jointIndices = self.joint_index, physicsClientId = self.client)
        torque = [state[3] for state in joint_state]
        return torque

    def safety_margin(self):
        """
        Safety margin of the robot. 
        If the robot gets too close to the ground, or if any of the knee touches the ground (within an error margin)
        """
        robot_observation = self.get_observation()
        # height, roll, pitch
        return max(0.2 - robot_observation[2], robot_observation[2] - 0.5, abs(robot_observation[3]) - math.pi * 0.0625, abs(robot_observation[4]) - math.pi * 0.0625)

    def target_margin(self):
        """
        Comparing the current stance of the robot with respect to the target stance, which is:
            joint_positions = [
                0, 1, 1.9, 
                0, 1, 1.9, 
                0, 1, 1.9, 
                0, 1, 1.9
            ]
            the index is w.r.t self.joint_index
        """
        target_stance = [
            0, 0.6, 1.45,
            0, 0.6, 1.45,
            0, 0.6, 1.45,
            0, 0.6, 1.45
        ]

        target_margin = np.array([0.3, 0.3, 0.25] * 4)
        
        current_stance = self.get_joint_position()
        stance_error = np.array(current_stance)  - np.array(target_stance)

        robot_observation = self.get_observation()
        vel_z = robot_observation[-1]

        return max(max(abs(stance_error) - target_margin), abs(vel_z) - 1.0)
        # return max(abs(stance_error) - target_margin)

    def make_joint_list(self):
        damaged_legs = []
        joint_names = [
            b'hip0', b'upper0', b'lower0',
            b'hip1', b'upper1', b'lower1',
            b'hip2', b'upper2', b'lower2',
            b'hip3', b'upper3', b'lower3'
        ]
        joint_list = []
        for n in joint_names:
            joint_found = False
            for joint in range(p.getNumJoints(self.robot, physicsClientId = self.client)):
                name = p.getJointInfo(self.robot, joint, physicsClientId = self.client)[12]
                if name == n and name not in damaged_legs:
                    joint_list += [joint]
                    joint_found = True
                elif name == n and name in damaged_legs:
                    p.changeVisualShape(
                        1, joint, rgbaColor=[0.5, 0.5, 0.5, 0.5], physicsClientId = self.client)
            if joint_found is False:
                # if the joint is not here (aka broken leg case) put 1000
                # joint_list += [1000]
                continue
        return joint_list

    def linc_get_joints_positions(self):
        """Return the actual position in the physics engine"""
        pos = np.zeros(len(self.joint_index))
        i = 0
        # be careful that the joint_list is not necessarily in the same order as
        # in bullet (see make_joint_list)
        for joint in self.joint_index:
            if joint != 1000:
                pos[i] = p.getJointState(self.robot, joint, physicsClientId = self.client)[0]
                i += 1
        return pos

    def linc_get_pos(self):
        """
        Return the position list of 3 floats and orientation as list of
        4 floats in [x,y,z,w] order. Use pb.getEulerFromQuaternion to convert
        the quaternion to Euler if needed.
        """
        return p.getBasePositionAndOrientation(self.robot, physicsClientId = self.client)
    
    def linc_get_ground_contacts(self):
        leg_link_ids = [17, 14, 2, 5, 8, 11]
        descriptor = {17: [], 14: [], 2: [], 5: [], 8: [], 11: []}
        ground_contacts = np.zeros_like(leg_link_ids)

        # Get contact points between robot and world plane
        contact_points = p.getContactPoints(self.robot, physicsClientId = self.client)
        link_ids = []  # list of links in contact with the ground plane
        if len(contact_points) > 0:
            for cn in contact_points:
                linkid = cn[3]  # robot link id in contact with world plane
                if linkid not in link_ids:
                    link_ids.append(linkid)
        for l in leg_link_ids:
            cns = descriptor[l]
            if l in link_ids:
                cns.append(1)
            else:
                cns.append(0)
            descriptor[l] = cns

        for i, ll in enumerate(leg_link_ids):
            if ll in link_ids:
                ground_contacts[i] = 1
        
        return ground_contacts

    def linc_get_state(self, t):
        """Combine the elements of the state vector."""
        state = np.concatenate([
            [t],
            list(sum(self.linc_get_pos(), ())),
            self.linc_get_joints_positions(),
            self.linc_get_ground_contacts()])
        return state

    def get_joint_position_wrt_body(self, alpha, beta):
        # get the joint position wrt to body (which joint is closer to the body), from this, the further a joint away from the body, the closer the joint to ground
        # leg length l1 = l2 = 0.206
        # alpha is the angle between upper link and body (upper joint)
        # beta is the angle between lower link and upper link (lower joint)
        # ------ O -- BODY ---------> HEAD
        #      |  \\         | h2
        #      |   \\       B 
        #     h1    \\    //
        #      |      A //
        #----------- GROUND --------
        # for all legs, upper joint moving forward to the head will be to 3.14 (180 degree)
        l1 = 0.206
        l2 = 0.206
        h1 = math.sin(math.pi - alpha) * l1
        theta = math.pi * 1.5 - (math.pi - alpha) - beta
        OB = math.sqrt(l1*l1 + l2*l2 - 2*l1*l2*math.cos(beta))
        if OB == 0:
            return h1, 0
        theta_1 = math.acos((l1 ** 2 + OB ** 2 - l2 ** 2) / (2 * l1 * OB))
        theta_2 = theta - theta_1
        h2 = math.cos(theta_2) * OB
        return h1, h2

    def calculate_ground_footing(self):
        joints = self.get_joint_position()
        leg0h1, leg0h2 = self.get_joint_position_wrt_body(joints[1], joints[2])
        leg1h1, leg1h2 = self.get_joint_position_wrt_body(joints[4], joints[5])
        leg2h1, leg2h2 = self.get_joint_position_wrt_body(joints[7], joints[8])
        leg3h1, leg3h2 = self.get_joint_position_wrt_body(joints[10], joints[11])

        return leg0h1, leg0h2, leg1h1, leg1h2, leg2h1, leg2h2, leg3h1, leg3h2
