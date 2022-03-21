import numpy as np
import numpy as np
from inverse_kinematics.kinematic_model import robotKinematics
from inverse_kinematics.gaitPlanner import trotGait

class InverseKinematicsController():
    def __init__(self):
        """
        Initialize an inverse kinematics controller with step size of 0.002
        """
        Xdist = 0.39
        Ydist = 0.28
        height = 0.3

        self.Lrot = 0
        self.angle = 180
        self.L = 1.2
        self.T = 1.0

        self.offset = np.array([0.5, 0.0, 0.0, 0.5])

        self.bodytoFeet0 = np.matrix([[ Xdist/2 , -Ydist/2 , -height],
                         [ Xdist/2 ,  Ydist/2 , -height],
                         [-Xdist/2 , -Ydist/2 , -height],
                         [-Xdist/2 ,  Ydist/2 , -height]])

        self.robotKinematics = robotKinematics()
        self.trot = trotGait()

    def get_action(self, **kwargs):
        """
        Return the next joint positions of the inverse kinematics controller
        """
        bodytoFeet = self.trot.loop(self.L , self.angle , self.Lrot , self.T , self.offset , self.bodytoFeet0)
        FR_angles, FL_angles, BR_angles, BL_angles , _ = self.robotKinematics.solve(np.zeros([3]), np.zeros([3]), bodytoFeet)
        
        return np.array([
            FL_angles[0], FL_angles[1], FL_angles[2] + 3.14,
            BL_angles[0], BL_angles[1], BL_angles[2] + 3.14,
            FR_angles[0], FR_angles[1], FR_angles[2] + 3.14,
            BR_angles[0], BR_angles[1], BR_angles[2] + 3.14
        ])