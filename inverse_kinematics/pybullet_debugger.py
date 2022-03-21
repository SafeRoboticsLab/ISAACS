#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 22:15:21 2020

@author: linux-asd
"""
import pybullet as p
import time
import numpy as np
import sys

class pybulletDebug:
    def __init__(self):
        #Camera paramers to be able to yaw pitch and zoom the camera (Focus remains on the robot) 
        self.cyaw = 23
        self.cpitch = -24
        self.cdist = 1.5
        time.sleep(0.5)

        self.xId = p.addUserDebugParameter("x" , -0.10 , 0.10 , 0.)
        self.yId = p.addUserDebugParameter("y" , -0.10 , 0.10 , 0.)
        self.zId = p.addUserDebugParameter("z" , -0.10 , 0.10 , 0.)
        self.rollId = p.addUserDebugParameter("roll" , -np.pi/4 , np.pi/4 , 0.)
        self.pitchId = p.addUserDebugParameter("pitch" , -np.pi/4 , np.pi/4 , 0.)
        self.yawId = p.addUserDebugParameter("yaw" , -np.pi/4 , np.pi/4 , 0.)
        self.LId = p.addUserDebugParameter("L" , -1, 1, 0)
        self.LrotId = p.addUserDebugParameter("Lrot" , -1.5 , 1.5 , 0.)
        self.angleId = p.addUserDebugParameter("angleWalk" , -180. , 180. , 0.)
        self.periodId = p.addUserDebugParameter("stepPeriod" , 0.1 , 3. , 1.0)

    def cam_and_robotstates(self, boxId):
        cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
        p.resetDebugVisualizerCamera( cameraDistance=self.cdist, cameraYaw=self.cyaw, cameraPitch=self.cpitch, cameraTargetPosition=cubePos)
        # print("cdist: {}, cyaw: {}, cpitch: {}".format(self.cdist, self.cyaw, self.cpitch))
        vel = p.getBaseVelocity(boxId)[0][:]
        keys = p.getKeyboardEvents()
        #Keys to change camera
        if keys.get(100):  #D
            self.cyaw+=2
        if keys.get(97):   #A
            self.cyaw-=2
        if keys.get(99):   #C
            self.cpitch+=2
        if keys.get(102):  #F
            self.cpitch-=2
        if keys.get(122):  #Z
            self.cdist+=.1
        if keys.get(120):  #X
            self.cdist-=.1
        if keys.get(27):  #ESC
            p.disconnect()
            sys.exit()
            
        #read position from debug
        pos = np.array([p.readUserDebugParameter(self.xId),p.readUserDebugParameter(self.yId), p.readUserDebugParameter(self.zId)])
        orn = np.array([p.readUserDebugParameter(self.rollId),p.readUserDebugParameter(self.pitchId), p.readUserDebugParameter(self.yawId)])
        L = p.readUserDebugParameter(self.LId)
        Lrot = p.readUserDebugParameter(self.LrotId)
        angle = p.readUserDebugParameter(self.angleId)
        T = p.readUserDebugParameter(self.periodId)
        
        return pos , orn , L , angle , Lrot , T