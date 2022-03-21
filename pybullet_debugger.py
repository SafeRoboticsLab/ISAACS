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
    #Camera paramers to be able to yaw pitch and zoom the camera (Focus remains
    # on the robot)
    self.cyaw = 45
    self.cpitch = -20
    self.cdist = 2
    time.sleep(0.5)

  def cam_and_robotstates(self, boxId):
    robotPos, robotOrn = p.getBasePositionAndOrientation(boxId)
    p.resetDebugVisualizerCamera(
        cameraDistance=self.cdist, cameraYaw=self.cyaw,
        cameraPitch=self.cpitch, cameraTargetPosition=robotPos
    )
    keys = p.getKeyboardEvents()
    #Keys to change camera
    if keys.get(100):  #D
      self.cyaw += .5
    if keys.get(97):  #A
      self.cyaw -= .5
    if keys.get(99):  #C
      self.cpitch += .5
    if keys.get(102):  #F
      self.cpitch -= .5
    if keys.get(122):  #Z
      self.cdist += .02
    if keys.get(120):  #X
      self.cdist -= .02
    if keys.get(27):  #ESC
      p.disconnect()
      sys.exit()
