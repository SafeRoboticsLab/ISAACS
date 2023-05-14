# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Classes for hyper-parameters schedulers.

modified from: https://github.com/SafeRoboticsLab/SimLabReal/blob/main/agent/scheduler.py
"""


class _scheduler(object):

  def __init__(self, last_epoch=-1, verbose=False):
    self.cnt = last_epoch
    self.verbose = verbose
    self.variable = None
    self.step()

  def step(self):
    self.cnt += 1
    value = self.get_value()
    self.variable = value

  def get_value(self):
    raise NotImplementedError

  def get_variable(self):
    return self.variable


class StepLR(_scheduler):

  def __init__(
      self, init_value, period, decay=0.1, end_value=None, last_epoch=-1,
      threshold=0, verbose=False
  ):
    self.init_value = init_value
    self.period = period
    self.decay = decay
    self.end_value = end_value
    self.threshold = threshold
    super(StepLR, self).__init__(last_epoch, verbose)

  def get_value(self):
    cnt = self.cnt - self.threshold
    if cnt < 0:
      return self.init_value

    numDecay = int(cnt / self.period)
    tmpValue = self.init_value * (self.decay**numDecay)
    if self.end_value is not None and tmpValue <= self.end_value:
      return self.end_value
    return tmpValue


class StepLRMargin(_scheduler):

  def __init__(
      self, init_value, period, goal_value, decay=0.1, end_value=None,
      last_epoch=-1, threshold=0, verbose=False
  ):
    self.init_value = init_value
    self.period = period
    self.decay = decay
    self.end_value = end_value
    self.goal_value = goal_value
    self.threshold = threshold
    super(StepLRMargin, self).__init__(last_epoch, verbose)

  def get_value(self):
    cnt = self.cnt - self.threshold
    if cnt < 0:
      return self.init_value

    numDecay = int(cnt / self.period)
    tmpValue = self.goal_value - (self.goal_value
                                  - self.init_value) * (self.decay**numDecay)
    if self.end_value is not None and tmpValue >= self.end_value:
      return self.end_value
    return tmpValue


class StepResetLR(_scheduler):

  def __init__(
      self, init_value, period, resetPeriod, decay=0.1, end_value=None,
      last_epoch=-1, verbose=False
  ):
    self.init_value = init_value
    self.period = period
    self.decay = decay
    self.end_value = end_value
    self.resetPeriod = resetPeriod
    super(StepResetLR, self).__init__(last_epoch, verbose)

  def get_value(self):
    if self.cnt == -1:
      return self.init_value

    numDecay = int(self.cnt / self.period)
    tmpValue = self.init_value * (self.decay**numDecay)
    if self.end_value is not None and tmpValue <= self.end_value:
      return self.end_value
    return tmpValue

  def step(self):
    self.cnt += 1
    value = self.get_value()
    self.variable = value
    if (self.cnt + 1) % self.resetPeriod == 0:
      self.cnt = -1
