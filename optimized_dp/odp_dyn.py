from abc import ABC, abstractmethod
from typing import Optional, Union, List
import numpy as np
import heterocl as hcl


class HCL_dyn(ABC):

  def __init__(
      self,
      x: Union[float, List, np.ndarray],
      uMax: Union[float, List, np.ndarray],
      dMax: Union[float, List, np.ndarray],
      uMode: str = "max",
      dMode: str = "min",
      uMin: Optional[Union[float, List, np.ndarray]] = None,
      dMin: Optional[Union[float, List, np.ndarray]] = None,
  ):
    self.x = x
    self.uMax = uMax
    self.dMax = dMax
    print("uMax", self.uMax)
    print("dMax", self.dMax)
    if uMin is None:
      self.uMin = -self.uMax.copy()
    if dMin is None:
      self.dMin = -self.dMax.copy()

    assert ((uMode == 'max' and dMode == 'min')
            or (uMode == 'min' and dMode
                == 'max')), "Dstb mode should be the oopsite of Ctrl mode."
    self.uMode = uMode
    self.dMode = dMode

  @abstractmethod
  def opt_ctrl(
      self, t: float, state: np.ndarray, spat_deriv: np.ndarray
  ) -> List:
    raise NotImplementedError

  @abstractmethod
  def opt_dstb(
      self, t: float, state: np.ndarray, spat_deriv: np.ndarray
  ) -> List:
    raise NotImplementedError

  def opt_dstb_np(
      self, t: float, state: np.ndarray, spat_deriv: np.ndarray
  ) -> np.ndarray:
    d = self.dMax.copy()
    b_term = spat_deriv

    for idx in range(len(d)):
      if b_term[idx] >= 0:
        if self.dMode == "min":
          d[idx] = self.dMin[idx]
      elif b_term[idx] < 0:
        if self.dMode == "max":
          d[idx] = self.dMin[idx]

    return d

  @abstractmethod
  def opt_ctrl_np(
      self, t: float, state: np.ndarray, spat_deriv: np.ndarray
  ) -> np.ndarray:
    raise NotImplementedError

  @abstractmethod
  def dynamics_np(
      self, t, state: np.ndarray, uOpt: np.ndarray, dOpt: np.ndarray
  ) -> np.ndarray:
    raise NotImplementedError


class DubinsCarDstb3D(HCL_dyn):
  """
  Dynamics of the Dubins Car
    x1_dot = v * cos(x3) + d1
    x2_dot = v * sin(x3) + d2
    x3_dot = w
    Control: u = w;
  """

  def __init__(
      self, x=[0, 0, 0], uMax=1.0, speed=1.0, dMax=np.array([0.25, 0.25, 0.]),
      dMin: Optional[np.ndarray] = None, uMode="max", dMode="min"
  ):
    super().__init__(
        x=x, uMax=uMax, uMin=None, dMax=dMax, dMin=dMin, uMode=uMode,
        dMode=dMode
    )
    self.speed = speed

  def opt_ctrl(self, t, state, spat_deriv):

    u0 = hcl.scalar(self.uMax, "u0")

    # Just create and pass back, even though they're not used
    in1 = hcl.scalar(0, "in1")
    in2 = hcl.scalar(0, "in2")

    a_term = hcl.scalar(spat_deriv[2], "a_term")

    # use the scalar by indexing 0 everytime
    with hcl.if_(a_term >= 0):
      with hcl.if_(self.uMode == "min"):
        u0[0] = self.uMin
    with hcl.elif_(a_term < 0):
      with hcl.if_(self.uMode == "max"):
        u0[0] = self.uMin
    return (u0[0], in1[0], in2[0])

  def opt_dstb(self, t, state, spat_deriv):
    d0 = hcl.scalar(self.dMax[0], "d0")
    d1 = hcl.scalar(self.dMax[1], "d1")
    d2 = hcl.scalar(self.dMax[2], "d2")

    b_term0 = hcl.scalar(spat_deriv[0], "b_term0")
    b_term1 = hcl.scalar(spat_deriv[1], "b_term1")
    b_term2 = hcl.scalar(spat_deriv[2], "b_term2")

    with hcl.if_(b_term0[0] >= 0):
      with hcl.if_(self.dMode == "min"):
        d0[0] = self.dMin[0]
    with hcl.elif_(b_term0[0] < 0):
      with hcl.if_(self.dMode == "max"):
        d0[0] = self.dMin[0]

    with hcl.if_(b_term1[0] >= 0):
      with hcl.if_(self.dMode == "min"):
        d1[0] = self.dMin[1]
    with hcl.elif_(b_term1[0] < 0):
      with hcl.if_(self.dMode == "max"):
        d1[0] = self.dMin[1]

    with hcl.if_(b_term2[0] >= 0):
      with hcl.if_(self.dMode == "min"):
        d2[0] = self.dMin[2]
    with hcl.elif_(b_term2[0] < 0):
      with hcl.if_(self.dMode == "max"):
        d2[0] = self.dMin[2]
    return (d0[0], d1[0], d2[0])

  def dynamics(self, t, state, uOpt, dOpt):
    x_dot = hcl.scalar(0, "x_dot")
    y_dot = hcl.scalar(0, "y_dot")
    theta_dot = hcl.scalar(0, "theta_dot")

    x_dot[0] = self.speed * hcl.cos(state[2]) + dOpt[0]
    y_dot[0] = self.speed * hcl.sin(state[2]) + dOpt[1]
    theta_dot[0] = uOpt[0] + dOpt[2]

    return (x_dot[0], y_dot[0], theta_dot[0])

  # The below function can have whatever form or parameters users want
  # These functions are not used in HeteroCL program, hence is pure Python code
  # and can be used after the value function has been obtained.
  def opt_ctrl_np(
      self, t: float, state: np.ndarray, spat_deriv: np.ndarray
  ) -> np.ndarray:
    a_term = spat_deriv[2]

    opt_w = self.uMax
    if a_term >= 0:
      if self.uMode == "min":
        opt_w = self.uMin
    else:
      if self.uMode == "max":
        opt_w = self.uMin
    return opt_w


class DubinsCarDstb4D(HCL_dyn):

  def __init__(
      self, x: np.ndarray = np.zeros(4), uMax: np.ndarray = np.array([1., 1.]),
      uMin: Optional[np.ndarray] = None,
      dMax: np.ndarray = np.array([0.2, 0.2, 0.2, 0.2]),
      dMin: Optional[np.ndarray] = None, uMode: str = "max", dMode: str = "min"
  ):

    super().__init__(
        x=x, uMax=uMax, uMin=uMin, dMax=dMax, dMin=dMin, uMode=uMode,
        dMode=dMode
    )

  def opt_ctrl(self, t, state, spat_deriv):

    u0 = hcl.scalar(self.uMax[0], "u0")
    u1 = hcl.scalar(self.uMax[1], "u1")

    # Just create and pass back, even though they're not used
    in2 = hcl.scalar(0, "in2")
    in3 = hcl.scalar(0, "in3")

    a_term0 = hcl.scalar(spat_deriv[2], "a_term0")
    a_term1 = hcl.scalar(spat_deriv[3], "a_term1")

    with hcl.if_(a_term0 >= 0):
      with hcl.if_(self.uMode == "min"):
        u0[0] = self.uMin[0]
    with hcl.elif_(a_term0 < 0):
      with hcl.if_(self.uMode == "max"):
        u0[0] = self.uMin[0]

    with hcl.if_(a_term1 >= 0):
      with hcl.if_(self.uMode == "min"):
        u1[0] = self.uMin[1]
    with hcl.elif_(a_term1 < 0):
      with hcl.if_(self.uMode == "max"):
        u1[0] = self.uMin[1]

    return (u0[0], u1[0], in2[0], in3[0])

  def opt_dstb(self, t, state, spat_deriv):
    d0 = hcl.scalar(self.dMax[0], "d0")
    d1 = hcl.scalar(self.dMax[1], "d1")
    d2 = hcl.scalar(self.dMax[2], "d2")
    d3 = hcl.scalar(self.dMax[3], "d3")

    b_term0 = hcl.scalar(spat_deriv[0], "b_term0")
    b_term1 = hcl.scalar(spat_deriv[1], "b_term1")
    b_term2 = hcl.scalar(spat_deriv[2], "b_term2")
    b_term3 = hcl.scalar(spat_deriv[3], "b_term3")

    with hcl.if_(b_term0[0] >= 0):
      with hcl.if_(self.dMode == "min"):
        d0[0] = self.dMin[0]
    with hcl.elif_(b_term0[0] < 0):
      with hcl.if_(self.dMode == "max"):
        d0[0] = self.dMin[0]

    with hcl.if_(b_term1[0] >= 0):
      with hcl.if_(self.dMode == "min"):
        d1[0] = self.dMin[1]
    with hcl.elif_(b_term1[0] < 0):
      with hcl.if_(self.dMode == "max"):
        d1[0] = self.dMin[1]

    with hcl.if_(b_term2[0] >= 0):
      with hcl.if_(self.dMode == "min"):
        d2[0] = self.dMin[2]
    with hcl.elif_(b_term2[0] < 0):
      with hcl.if_(self.dMode == "max"):
        d2[0] = self.dMin[2]

    with hcl.if_(b_term3[0] >= 0):
      with hcl.if_(self.dMode == "min"):
        d3[0] = self.dMin[3]
    with hcl.elif_(b_term3[0] < 0):
      with hcl.if_(self.dMode == "max"):
        d3[0] = self.dMin[3]

    return (d0[0], d1[0], d2[0], d3[0])

  def dynamics(self, t, state, uOpt, dOpt):
    x_dot = hcl.scalar(0, "x_dot")
    y_dot = hcl.scalar(0, "y_dot")
    v_dot = hcl.scalar(0, "v_dot")
    yaw_dot = hcl.scalar(0, "yaw_dot")

    x_dot[0] = state[2] * hcl.cos(state[3]) + dOpt[0]
    y_dot[0] = state[2] * hcl.sin(state[3]) + dOpt[1]
    v_dot[0] = uOpt[0] + dOpt[2]
    yaw_dot[0] = uOpt[1] + dOpt[3]

    return (x_dot[0], y_dot[0], v_dot[0], yaw_dot[0])

  def opt_ctrl_np(
      self, t: float, state: np.ndarray, spat_deriv: np.ndarray
  ) -> np.ndarray:
    u = self.uMax.copy()
    a_term0 = spat_deriv[2]
    a_term1 = spat_deriv[3]

    if a_term0 >= 0:
      if self.uMode == "min":
        u[0] = self.uMin[0]
    if a_term0 < 0:
      if self.uMode == "max":
        u[0] = self.uMin[0]

    if a_term1 >= 0:
      if self.uMode == "min":
        u[1] = self.uMin[1]
    if a_term1 < 0:
      if self.uMode == "max":
        u[1] = self.uMin[1]

    return u


class BicycleDstb5D(HCL_dyn):

  def __init__(
      self, L: float, x: np.ndarray = np.zeros(5),
      uMax: np.ndarray = np.array([3.5, 5.]),
      dMax: np.ndarray = np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
      uMin: Optional[np.ndarray] = None, dMin: Optional[np.ndarray] = None,
      uMode: str = "max", dMode: str = "min", v_max: float = 2.,
      v_min: float = 0.4, delta_max: float = 0.35, delta_min: float = -0.35
  ):
    self.L = L
    super().__init__(
        x=x, uMax=uMax, uMin=uMin, dMax=dMax, dMin=dMin, uMode=uMode,
        dMode=dMode
    )
    #* Clips linear accel and steering angular velocity.
    self.v_max = v_max
    self.v_min = v_min
    self.delta_max = delta_max
    self.delta_min = delta_min

  def opt_ctrl(self, t, state, spat_deriv):

    u0 = hcl.scalar(self.uMax[0], "u0")
    u1 = hcl.scalar(self.uMax[1], "u1")

    # Just create and pass back, even though they're not used
    in2 = hcl.scalar(0, "in2")
    in3 = hcl.scalar(0, "in3")
    in4 = hcl.scalar(0, "in4")

    a_term0 = hcl.scalar(spat_deriv[2], "a_term0")
    a_term1 = hcl.scalar(spat_deriv[4], "a_term1")

    with hcl.if_(a_term0 >= 0):
      with hcl.if_(self.uMode == "min"):
        u0[0] = self.uMin[0]
    with hcl.elif_(a_term0 < 0):
      with hcl.if_(self.uMode == "max"):
        u0[0] = self.uMin[0]

    with hcl.if_(a_term1 >= 0):
      with hcl.if_(self.uMode == "min"):
        u1[0] = self.uMin[1]
    with hcl.elif_(a_term1 < 0):
      with hcl.if_(self.uMode == "max"):
        u1[0] = self.uMin[1]

    return (u0[0], u1[0], in2[0], in3[0], in4[0])

  def opt_dstb(self, t, state, spat_deriv):
    d0 = hcl.scalar(self.dMax[0], "d0")
    d1 = hcl.scalar(self.dMax[1], "d1")
    d2 = hcl.scalar(self.dMax[2], "d2")
    d3 = hcl.scalar(self.dMax[3], "d3")
    d4 = hcl.scalar(self.dMax[4], "d4")

    b_term0 = hcl.scalar(spat_deriv[0], "b_term0")
    b_term1 = hcl.scalar(spat_deriv[1], "b_term1")
    b_term2 = hcl.scalar(spat_deriv[2], "b_term2")
    b_term3 = hcl.scalar(spat_deriv[3], "b_term3")
    b_term4 = hcl.scalar(spat_deriv[4], "b_term4")

    with hcl.if_(b_term0[0] >= 0):
      with hcl.if_(self.dMode == "min"):
        d0[0] = self.dMin[0]
    with hcl.elif_(b_term0[0] < 0):
      with hcl.if_(self.dMode == "max"):
        d0[0] = self.dMin[0]

    with hcl.if_(b_term1[0] >= 0):
      with hcl.if_(self.dMode == "min"):
        d1[0] = self.dMin[1]
    with hcl.elif_(b_term1[0] < 0):
      with hcl.if_(self.dMode == "max"):
        d1[0] = self.dMin[1]

    with hcl.if_(b_term2[0] >= 0):
      with hcl.if_(self.dMode == "min"):
        d2[0] = self.dMin[2]
    with hcl.elif_(b_term2[0] < 0):
      with hcl.if_(self.dMode == "max"):
        d2[0] = self.dMin[2]

    with hcl.if_(b_term3[0] >= 0):
      with hcl.if_(self.dMode == "min"):
        d3[0] = self.dMin[3]
    with hcl.elif_(b_term3[0] < 0):
      with hcl.if_(self.dMode == "max"):
        d3[0] = self.dMin[3]

    with hcl.if_(b_term4[0] >= 0):
      with hcl.if_(self.dMode == "min"):
        d4[0] = self.dMin[4]
    with hcl.elif_(b_term4[0] < 0):
      with hcl.if_(self.dMode == "max"):
        d4[0] = self.dMin[4]

    return (d0[0], d1[0], d2[0], d3[0], d4[0])

  def dynamics(self, t, state, uOpt, dOpt):
    x_dot = hcl.scalar(0, "x_dot")
    y_dot = hcl.scalar(0, "y_dot")
    v_dot = hcl.scalar(0, "v_dot")
    yaw_dot = hcl.scalar(0, "yaw_dot")
    delta_dot = hcl.scalar(0, "delta_dot")

    x_dot[0] = state[2] * hcl.cos(state[3]) + dOpt[0]
    y_dot[0] = state[2] * hcl.sin(state[3]) + dOpt[1]

    cond = hcl.or_(
        hcl.and_(state[2] >= self.v_max, uOpt[0] > 0.),
        hcl.and_(state[2] <= self.v_min, uOpt[0] < 0.)
    )
    with hcl.if_(cond):
      v_dot[0] = dOpt[2]
    with hcl.else_():
      v_dot[0] = uOpt[0] + dOpt[2]

    yaw_dot[0] = (
        state[2] * hcl.sin(state[4]) / hcl.cos(state[4]) / self.L + dOpt[3]
    )

    cond = hcl.or_(
        hcl.and_(state[4] >= self.delta_max, uOpt[1] > 0.),
        hcl.and_(state[4] <= self.delta_min, uOpt[1] < 0.)
    )
    with hcl.if_(cond):
      delta_dot[0] = dOpt[4]
    with hcl.else_():
      delta_dot[0] = uOpt[1] + dOpt[4]

    return (x_dot[0], y_dot[0], v_dot[0], yaw_dot[0], delta_dot[0])

  def opt_ctrl_np(
      self, t: float, state: np.ndarray, spat_deriv: np.ndarray
  ) -> np.ndarray:
    u = self.uMax.copy()
    a_term0 = spat_deriv[2]
    a_term1 = spat_deriv[4]

    if a_term0 >= 0:
      if self.uMode == "min":
        u[0] = self.uMin[0]
    elif a_term0 < 0:
      if self.uMode == "max":
        u[0] = self.uMin[0]

    if a_term1 >= 0:
      if self.uMode == "min":
        u[1] = self.uMin[1]
    elif a_term1 < 0:
      if self.uMode == "max":
        u[1] = self.uMin[1]

    return u

  def dynamics_np(
      self, t, state: np.ndarray, uOpt: np.ndarray, dOpt: np.ndarray
  ) -> np.ndarray:
    state_dot = np.zeros_like(self.x)

    state_dot[0] = state[2] * np.cos(state[3]) + dOpt[0]
    state_dot[1] = state[2] * np.sin(state[3]) + dOpt[1]
    state_dot[2] = uOpt[0] + dOpt[2]
    state_dot[3] = (state[2] * np.tan(state[4]) / self.L + dOpt[3])
    state_dot[4] = uOpt[1] + dOpt[4]

    return state_dot
