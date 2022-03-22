# Note on the URDF of Spirit robot
The following information will be recorded:

For each joint:
* Max force (torque)
* Max velocity (angular)

For each link:
* Size
* Inertia (mass)

* Body (link)
  * Size: 0.335 0.24 0.104
  * Mass: 5.75
* Hip (link)
  * Size: cylinder length=0.08 radius=0.055
  * Mass: 0.575
* Body - hip joint (abduction joint)
  * torque: 40 (Nm)
  * velocity: 30 (rad/s)
  * lim: [-0.707, 0.707]
* Hip - Upper joint (Hip joint)
  * torque: 40 (Nm)
  * velocity: 30 (rad/s)
  * lim: [-6.28, 6.28]
* Upper link
  * Size: 0.206 0.022 0.055
  * Mass: 0.775
* Upper - Lower joint (Knee joint)
  * torque: 40 (Nm)
  * velocity: 30 (rad/s)
  * lim: [0, 3.14]
* Lower link
  * Size: cylinder length=0.206 radius=0.013
  * Mass: 0.075
* Toe link
  * Size: sphere radius 0.02
  * mass: 0.015

Mass of robot: body + 4 x (hip link + upper link + lower link + toe) = 11.51 (kg)

# From datasheet
![Datasheet](https://uspto.report/ts/cd/pdfs?f=/SOU/2020/06/23/20200623125618914893-88276426-004_003/SPN1-3898152106-20200623124743387022_._GR_Spirit_40-P_Quad_UGV-_Full_Spec_rev1.0.pdf)

* Maximum output torque: hip and abduction motors at 21Nm, and knee motor at 32Nm
* Maximum speed: hip and abduction motors at 360 rpm, and knee motor at 245 rpm

360 rpm ~ 37.699112 rad/s
245 rpm ~ 25.65634 rad/s

# Proposed changes
Hip and abduction joint:
* torque: 40 --> 21 Nm
* velocity: 30 --> 37.7 rad/s

Knee joint
* torque: 40 --> 32 Nm
* velocity: 30 --> 25.67 rad/s

Robot's mass can be left unchanged.

# Reference
![ROS URDF safety limit for joint explanation](http://wiki.ros.org/pr2_controller_manager/safety_limits)