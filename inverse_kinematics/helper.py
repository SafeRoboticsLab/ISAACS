import matplotlib.pyplot as plt
from IPython import display

plt.ion()
plt.show()

def plot(roll, pitch, yaw):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Joint Movement')
    plt.xlabel('Number of steps')
    plt.ylabel('Angle (Rad)')
    plt.plot(roll, label = "Body Roll")
    plt.plot(pitch, label = "Body Pitch")
    plt.plot(yaw, label = "Body Yaw")
    plt.legend()
    plt.pause(0.001)