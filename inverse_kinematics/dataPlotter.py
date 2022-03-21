import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("data.csv")
data = data.drop([data.columns[0]], axis = 1)
data.plot()
plt.xlabel("Steps")
plt.ylabel("Velocity")
plt.title("Velocity of robot")
plt.legend()
plt.show()