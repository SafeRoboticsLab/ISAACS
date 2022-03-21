import matplotlib.pyplot as plt
from scipy.stats import truncnorm

clip_a = 1.5
clip_b = 2.3
mean = 1.9
sd = 0.1

a = (clip_a - mean)/sd
b = (clip_b - mean)/sd

print("a: {}, b: {}".format(a, b))

r1 = truncnorm.rvs(a, b, loc = mean, scale = sd, size = 10000)
r2 = truncnorm.rvs(a = -14.5, b = 16.9, loc = 1.45, scale = 0.3, size = 10000)
plt.hist(r2)
plt.show()