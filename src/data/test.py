import numpy as np
import matplotlib.pyplot as plt
from csaps import csaps


np.random.seed(1234)
theta = np.linspace(0, 2 * np.pi, 10)
x = np.cos(theta) + np.random.randn(10) * 0.1
y = np.sin(theta) + np.random.randn(10) * 0.1
data = [x, y]
theta_i = np.linspace(0, 2 * np.pi, 35)
data_i = csaps(theta, data, theta_i, smooth=0.95)
xi = data_i[0, :]
yi = data_i[1, :]

# plt.plot(x, y, ":o", xi, yi, "-")
result = " ".join(str(item) for item in data)

print(xi[10])
