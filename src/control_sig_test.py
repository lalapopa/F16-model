import numpy as np
import matplotlib.pyplot as plt
import utils


t0 = 0
dt = 0.02
tn = 20
time = np.arange(t0, tn + dt, dt)


plt.plot(
    time, utils.control.make_step_series(t0, dt, tn, 1.2, step_time=1, hold_time=4)
)
plt.show()
