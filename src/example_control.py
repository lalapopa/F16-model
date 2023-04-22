import numpy as np
import matplotlib.pyplot as plt

import utils
from model import States, Control, ODE_3DoF

t0 = 0
dt = 0.02
tn = 1
n = int((tn - t0) / dt + 1)

Ox0 = 0
Oy0 = 3000
Vx0 = 250
Vy0 = 0
wz0 = np.radians(0)
theta0 = np.radians(0)
stab0 = np.radians(0)
dstab0 = np.radians(10)
Pa0 = 0
x0 = np.array(
    [
        Ox0,
        Oy0,
        wz0,
        Vx0,
        Vy0,
        theta0,
        stab0,
        dstab0,
        Pa0,
    ]
)

# Control Define
stab_act = np.radians(utils.control.step_function(t0, dt, tn, 0, -10))
throttle_act = np.radians(utils.control.step_function(t0, dt, tn, 0, 0))
u = np.stack((stab_act, throttle_act), axis=-1)

# Calculate all state
x = np.zeros(n, dtype=np.object)
x[0] = x0
for i in range(1, n):
    next_state = dt * ODE_3DoF.solve(x[i - 1], u[i - 1])
    x[i] = x[i - 1] + next_state
    print(x[i])
    print(t0 + dt * i)
print(len(x), len(u))
# print(x[0].get("theta"), x[-1].get("theta"))
# # Plotting
# fig = plt.figure()
#
# plt.subplot(5, 1, 1)
# plt.plot(np.arange(t0, tn + dt, dt), np.degrees([i.stab for i in u]), "-b")
# plt.ylabel("stab_{act}, deg")
#
# plt.subplot(5, 1, 2)
# plt.plot(np.arange(t0, tn + dt, dt), np.degrees([i.throttle for i in u]), "-b")
# plt.ylabel("throttle, deg")
#
# plt.subplot(5, 1, 3)
# plt.plot(np.arange(t0, tn + dt, dt), np.degrees([i.theta for i in x]), "-b")
# plt.ylabel("\theta, deg")
#
# plt.subplot(5, 1, 4)
# plt.plot(np.arange(t0, tn + dt, dt), np.degrees([i.wz for i in x]), "-b")
# plt.ylabel("\omega_{z}, deg/sec")
#
# plt.subplot(5, 1, 5)
# plt.plot(np.arange(t0, tn + dt, dt), [i.Vx for i in x], "-b")
# plt.ylabel("V_{x}, m/s")
# plt.xlabel("t, sec")
# plt.show()
