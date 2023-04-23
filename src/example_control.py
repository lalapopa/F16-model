import numpy as np
import matplotlib.pyplot as plt
import time

import utils
from model import States, Control, ODE_3DoF

t0 = 0
dt = 0.05
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
x0 = States(Ox0, Oy0, Vx0, Vy0, wz0, theta0, stab0, dstab0, Pa0)

# Control Define
stab_act = np.radians(utils.control.step_function(t0, dt, tn, 0, -10))
throttle_act = np.radians(utils.control.step_function(t0, dt, tn, 0, 0))
u = np.zeros(n, dtype=object)

u = [Control(stab_act[i], throttle_act[i]) for i, _ in enumerate(u)]


# Calculate all state
start = time.time()
x = []
x = np.zeros(n, dtype=object)
x[0] = x0
for i in range(1, n):
    x[i] = x[i - 1] + dt * ODE_3DoF.solve(x[i - 1], u[i - 1])
    print(t0 + dt * i)
print(f"FNISED IN {time.time() - start }s")

# Plotting
fig = plt.figure()

plt.subplot(5, 1, 1)
plt.plot(np.arange(t0, tn + dt, dt), np.degrees([i.stab for i in u]), "-b")
plt.grid()
plt.xlim(t0, tn)
plt.ylabel(r"$stab_{act}$, deg")

plt.subplot(5, 1, 2)
plt.plot(np.arange(t0, tn + dt, dt), np.degrees([i.throttle for i in u]), "-b")
plt.xlim(t0, tn)
plt.grid()
plt.ylabel("throttle, deg")

plt.subplot(5, 1, 3)
plt.plot(np.arange(t0, tn + dt, dt), np.degrees([i.theta for i in x]), "-b")
plt.grid()
plt.xlim(t0, tn)
plt.ylabel(r"$\theta$, deg")

plt.subplot(5, 1, 4)
plt.plot(np.arange(t0, tn + dt, dt), np.degrees([i.wz for i in x]), "-b")
plt.grid()
plt.xlim(t0, tn)
plt.ylabel("$\omega_{z}$, deg/sec")

plt.subplot(5, 1, 5)
plt.plot(np.arange(t0, tn + dt, dt), [i.Vx for i in x], "-b")
plt.ylabel("$V_{x}$, m/s")
plt.grid()
plt.xlim(t0, tn)
plt.xlabel("t, sec")
plt.show()
