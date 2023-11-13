import numpy as np

m = 9295.44  # kg
l = 9.144  # m
S = 27.87  # m^2
b_a = 3.45  # m
Jz = 75673.6  # m/sec^2
rcgx = -0.05 * b_a  # m

Tstab = 0.03
Xistab = 0.707
maxabsstab = np.radians(25)
maxabsdstab = np.radians(60)
minthrottle = 0
maxthrottle = 1

lef = 0
sb = 0

state_restrictions = {
    "Oy": [0, 15000],
    "wz": [np.radians(-60), np.radians(60)],
    "theta": [np.radians(-60), np.radians(60)],
    "V": [0, 600],
    "alpha": [np.radians(-90), np.radians(90)],
}
