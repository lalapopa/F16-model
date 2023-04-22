from math import cos, sin
import numpy as np

import data
import utils
import model
from data.thrust import get_thrust
from model.engine import engine_power_level


def solve(x, u):
    V, alpha, _ = utils.cs_transform.body2wind(x[3], x[4], 0)

    rho = data.environment.get_density(x[1])
    a = data.environment.get_speed_of_sound(x[1])
    M = V / a
    q = (rho * (V**2)) / 2

    cx = data.get_Cx(
        alpha,
        0,
        x[6],
        data.plane.lef,
        x[2],
        V,
        data.plane.bA,
        data.plane.sb,
    )
    cy = data.get_Cy(
        alpha,
        0,
        x[6],
        data.plane.lef,
        x[2],
        V,
        data.plane.bA,
        data.plane.sb,
    )
    mz = data.get_Mz(
        alpha,
        0,
        x[6],
        data.plane.lef,
        x[2],
        V,
        data.plane.bA,
        data.plane.sb,
    )

    X = -q * data.plane.S * cx
    Y = q * data.plane.S * cy

    Px = get_thrust(x[1], M, x[8])
    Py = 0

    Mz = q * data.plane.S * data.plane.bA * mz
    MPz = 0

    Rx = X + Px
    Ry = Y + Py
    MRz = Mz + MPz + data.plane.rcgx * Ry

    control_stab = np.minimum(
        np.maximum(u[0], -data.plane.maxabsstab), data.plane.maxabsstab
    )
    control_throttle = np.minimum(
        np.maximum(u[1], data.plane.minthrottle), data.plane.maxthrottle
    )
    # "Ox", "Oy", "wz", "Vx", "Vy", "theta", "stab", "dstab", "Pa"
    Dx = np.array(
        [
            float(cos(x[5]) * x[3] - sin(x[5]) * x[4]),
            float(sin(x[5]) * x[3] + cos(x[5]) * x[4]),
            float(MRz / data.plane.Jz),
            float(x[2] * x[4] - data.environment.g * sin(x[5]) + Rx / data.plane.m),
            float(-x[2] * x[3] - data.environment.g * cos(x[5]) + Ry / data.plane.m),
            float(x[2]),
            float(
                np.minimum(
                    np.maximum(x[7], -data.plane.maxabsdstab), data.plane.maxabsdstab
                )
            ),
            float(
                (-2 * data.plane.Tstab * data.plane.Xistab * x[7] - x[6] + control_stab)
                / (data.plane.Tstab**2)
            ),
            float(engine_power_level(x[8], control_throttle)),
        ]
    )
    return Dx
