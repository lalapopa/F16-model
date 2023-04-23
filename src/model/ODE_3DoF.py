from math import cos, sin 

import data
import utils
import model
from data.thrust import get_thrust
from model.engine import engine_power_level


def solve(x, u):
    V, alpha, _ = utils.cs_transform.body2wind(x.Vx, x.Vy, 0)

    rho = data.environment.get_density(x.Oy)
    a = data.environment.get_speed_of_sound(x.Oy)
    M = V / a
    q = (rho * (V**2)) / 2

    cx = data.get_Cx(
        alpha, 0, x.stab, data.plane.lef, x.wz, V, data.plane.bA, data.plane.sb
    )
    cy = data.get_Cy(
        alpha, 0, x.stab, data.plane.lef, x.wz, V, data.plane.bA, data.plane.sb
    )
    mz = data.get_Mz(
        alpha, 0, x.stab, data.plane.lef, x.wz, V, data.plane.bA, data.plane.sb
    )

    X = -q * data.plane.S * cx
    Y = q * data.plane.S * cy

    Px = get_thrust(x.Oy, M, x.Pa)
    Py = 0

    Mz = q * data.plane.S * data.plane.bA * mz
    MPz = 0

    Rx = X + Px
    Ry = Y + Py
    MRz = Mz + MPz + data.plane.rcgx * Ry

    Dx = x
    control_stab = min(max(u.stab, -data.plane.maxabsstab), data.plane.maxabsstab)
    control_throttle = min(
        max(u.throttle, data.plane.minthrottle), data.plane.maxthrottle
    )
    Dx = model.States(
        Ox=float(cos(x.theta) * x.Vx - sin(x.theta) * x.Vy),
        Oy=float(sin(x.theta) * x.Vx + cos(x.theta) * x.Vy),
        wz=float(MRz / data.plane.Jz),
        Vx=float(
            x.wz * x.Vy - data.environment.g * sin(x.theta) + Rx / data.plane.m
        ),
        Vy=float(
            -x.wz * x.Vx - data.environment.g * cos(x.theta) + Ry / data.plane.m
        ),
        theta=float(x.wz),
        stab=float(min(max(x.dstab, -data.plane.maxabsdstab), data.plane.maxabsdstab)),
        dstab=float(
            (
                -2 * data.plane.Tstab * data.plane.Xistab * x.dstab
                - x.stab
                + control_stab
            )
            / (data.plane.Tstab**2)
        ),
        Pa=float(engine_power_level(x.Pa, control_throttle)),
    )
    return Dx
