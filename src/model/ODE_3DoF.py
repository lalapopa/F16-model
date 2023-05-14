from math import cos, sin

import utils.cs_transform
import model
import data
from data.thrust import get_thrust
from model.engine import engine_power_level


def solve(x, u):
    Vx, Vy, _ = utils.cs_transform.wind2body(x.V, x.alpha, 0)

    rho = data.environment.get_density(x.Oy)
    a = data.environment.get_speed_of_sound(x.Oy)
    M = x.V / a
    q = (rho * (x.V**2)) / 2

    Ox_dot = cos(x.theta) * Vx - sin(x.theta) * Vy
    Oy_dot = sin(x.theta) * Vx + cos(x.theta) * Vy
    theta_dot = x.wz

    cx = data.get_Cx(
        x.alpha, 0, x.stab, data.plane.lef, x.wz, x.V, data.plane.bA, data.plane.sb
    )
    cy = data.get_Cy(
        x.alpha, 0, x.stab, data.plane.lef, x.wz, x.V, data.plane.bA, data.plane.sb
    )
    mz = data.get_Mz(
        x.alpha, 0, x.stab, data.plane.lef, x.wz, x.V, data.plane.bA, data.plane.sb
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

    # Limits Control and Throttle value
    control_stab = min(max(u.stab, -data.plane.maxabsstab), data.plane.maxabsstab)
    control_throttle = min(
        max(u.throttle, data.plane.minthrottle), data.plane.maxthrottle
    )

    Vx_dot = x.wz * Vy - data.environment.g * sin(x.theta) + Rx / data.plane.m
    Vy_dot = -x.wz * Vx - data.environment.g * cos(x.theta) + Ry / data.plane.m

    V_dot = (Vx * Vx_dot - Vy * Vy_dot) / x.V
    alpha_dot = (-Vx * Vy_dot + Vy * Vx_dot) / (Vx**2 + Vy**2)
    wz_dot = MRz / data.plane.Jz

    Dx = model.States(
        Ox=float(Ox_dot),
        Oy=float(Oy_dot),
        wz=float(wz_dot),
        theta=float(theta_dot),
        V=float(V_dot),
        alpha=float(alpha_dot),
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
