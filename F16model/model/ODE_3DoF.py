from math import cos, sin

from F16model.utils.cs_transform import wind2body
import F16model.model as model
from F16model.data import thrust, plane, atmosphere, coeff


def solve(x, u):
    Vx, Vy, _ = wind2body(x.V, x.alpha, 0)

    rho = atmosphere.get_density(x.Oy)
    a = atmosphere.get_speed_of_sound(x.Oy)
    M = x.V / a
    q = (rho * (x.V**2)) / 2

    Ox_dot = cos(x.theta) * Vx - sin(x.theta) * Vy
    Oy_dot = sin(x.theta) * Vx + cos(x.theta) * Vy
    theta_dot = x.wz

    cx = coeff.get_Cx(x.alpha, 0, x.stab, plane.lef, x.wz, x.V, plane.b_a, plane.sb)
    cy = coeff.get_Cy(x.alpha, 0, x.stab, plane.lef, x.wz, x.V, plane.b_a, plane.sb)
    mz = coeff.get_Mz(x.alpha, 0, x.stab, plane.lef, x.wz, x.V, plane.b_a, plane.sb)

    X = -q * plane.S * cx
    Y = q * plane.S * cy

    Px = thrust.get_thrust(x.Oy, M, x.Pa)
    Py = 0
    Mz = q * plane.S * plane.b_a * mz
    MPz = 0

    Rx = X + Px
    Ry = Y + Py
    MRz = Mz + MPz + plane.rcgx * Ry

    # Limits Control and Throttle value
    control_stab = min(max(u.stab, -plane.maxabsstab), plane.maxabsstab)
    control_throttle = min(max(u.throttle, plane.minthrottle), plane.maxthrottle)

    Vx_dot = x.wz * Vy - atmosphere.g * sin(x.theta) + Rx / plane.m
    Vy_dot = -x.wz * Vx - atmosphere.g * cos(x.theta) + Ry / plane.m

    V_dot = (Vx * Vx_dot - Vy * Vy_dot) / x.V
    alpha_dot = (-Vx * Vy_dot + Vy * Vx_dot) / (Vx**2 + Vy**2)
    wz_dot = MRz / plane.Jz

    Dx = model.States(
        Ox=float(Ox_dot),
        Oy=float(Oy_dot),
        wz=float(wz_dot),
        theta=float(theta_dot),
        V=float(V_dot),
        alpha=float(alpha_dot),
        stab=float(min(max(x.dstab, -plane.maxabsdstab), plane.maxabsdstab)),
        dstab=float(
            (-2 * plane.Tstab * plane.Xistab * x.dstab - x.stab + control_stab)
            / (plane.Tstab**2)
        ),
        Pa=float(model.engine.engine_power_level(x.Pa, control_throttle)),
    )
    return Dx
