import numpy as np

import model.ODE_3DoF


class States:
    def __init__(self, Ox, Oy, Vx, Vy, wz, theta, stab, dstab, Pa):
        self.Ox = Ox
        self.Oy = Oy
        self.Vx = Vx
        self.Vy = Vy
        self.wz = wz
        self.theta = theta
        self.stab = stab
        self.dstab = dstab
        self.Pa = Pa

    def __add__(self, other):
        if isinstance(other, States):
            return States(
                other.Ox + self.Ox,
                other.Oy + self.Oy,
                other.Vx + self.Vx,
                other.Vy + self.Vy,
                other.wz + self.wz,
                other.theta + self.theta,
                other.stab + self.stab,
                other.dstab + self.dstab,
                other.Pa + self.Pa,
            )
        else:
            return NotImplemented

    def __rmul__(self, other):
        if np.isscalar(other):
            return States(
                self.Ox * other,
                self.Oy * other,
                self.Vx * other,
                self.Vy * other,
                self.wz * other,
                self.theta * other,
                self.stab * other,
                self.dstab * other,
                self.Pa * other,
            )
        else:
            return NotImplemented

    def __mul__(self, other):
        return self.__rmul__(other)


class Control:
    def __init__(self, stab, throttle):
        self.stab = stab
        self.throttle = throttle

    def __rmul__(self, other):
        if np.isscalar(other):
            return Control(
                self.stab * other,
                self.throttle * other,
            )
        else:
            return NotImplemented

    def __mul__(self, other):
        return self.__rmul__(other)
