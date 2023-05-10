import numpy as np

import utils


class States:
    def __init__(self, Ox, Oy, Vx, Vy, wz, theta, stab, dstab, Pa):
        self.Ox = Ox  # m
        self.Oy = Oy  # m
        self.Vx = Vx  # m
        self.Vy = Vy  # m
        self.wz = wz  # rad/s
        self.theta = theta  # rad
        self.stab = stab  # rad
        self.dstab = dstab  # rad/s
        self.Pa = Pa  # 0 to 1

    def to_array(self):
        return np.array(
            [
                self.Ox,
                self.Oy,
                self.Vx,
                self.Vy,
                self.wz,
                self.theta,
                self.stab,
                self.dstab,
                self.Pa,
            ]
        )

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
        elif isinstance(other, States):
            return States(
                self.Ox * other.Ox,
                self.Oy * other.Oy,
                self.Vx * other.Vx,
                self.Vy * other.Vy,
                self.wz * other.wz,
                self.theta * other.theta,
                self.stab * other.stab,
                self.dstab * other.dstab,
                self.Pa * other.Pa,
            )
        else:
            return NotImplemented

    def __mul__(self, other):
        return self.__rmul__(other)

    def __repr__(self):
        return f"Ox = {self.Ox} m;\nOy = {self.Oy} m;\nwz = {self.wz} m/s;\nVx = {self.Vx} m/s;\ntheta = {self.theta};\nstab_pos = {np.degrees(self.stab)} deg;\ndstab = {self.dstab} deg/s;\nthrust = {self.Pa} H?"


class Control:
    def __init__(self, stab, throttle):
        self.stab = stab  # rad
        self.throttle = throttle  # from 0 to 1

    def to_array(self):
        return np.array([self.stab, self.throttle])

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

    def __repr__(self):
        return f"stab = {np.degrees(self.stab)} deg;\nthrottle = {self.throttle};"
