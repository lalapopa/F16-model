import numpy as np

from F16model.model.ODE_3DoF import solve


class States:
    def __init__(self, Ox, Oy, wz, theta, V, alpha, stab, dstab, Pa):
        self.Ox = Ox  # m
        self.Oy = Oy  # m
        self.wz = wz  # rad/s
        self.theta = theta  # rad
        self.V = V  # m/s
        self.alpha = alpha  # rad
        self.stab = stab  # rad
        self.dstab = dstab  # rad/s
        self.Pa = Pa  # 0 to 1

    def to_array(self):
        return np.array(
            [
                self.Ox,
                self.Oy,
                self.wz,
                self.theta,
                self.V,
                self.alpha,
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
                other.wz + self.wz,
                other.theta + self.theta,
                other.V + self.V,
                other.alpha + self.alpha,
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
                self.wz * other,
                self.theta * other,
                self.V * other,
                self.alpha * other,
                self.stab * other,
                self.dstab * other,
                self.Pa * other,
            )
        elif isinstance(other, States):
            return States(
                self.Ox * other.Ox,
                self.Oy * other.Oy,
                self.wz * other.wz,
                self.theta * other.theta,
                self.V * other.V,
                self.alpha * other.alpha,
                self.stab * other.stab,
                self.dstab * other.dstab,
                self.Pa * other.Pa,
            )
        else:
            return NotImplemented

    def __mul__(self, other):
        return self.__rmul__(other)

    def __repr__(self):
        return f"Ox = {self.Ox} m; Oy = {self.Oy} m; wz = {np.degrees(self.wz)} deg/s; V = {self.V}; theta = {np.degrees(self.theta)}; stab_pos = {np.degrees(self.stab)} deg; dstab = {np.degrees(self.dstab)} deg/s; thrust = {self.Pa} H %"


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
        return f"stab = {np.degrees(self.stab)} deg; throttle = {self.throttle};"


class F16model:
    """Bearbone interface for calculating next state"""

    def __init__(self, x0: States, dt=0.001):
        self.state_prev = x0
        self.init_state = x0
        self.dt = dt

    def step(self, u_i: Control):
        next_state = self.state_prev + self.dt * solve(self.state_prev, u_i)
        clip_wz = np.clip(next_state.wz, np.radians(-60), np.radians(60))
        next_state.V = self.init_state.V
        next_state.wz = clip_wz
        self.state_prev = next_state
        return next_state

    def reset(self):
        self.state_prev = self.init_state
        return self.state_prev
