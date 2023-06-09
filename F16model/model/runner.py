import numpy as np
from F16model.model.ODE_3DoF import solve
from . import States


class Env:

    """Main class for interacting with F-16 model"""

    def __init__(self, x0, dt=0.001):
        """TODO: to be defined."""
        self.dt = dt  #
        self._x_full = [x0]

    def step(self, u_i):
        x_full_i_1 = self._x_full[-1] + self.dt * solve(self._x_full[-1], u_i)
        if np.isnan(x_full_i_1.Ox):
            return False
        self._x_full.append(x_full_i_1)
        return x_full_i_1

    def get_states(self):
        return self._x_full[1::]
