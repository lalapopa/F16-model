import numpy as np
from F16model.model.ODE_3DoF import solve
from . import States


class F16model:

    """Main class for interacting with F-16 model"""

    def __init__(self, x0, dt=0.001):
        self.state_prev = x0
        self.init_state = x0
        self.dt = dt

    def step(self, u_i):
        next_state = self.state_prev + self.dt * solve(self.state_prev, u_i)
        if np.isnan(next_state.Ox):
            return False
        self.state_prev = next_state
        return next_state

    def reset(self):
        self.states_prev = self.init_state
        return self.states_prev
