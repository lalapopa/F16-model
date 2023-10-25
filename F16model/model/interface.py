import numpy as np
from F16model.model.ODE_3DoF import solve
from . import States, Control


class F16model:

    """Main class for interacting with F-16 model"""

    def __init__(self, x0: States, dt=0.001):
        self.state_prev = x0
        self.init_state = x0
        self.dt = dt

    def step(self, u_i: Control):
        next_state = self.state_prev + self.dt * solve(self.state_prev, u_i)
        self.state_prev = next_state
        return next_state

    def reset(self):
        self.state_prev = self.init_state
        return self.state_prev.to_array()
