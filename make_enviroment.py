import numpy as np
import time

from F16model.model import States, Control, interface
from F16model.model.engine import find_correct_thrust_position
import F16model.utils.plots as utils_plots


class Env:

    """RL like enviroment for F16model"""

    def __init__(self, init_state: States):
        self.dt = 0.02  # simulation step
        self.tn = 20  # finish time
        self.clock = 0
        self.done = False
        self.init_state = init_state
        self.model = interface.F16model(init_state, self.dt)

    def step(self, action: Control):
        state = self.model.step(action)
        self.clock += self.dt
        reward = 0.001

        if self.clock > self.tn:
            reward += 50
            self.done = True
        if not state:
            reward -= 50
            self.done = True
        return state, reward, self.done, self.clock

    def reset():
        init_state = self.model.reset()
        return init_state


def run_episode(init_state: States, max_steps=2000):
    env = Env(init_state)
    actions = []
    states = []
    rewards = []
    times = []
    for t in range(1, max_steps):
        action = get_action()
        state, reward, done, current_time = env.step(action)
        if state:
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            times.append(current_time)
        if done:
            break
    return states, actions, sum(rewards), times


def get_action():
    return Control(np.radians(np.random.uniform(-25, 25)), np.random.uniform(0, 1))


def get_trimmed_state_control():
    u_trimmed = Control(
        np.radians(-4.3674), 0.3767
    )  # Trimmed values for V = 200 m/s, H = 3000 m
    x0 = States(
        Ox=0,
        Oy=3000,
        wz=0,
        theta=np.radians(2.7970),
        V=200,
        alpha=np.radians(2.7970),
        stab=u_trimmed.stab,
        dstab=np.radians(0),
        Pa=find_correct_thrust_position(u_trimmed.throttle),
    )
    return x0, u_trimmed


if __name__ == "__main__":
    x0, _ = get_trimmed_state_control()
    states, actions, reward, t = run_episode(x0)
    print(f"TOTAL REWARD = {reward}, TOTAL TIME = {t[-1]}")
    utils_plots.result(states, actions, t)
