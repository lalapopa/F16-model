import numpy as np

from F16model.model import States, Control, interface
from F16model.model.engine import find_correct_thrust_position
import F16model.utils.plots as utils_plots


class Env:

    """RL like enviroment for F16model"""

    def __init__(self, init_state: np.ndarray, init_control: np.ndarray):
        self.dt = 0.02  # simulation step
        self.tn = 30  # finish time
        self.clock = 0
        self.done = False
        self.init_state = States(
            Ox=init_state[0],
            Oy=init_state[1],
            wz=init_state[2],
            theta=init_state[3],
            V=init_state[4],
            alpha=init_state[5],
            stab=init_control[0],
            dstab=np.radians(0),
            Pa=find_correct_thrust_position(init_control[1]),
        )
        self.model = interface.F16model(self.init_state, self.dt)
        self.total_return = 0
        self.episode_length = 0

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = Control(action[0], action[1])
        else:
            raise ValueError(f"Action has type '{type(action)}' should be np.array")
        state = self.model.step(action)
        reward = 0

        #        print(f"TIME:{self.clock}| {state}")
        self.clock += self.dt
        reward = -sum(state.to_array()[:6] - self.init_state.to_array()[:6]) * 0.01

        if self.clock >= self.tn:
            reward += 50
            self.done = True

        if np.radians(-20) < state.alpha < np.radians(45):
            reward += 0.1
        else:
            reward -= 100
            self.done = True

        self.episode_length += 1
        self.total_return += reward
        info = {
            "episode_length": self.episode_length,
            "total_return": self.total_return,
        }

        out_state = state.to_array()[
            0:6
        ]  # dont need return last 3 states its only for internal use

        return out_state, reward, self.done, self.clock, info

    def reset(self):
        init_state = self.model.reset()[0:6]  # same here we dont need 3 last states
        self.total_return = 0
        self.episode_length = 0
        return init_state


def run_episode(init_state, init_action, max_steps=2000):
    env = Env(init_state, init_action)
    actions = []
    states = []
    rewards = []
    times = []
    for t in range(1, max_steps):
        # action = get_action()
        action = np.array([0, 0])
        state, reward, done, current_time, _ = env.step(action)  # give as numpy array
        if state.all():
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
    return x0.to_array(), u_trimmed.to_array()


if __name__ == "__main__":
    x0, u0 = get_trimmed_state_control()
    states, actions, reward, t = run_episode(x0, u0)
    print(f"TOTAL REWARD = {reward}, TOTAL TIME = {t[-1]}")
    utils_plots.result(states, actions, t)
