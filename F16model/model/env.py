import numpy as np

from ._interface import States, Control, F16model
from .engine import find_correct_thrust_position
import F16model.data.plane as plane


class F16:

    """RL like enviroment for F16model"""

    def __init__(
        self, init_state: np.ndarray, init_control: np.ndarray, norm_state=False
    ):
        self.dt = 0.02  # simulation step
        self.tn = 10  # finish time
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
        self.model = F16model(self.init_state, self.dt)
        self.total_return = 0
        self.episode_length = 0
        self.prev_done = False
        self.norm_state = norm_state

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = Control(action[0], action[1])
        else:
            raise ValueError(f"Action has type '{type(action)}' should be np.array")

        if self.prev_done:
            self.prev_done = False
            _ = self.reset()

        state = self.model.step(action)

        self.clock += self.dt
        self.episode_length += 1

        reward = self.compute_reward(state, action)
        out_state = self.state_transform(state)

        self.total_return += reward
        info = {
            "episode_length": self.episode_length,
            "total_return": self.total_return,
        }
        return out_state, reward, self.done, self.clock, info

    def compute_reward(self, state, action):
        a_w, a_V, a_y, a_theta, a_stab, a_throttle = 1e-3, 1e-2, 1e-2, 1e-3, 1e-3, 1e-2
        reward = 0
        if self.clock >= self.tn - self.dt:
            reward += 100
            print("clock done")
            self.done = True

        if state.Oy <= 0:
            reward -= 100
            print("Oy done")
            self.done = True

        if np.radians(-20) < state.alpha < np.radians(45):
            reward += (self.episode_length**0.5) / 2
        else:
            reward -= 100
            print("alpha done")
            self.done = True

        if self.done:
            reward_s = -(
                a_w * (state.wz - self.init_state.wz) ** 2
                + a_V * (state.V - self.init_state.V) ** 2
                + a_y * (state.Oy - self.init_state.Oy) ** 2
                + a_theta * (state.theta - self.init_state.theta) ** 2
            )
            reward_a = -(
                a_stab * (action.stab) ** 2 + a_throttle * (action.throttle) ** 2
            )
            reward += reward_s + reward_a
            self.prev_done = True
        else:
            reward_a = -(
                a_stab * (action.stab) ** 2 + a_throttle * (action.throttle) ** 2
            )
            reward += reward_a
        reward = 0.3 * min(np.tanh(reward), 0) + 5.0 * max(np.tanh(reward), 0)
        return reward

    def state_transform(self, state):
        state_short = {
            k: vars(state)[k]  for k, _ in plane.state_restrictions.items()  if k in vars(state) 
        } # take keys that defines in state_restrictions from `State` class
        state_short = np.array(list(state_short.values()))
        if self.norm_state:
            state_short = F16.normalize(state_short)
        return state_short

    def normalize(truncated_state):
        norm_values = []
        for i, values in enumerate(plane.state_restrictions.values()):
            norm_values.append(minmaxscaler(truncated_state[i], values[0], values[1]))
        return np.array(norm_values)

    def denormalize(state_norm):
        norm_values = []
        for i, values in enumerate(plane.state_restrictions.values()):
            norm_values.append(minmaxscaler(state_norm[i], values[0], values[1], inverse_transform=True))
        return np.array(norm_values)

    def reset(self):
        init_state = self.model.reset()
        init_state = self.state_transform(self, init_state)
        self.total_return = 0
        self.episode_length = 0
        self.clock = 0
        self.done = False
        return init_state


def minmaxscaler(value, min_value, max_value, inverse_transform=False):
    if inverse_transform:
        return value * (max_value - min_value) + min_value
    return (value - min_value) / (max_value - min_value)


def run_episode(init_state, init_action, max_steps=2000):
    """Example of running F16 model"""
    env = F16(init_state, init_action)
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
    """Short cut insted of calculating from trim_app.py"""
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
