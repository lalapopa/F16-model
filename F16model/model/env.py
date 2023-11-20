import numpy as np

from ._interface import States, Control, F16model
from .engine import find_correct_thrust_position
import F16model.data.plane as plane


class F16:

    """RL like enviroment for F16model"""

    def __init__(self, init_state=None, init_control=None, norm_state=False, debug_state=False):
        self.dt = 0.01  # simulation step
        self.tn = 20  # finish time
        self.clock = 0
        self.done = False
        if debug_state: # init state should be in States 
            self.init_state = init_state
        else:
            if (init_state and init_control) == None:
                self.init_state = get_random_state()
            else:
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
        self.episode_length = 1
        self.prev_done = False
        self.prev_action = Control(0, 0)
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
        self.check_state(state)

        reward = self.compute_reward(state, action)
        out_state = self.state_transform(state)
        self.total_return += reward
        self.episode_length += 1
        self.prev_action = action

        info = {
            "episode_length": self.episode_length,
            "total_return": self.total_return,
        }
        if self.done:
            self.prev_done = True
        return out_state, reward, self.done, self.clock, info

    def compute_reward(self, state, action):
        target_oy = self.init_state.Oy
        target_wz = 0
        target_V = self.init_state.V
#       state_coeff = np.array([1/10000, np.radians(1/40), 1/500]) # 8501, 5a80
        state_coeff = np.array([1/50, np.radians(1/0.1), 1/6])

        state_err = np.array([
            state.Oy - target_oy,
            state.wz - target_wz,
            target_V - target_V,
            ])
        reward = -1/3*np.abs(np.clip(state_err * state_coeff, [-1,-1,-1], [1, 1, 1])).sum()
        return reward

    def state_transform(self, state):
        state_short = {
            k: vars(state)[k]
            for k, _ in plane.state_restrictions.items()
            if k in vars(state)
        }  # take keys that defines in state_restrictions from `State` class
        state_short = np.array(list(state_short.values()))
        if self.norm_state:
            state_short = F16.normalize(state_short)
        return state_short

    def check_state(self, state):
        if state.Oy <= plane.state_restrictions["Oy"][0]-1000:
            self.done = True

        if state.Oy >= plane.state_restrictions["Oy"][1]+1000:
            self.done = True

        if self.episode_length >= self.tn / self.dt:
            self.done = True

    def normalize(truncated_state):
        norm_values = []
        for i, values in enumerate(plane.state_restrictions.values()):
            norm_values.append(minmaxscaler(truncated_state[i], values[0], values[1]))
        return np.array(norm_values)

    def denormalize(state_norm):
        norm_values = []
        for i, values in enumerate(plane.state_restrictions.values()):
            norm_values.append(
                minmaxscaler(
                    state_norm[i], values[0], values[1], inverse_transform=True
                )
            )
        return np.array(norm_values)

    def reset(self):
        self.init_state = get_random_state()
        self.model = F16model(self.init_state, self.dt)
        init_state = self.model.reset()
        init_state = self.state_transform(init_state)
        self.total_return = 0
        self.episode_length = 1
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


def get_random_state():
    alpha_angle_random = np.random.uniform(
        plane.state_restrictions["alpha"][0], plane.state_restrictions["theta"][1], 1
    )[0]
    return States(
        Ox=0,
        Oy=np.random.uniform(
            plane.state_restrictions["Oy"][0]+500,
            plane.state_restrictions["Oy"][1]-1000,
            1,
        )[0],
        wz=np.random.uniform(
            plane.state_restrictions["wz"][0], plane.state_restrictions["wz"][1], 1
        )[0],
        theta=alpha_angle_random,
        V=np.random.uniform(
            plane.state_restrictions["V"][0] + 50, plane.state_restrictions["V"][1], 1
        )[0],
        alpha=alpha_angle_random,
        stab=np.random.uniform(-plane.maxabsstab, plane.maxabsstab, 1)[0],
        dstab=np.random.uniform(-plane.maxabsdstab, plane.maxabsdstab, 1)[0],
        Pa=find_correct_thrust_position(np.random.uniform(0.2, 1, 1)[0]),
    )


def get_trimmed_state_control():
    """Short cut insted of calculating from trim_app.py"""
    u_trimmed = Control(
        np.radians(-4.3674), 0.3767
    )  # Trimmed values for V = 200 m/s, H = 3000 m
    x0 = States(
        Ox=0,
        Oy=2000,
        wz=0,
        theta=np.radians(2.7970),
        V=90,
        alpha=np.radians(2.7970),
        stab=u_trimmed.stab,
        dstab=np.radians(0),
        Pa=find_correct_thrust_position(u_trimmed.throttle),
    )
    return x0.to_array(), u_trimmed.to_array()
