import numpy as np

from F16model.model import States, Control, F16model
from F16model.env.task import ReferenceSignal
from F16model.model.engine import find_correct_thrust_position
import F16model.data.plane as plane


class F16:

    """RL like enviroment for F16model"""

    def __init__(self, config):
        self.dt = config["dt"]  # simulation step
        self.tn = config["tn"]  # finish time
        self.clock = 0
        self.done = False
        if config["debug_state"]:  # init state should be in States
            self.init_state = config["init_state"]
        else:
            try:
                if isinstance(config["init_state"], np.ndarray) and isinstance(
                    config["init_control"], np.ndarray
                ):
                    init_state = config["init_state"]
                    init_control = config["init_control"]
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
            except KeyError:
                self.init_state = get_random_state()
        self.model = F16model(self.init_state, self.dt)
        self.ref_signal = ReferenceSignal(0, self.dt, self.tn)

        self.total_return = 0
        self.episode_length = 0
        self.prev_done = False
        self.prev_action = Control(0, 0)
        self.norm_state = config["norm_state"]

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
        V_ref_signal = self.init_state.V
        tracking_ref = np.array(
            [
                self.ref_signal.theta_ref[self.episode_length],
                self.init_state.V,
            ]
        )
        tracking_err = tracking_ref - np.array([state.theta, state.V])
        tracking_Q = np.array([1 / np.radians(30), 1 / 1000])

        reward_vec = np.abs(
            np.clip(
                tracking_err @ tracking_Q,
                -np.ones(tracking_err.shape),
                np.ones(tracking_err.shape),
            )
        )
        reward = -1 / 3 * reward_vec.sum()

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
        if state.Oy <= plane.state_restrictions["Oy"][0] - 1000:
            self.done = True

        if state.Oy >= plane.state_restrictions["Oy"][1] + 1000:
            self.done = True

        if self.episode_length == (self.tn / self.dt) - 1:
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
        self.ref_signal = ReferenceSignal(0, self.dt, self.tn)
        init_state = self.model.reset()
        init_state = self.state_transform(init_state)
        self.total_return = 0
        self.episode_length = 1
        self.clock = 0
        self.done = False
        return init_state

    def rescale_action(self, action):
        """
        Rescale action [stab, throttle]
        """
        action = action.cpu().numpy()[0]
        stab_rescale = rescale_value(
            action[0], np.array([-plane.maxabsstab, plane.maxabsstab])
        )
        throttle_rescale = rescale_value(action[1], np.array([0, 1]))
        return np.array([stab_rescale, throttle_rescale])


def rescale_value(value, min_max_range):
    """
    Rescale the action from [-1, 1] to [min_max_range[0], min_max_range[1]]
    """
    low, high = np.float32(min_max_range[0]), np.float32(min_max_range[1])
    return low + 0.5 * (value + 1.0) * (high - low)


def minmaxscaler(value, min_value, max_value, inverse_transform=False):
    if inverse_transform:
        return value * (max_value - min_value) + min_value
    return (value - min_value) / (max_value - min_value)


def get_action():
    return Control(np.radians(np.random.uniform(-25, 25)), np.random.uniform(0, 1))


def get_random_state():
    alpha_angle_random = np.random.uniform(
        plane.state_restrictions["alpha"][0], plane.state_restrictions["theta"][1], 1
    )[0]
    return States(
        Ox=0,
        Oy=np.random.uniform(
            plane.state_restrictions["Oy"][0] + 500,
            plane.state_restrictions["Oy"][1] - 1000,
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
