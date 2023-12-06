import os
import numpy as np
import random

from F16model.model import States, Control, F16model
from F16model.env.task import ReferenceSignal
from F16model.model.engine import find_correct_thrust_position
from F16model.data import plane
from F16model.utils.calc import normalize_value, minmaxscaler


class F16:

    """Gym like enviroment for F16model"""

    def __init__(self, config):
        self.dt = config["dt"]  # simulation step
        self.tn = config["tn"]  # finish time
        self.norm_state = config["norm_state"]
        if config["debug_state"]:  # init_state should class States
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
        self.config = config
        self.ref_signal = ReferenceSignal(
            0, self.dt, self.tn, self.config["determenistic_ref"]
        )

        self.clock = 0
        self.total_return = 0
        self.episode_length = 0
        self.done = False
        self.prev_done = False
        self.prev_action = Control(0, 0)

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = Control(action[0], action[1])
        else:
            raise TypeError(f"Action has type '{type(action)}' should be np.array")

        if self.prev_done:
            self.prev_done = False
            _ = self.reset()

        state = self.model.step(action)
        self.clock += self.dt
        self.episode_length += 1

        reward = self.check_state(state)  # If fly out of bound give -1000 reward
        reward += self.compute_reward(state)

        out_state = self.state_transform(state)
        self.total_return += reward

        self.prev_action = action

        info = {
            "episode_length": self.episode_length,
            "total_return": self.total_return,
        }
        if self.done:
            self.prev_done = True
        return out_state, reward, self.done, self.clock, info

    def compute_reward(self, state):
        tracking_ref = np.array(
            [
                self.ref_signal.theta_ref[self.episode_length],
                self.init_state.V,
            ]
        )
        tracking_err = tracking_ref - np.array([state.theta, state.V])
        tracking_Q = np.array([1 / np.radians(30), 1 / 240])

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
        """
        Return states are :
        Oy
        wz
        theta
        V
        alpha
        theta - theta_ref
        V - V_ref
        """
        state_short = {
            k: vars(state)[k] for k, _ in plane.state_bound.items() if k in vars(state)
        }  # take keys that defines in state_boundfrom `States` class
        state_short = list(state_short.values())
        if self.norm_state:
            state_short = F16.normalize(state_short)

        theta_err = state.theta - self.ref_signal.theta_ref[self.episode_length]
        v_err = float(state.V - self.init_state.V)
        state_short.append(theta_err)
        state_short.append(v_err)
        return np.array(state_short)

    def check_state(self, state):
        reward = 0
        if state.Oy <= 300:
            reward += -1000
            self.done = True

        if state.Oy >= 30000:
            reward += -1000
            self.done = True

        if self.episode_length == (self.tn / self.dt) - 1:
            self.done = True
        return reward

    def normalize(truncated_state):
        norm_values = []
        for i, values in enumerate(plane.state_bound.values()):
            norm_values.append(minmaxscaler(truncated_state[i], values[0], values[1]))
        return norm_values

    def denormalize(state_norm):
        norm_values = []
        for i, values in enumerate(plane.state_bound.values()):
            norm_values.append(
                minmaxscaler(
                    state_norm[i], values[0], values[1], inverse_transform=True
                )
            )
        norm_values = np.array(norm_values)
        if i < len(state_norm) - 1:
            additional_states = state_norm[i + 1 :]
            norm_values = np.concatenate((norm_values, additional_states))
        return norm_values

    def reset(self, seed=None):
        if seed:
            np.random.seed(seed)
            random.seed(seed)
        self.init_state = get_random_state()
        self.model = F16model(self.init_state, self.dt)
        self.ref_signal = ReferenceSignal(
            0,
            self.dt,
            self.tn,
            self.config["determenistic_ref"],
        )
        self.init_state = self.model.reset()
        self.episode_length = 0
        self.total_return = 0
        self.clock = 0
        self.done = False
        out_state = self.state_transform(self.init_state)
        return out_state

    def rescale_action(self, action):
        """
        Rescale action [stab, throttle]
        """
        action = action.cpu().numpy()[0]
        action = np.clip(action, -1, 1)
        stab_rescale = normalize_value(
            action[0],
            -plane.maxabsstab,
            plane.maxabsstab,
        )
        throttle_rescale = normalize_value(action[1], 0, 1)
        return np.array([stab_rescale, throttle_rescale])

    def action_size(self):
        return self.prev_action.to_array().size

    def state_size(self):
        return self.state_transform(self.init_state).size


def get_action():
    return Control(np.radians(np.random.uniform(-25, 25)), np.random.uniform(0, 1))


def get_random_state():
    alpha_angle_random = np.random.uniform(
        plane.random_state_bound["alpha"][0], plane.random_state_bound["alpha"][1], 1
    )[0]
    return States(
        Ox=0,
        Oy=np.random.uniform(
            plane.random_state_bound["Oy"][0],
            plane.random_state_bound["Oy"][1],
            1,
        )[0],
        wz=np.random.uniform(
            plane.random_state_bound["wz"][0], plane.random_state_bound["wz"][1], 1
        )[0],
        theta=alpha_angle_random,
        V=np.random.uniform(
            plane.random_state_bound["V"][0], plane.random_state_bound["V"][1], 1
        )[0],
        alpha=alpha_angle_random,
        stab=0,
        dstab=0.0,
        Pa=find_correct_thrust_position(0),
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