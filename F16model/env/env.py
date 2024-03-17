import random
import numpy as np
import gymnasium as gym

from F16model.model import States, Control, F16model
from F16model.utils.calc import normalize_value, minmaxscaler
from F16model.env.task import ReferenceSignal
from F16model.model.engine import find_correct_thrust_position
import F16model.data.plane as plane


class F16(gym.Env):
    """
    Gym like enviroment for custom F16model.
    """

    def __init__(self, config):
        self.config = config
        self.dt = float(self.config["dt"])
        self.tn = float(self.config["tn"])
        self.init_state = self._get_init_state()

        self.model = F16model(self.init_state, self.dt)
        self.ref_signal = ReferenceSignal(
            self.dt,
            self.tn,
            determenistic=self.config["determenistic_ref"],
            scenario=self.config["scenario"],
        )
        self._destroy()
        self.action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
        low = -np.ones(self._state_size())
        high = -low
        self.observation_space = gym.spaces.Box(low, high)

    def _destroy(self):
        self.clock = 0
        self.total_return = 0
        self.episode_length = 0
        self.done = False
        self.prev_state = self.init_state
        self.error_integral = 0
        self.compute_reward(self.prev_state)

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = F16.rescale_action(action)
            self.action = Control(action, 0)
        else:
            raise TypeError(f"Action has type '{type(action)}' should be np.array")
        state = self.model.step(self.action)
        self.clock += self.dt
        self.clock = round(self.clock, 4)
        reward = self.check_state(state)  # If fly out of bound give -1000 reward
        reward += self.compute_reward(state)

        out_state = self.state_transform(state)
        self.prev_state = state
        self.total_return += reward
        self.episode_length += 1

        info = {
            "episode_length": self.episode_length,
            "total_return": self.total_return,
            "clock": self.clock,
        }
        return out_state, reward, self.done, False, info

    def compute_reward(self, state):
        tracking_ref = (self.ref_signal.wz_ref[self.episode_length],)
        e = np.degrees(tracking_ref - state.wz)
        k = 1
        asymptotic_error = np.clip(1 - ((np.abs(e) / k) / (1 + (np.abs(e) / k))), a_min=0, a_max=1)
        linear_error = np.clip(1 - (1 / k) * e**2, a_min=0, a_max=1)
        reward = asymptotic_error + 0.04 * linear_error
        return reward.item()

    def state_transform(self, state):
        """
        Return states are :
        Oy
        wz
        V
        wz_ref
        0.5 * (wz_ref - wz)
        """
        state_short = {
            k: vars(state)[k] for k, _ in plane.state_bound.items() if k in vars(state)
        }  # take keys that defines in state_boundfrom `States` class
        state_short = list(state_short.values())
        state_short = F16.normalize(state_short)  # Always output normalized states
        state_short.append(self.ref_signal.wz_ref[self.episode_length])
        state_short.append(
            0.5 * (self.ref_signal.wz_ref[self.episode_length] - state.wz)
        )
        return np.array(state_short)

    def check_state(self, state):
        reward = 0
        if state.Oy <= 300:
            reward += -1000
            self.done = True
        if state.Oy >= 30000:
            reward += -1000
            self.done = True
        if abs(state.wz) >= np.radians(50):
            print(f"Early done in step #{self.episode_length}")
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed:
            random.seed(seed)
            np.random.seed(seed)
        self.init_state = self._get_init_state()
        self.model = F16model(self.init_state, self.dt)
        self.ref_signal = ReferenceSignal(
            self.dt,
            self.tn,
            determenistic=self.config["determenistic_ref"],
            scenario=self.config["scenario"],
        )
        self.init_state = self.model.reset()  # This line useless I guess
        self._destroy()
        out_state = self.state_transform(self.init_state)
        return out_state, {}

    def _get_init_state(self):
        if self.config["debug_state"]:
            state = self.config["init_state"]
        else:
            try:
                if isinstance(self.config["init_state"], np.ndarray) and isinstance(
                    self.config["init_control"], np.ndarray
                ):
                    init_state = self.config["init_state"]
                    init_control = self.config["init_control"]
                    state = States(
                        Ox=init_state[0],
                        Oy=init_state[1],
                        wz=init_state[2],
                        theta=init_state[3],
                        V=init_state[4],
                        alpha=init_state[5],
                        stab=init_control[0],
                        dstab=0,
                        Pa=find_correct_thrust_position(init_control[1]),
                    )
            except KeyError:
                state = get_random_state()
        return state

    def render(self):
        raise UserWarning("TODO: Implement this function")

    def rescale_action(action):
        """
        Rescale action [stab]
        """
        action = np.clip(action[0], -1, 1)
        stab_rescale = normalize_value(
            action,
            -plane.maxabsstab,
            plane.maxabsstab,
            inverse_transform=True,
        )
        return stab_rescale

    def _state_size(self):
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
        dstab=0,
        Pa=find_correct_thrust_position(0),
    )


def get_trimmed_state_control():
    """Shortcut insted of calculating from trim_app.py"""
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
