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
        self.clock += self.dt
        a_w, a_V, a_y, a_theta, a_stab, a_throttle = 1e-3, 1e-2, 1e-2, 1e-3, 1e-3, 1e-2
#        print(
#            f"rew diff {(state.wz - self.init_state.wz) ** 2}, {(state.V - self.init_state.V) ** 2}, {(state.Oy - self.init_state.Oy) ** 2}, {(state.theta - self.init_state.theta) ** 2}"
#        )
#        print(f"rew diff control {(action.stab) ** 2}, {(action.throttle) ** 2}")
        reward = 0 
        if self.clock >= self.tn:
            reward += 100
            print("clock done")
            self.done = True

        if state.Oy <= 0:
            reward -= 100
            print("Oy done")
            self.done = True

        if np.radians(-20) < state.alpha < np.radians(45):
            reward += (self.episode_length ** 0.5) / 2  
#            reward += 1
            pass
        else:
            reward -= 10
            print("alpha done")
            self.done = True

        out_state = state.to_array()[
            1:6
        ]  # dont need return last 3 states and Ox, its only for internal use
        
        out_state = self.normalize(out_state)

        self.episode_length += 1
        if self.done:
            reward_s = -(
                a_w * (state.wz - self.init_state.wz) ** 2
                + a_V * (state.V - self.init_state.V) ** 2
                + a_y * (state.Oy - self.init_state.Oy) ** 2
                + a_theta * (state.theta - self.init_state.theta) ** 2
            )
            reward_a = -(a_stab * (action.stab) ** 2 + a_throttle * (action.throttle) ** 2)
            reward += reward_s + reward_a
        else:
            reward_a = -(a_stab * (action.stab) ** 2 + a_throttle * (action.throttle) ** 2)
            reward += reward_a

        reward = 0.3*min(np.tanh(reward) ,0)+ 5.0 * max(np.tanh(reward),0)

        self.total_return += reward
        info = {
            "episode_length": self.episode_length,
            "total_return": self.total_return,
        }

        return out_state, reward, self.done, self.clock, info
    
    def normalize(self, truncated_state):
        Oy, wz, theta, V, alpha = truncated_state
        norm_values = np.array([
            minmaxscaler(Oy, 0, 15000), 
            minmaxscaler(wz, np.radians(-60), np.radians(60)), 
            minmaxscaler(theta, np.radians(-60), np.radians(60)),
            minmaxscaler(V, 0, 600),
            minmaxscaler(alpha, np.radians(-30), np.radians(60)),
        ])
        return norm_values

    def denormalize(self, state_norm):
        Oy_norm, wz_norm, theta_norm, V_norm, alpha_norm = state_norm
        norm_values = np.array([
            minmaxscaler(Oy_norm, 0, 15000, inverse_transform=True), 
            minmaxscaler(wz_norm, np.radians(-60), np.radians(60), inverse_transform=True), 
            minmaxscaler(theta_norm, np.radians(-60), np.radians(60), inverse_transform=True),
            minmaxscaler(V_norm, 0, 600, inverse_transform=True),
            minmaxscaler(alpha_norm, np.radians(-30), np.radians(60), inverse_transform=True),
        ])
        return norm_values

    def reset(self):
        init_state = self.model.reset()[1:6]  # same here we dont need 3 last states
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
