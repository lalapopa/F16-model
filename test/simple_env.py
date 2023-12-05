import numpy as np
import random
import gymnasium as gym


class DummestEnv(gym.Env):
    def __init__(self):
        self.history = []
        self.action_space = gym.spaces.Box(-1, +1, (1,), dtype=np.float32)
        low = np.array([-1], dtype=np.int32)
        high = np.array([1], dtype=np.int32)
        self.observation_space = gym.spaces.Box(low, high)
        self.clock = 0
        self._dt = 0.1
        self.prev_state = None

    def step(self, action):
        self.clock += self._dt
        if action > self.prev_state:
            reward = 1
        else:
            reward = 0
        self.history.append({f"{round(self.clock, 2)}": [action, self.prev_state]})
        self.prev_state = self._get_obs()
        if self.clock >= 10:
            terminated = True
        else:
            terminated = False
        return self.prev_state, reward, terminated, False, {}

    def _get_obs(self):
        return np.array([random.randint(-1, 0)], dtype=np.float32)

    def render(self):
        return self.history

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.history = []
        self.clock = 0
        self.prev_state = self._get_obs()

        return self.prev_state, {}
