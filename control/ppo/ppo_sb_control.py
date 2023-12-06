import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


from F16model.model import States
from F16model.env.env_gym import GymF16
from F16model.env import F16, get_trimmed_state_control
from F16model.data import plane
import F16model.utils.plots as utils_plots
import F16model.utils.control as utils_control
from ppo_model import Agent

CONST_STEP = True
model_name = "runs/model/F16__utils__sb__1__1701847427_65f5.zip"
ENV_CONFIG = {
    "dt": 0.01,
    "tn": 10,
    "norm_state": True,
    "debug_state": False,
    "determenistic_ref": False,
}


def env_wrapper():
    env = GymF16(ENV_CONFIG)
    return env


def run_sim(x0, u0, max_episode=2000):
    vec_env = make_vec_env(env_wrapper, n_envs=1)
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        use_sde=True,
    )
    model = PPO.load(model_name)
    state = vec_env.reset()

    t0 = 0
    dt = ENV_CONFIG["dt"]
    tn = ENV_CONFIG["tn"]
    t = np.arange(t0, tn + dt, dt)

    # Control Define
    if CONST_STEP:
        stab_act = np.radians(utils_control.step_function(t0, dt, tn, 0, 4))
        throttle_act = utils_control.step_function(t0, dt, tn, 0, 0.6)
    else:
        stab_act = utils_control.make_step_series(
            t0, dt, tn, np.radians(20), step_time=1, hold_time=1, bias=u0[0]
        )
        throttle_act = utils_control.step_function(t0, dt, tn, 5, 1, bias=u0[1])

    actions = []
    states = []
    rewards = []
    for _ in range(0, 2048):
        action, _ = model.predict(state)
        (
            state,
            reward,
            done,
            _,
        ) = vec_env.step(action)
        if state.all():
            states.append(state[0])
            rewards.append(reward[0])
            actions.append(action[0])
        if done:
            print("FOK")
            break
    states = list(map(GymF16.denormalize, states))
    ref_signal = vec_env.get_attr("ref_signal")[0]
    return states, actions, ref_signal.theta_ref[:-2], sum(rewards), t


if __name__ == "__main__":
    x0, u0 = get_trimmed_state_control()
    states, actions, ref_signal, r, t = run_sim(x0, u0)
    print(f"total reward {r}")
    utils_plots.result(states, actions, t[:-3], "test_agent", ref_signal)
