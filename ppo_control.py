import numpy as np
import torch
import torch.nn as nn

from make_enviroment import Env, get_trimmed_state_control
import F16model.utils.plots as utils_plots
import F16model.utils.control as utils_control
from model import Agent


model_name = "models/F16__ppo_trian__1__1698157345_fee7"


CONST_STEP = True


def run_sim(x0, u0, max_episode=2000):
    env = Env(x0, u0)

    action_size = 2
    state = env.reset()
    agent = Agent(state.shape[0], action_size)
    agent.load(model_name)

    t0 = 0
    dt = env.dt
    tn = env.tn
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
    times = []
    for i, _ in enumerate(t):
        state = torch.Tensor(state).reshape(-1, state.shape[0])
        state = torch.nn.functional.normalize(state, p=2, dim=-1)

        action, _, _, _ = agent.get_action_and_value(state)
        action = action.numpy()[0]
        action = np.clip(action, np.radians(-25), np.radians(25))
        state, reward, done, current_time, _ = env.step(action)
        if state.all():
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            times.append(current_time)
        if done:
            print("FOK")
            break
    return states, actions, sum(rewards), times


if __name__ == "__main__":
    x0, u0 = get_trimmed_state_control()
    states, actions, r, t = run_sim(x0, u0)
    print(f"total reward {r}")
    utils_plots.result(states, actions, t, "test_name")
