import numpy as np
import torch
import torch.nn as nn


from F16model.model import States
from F16model.env import F16, get_trimmed_state_control
from F16model.data import plane
import F16model.utils.plots as utils_plots
import F16model.utils.control as utils_control
from ppo_model import Agent

model_name = "runs/models/F16__ppo_train__1__1700912494_244b"
CONST_STEP = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_CONFIG = {
    "dt": 0.01,
    "tn": 10,
    "norm_state": True,
    "debug_state": False,
}


def run_sim(x0, u0, max_episode=2000):
    env = F16(ENV_CONFIG)
    action_size = 2
    state = env.reset()
    agent = Agent(state.shape[0], action_size)
    agent.load(model_name)

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
    for _ in t:
        state = torch.Tensor(state).to(device).reshape(-1, state.shape[0])
        action, _, _, _ = agent.get_action_and_value(state)

        action = env.rescale_action(action)
        state, reward, done, _, _ = env.step(action)
        if state.all():
            states.append(state)
            rewards.append(reward)
            actions.append(action)
        if done:
            print("FOK")
            break
    states = list(map(F16.denormalize, states))
    return states, actions, env.ref_signal, sum(rewards), t


if __name__ == "__main__":
    x0, u0 = get_trimmed_state_control()
    states, actions, ref_signal, r, t = run_sim(x0, u0)
    print(f"total reward {r}")
    utils_plots.result(states, actions, t[1:-1], "test_agent", ref_signal)
