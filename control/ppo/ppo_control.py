import numpy as np
import torch
import torch.nn as nn


from F16model.model import States
from F16model.env import F16, get_trimmed_state_control
from F16model.data import plane
import F16model.utils.plots as utils_plots
import F16model.utils.control as utils_control
import F16model.utils.control_metrics as utils_metrics

from ppo_model_gsde import Agent

model_name = "runs/models/F16__utils__1__1706536287_cf62"

CONST_STEP = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_CONFIG = {
    "dt": 0.01,
    "tn": 10,
    "norm_state": True,
    "debug_state": False,
    "determenistic_ref": False,
}


def run_sim(x0, u0, max_episode=2000):
    env = F16(ENV_CONFIG)
    action_size = np.array(env.action_space.shape).prod()
    obs_size = np.array(env.observation_space.shape).prod()
    print(obs_size, action_size)
    agent = Agent(obs_size, action_size).to(device)
    agent.load(model_name)
    state, _ = env.reset()
    print(f"Start state: {env.init_state}")
    state = torch.Tensor(state).to(device).reshape(-1, obs_size)
    agent.sample_theta_gsde(state)
    actions = []
    states = []
    rewards = []
    clock = []
    for _ in range(0, 2048):
        state = torch.Tensor(state).to(device).reshape(-1, obs_size)
        action, _, _, _ = agent.get_action_and_value(state)
        action = action.cpu().detach().numpy()
        state, reward, done, _, info = env.step(action)
        if state.all():
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            clock.append(info["clock"])
        if done:
            print("FOK")
            break
    states = list(map(F16.denormalize, states))
    actions = list(map(F16.rescale_action, actions))
    return states, actions, env.ref_signal.theta_ref, sum(rewards), clock


if __name__ == "__main__":
    x0, u0 = get_trimmed_state_control()
    states, actions, ref_signal, r, t = run_sim(x0, u0)
    print(f"total reward {r}")
    print(f"nMAE: { utils_metrics.nMAE(ref_signal, [i[2] for i in states])}")
    utils_plots.result(states, actions, t, ref_signal=ref_signal)
