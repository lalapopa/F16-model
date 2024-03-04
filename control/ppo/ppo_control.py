import torch
import gymnasium as gym
import numpy as np
import random

from F16model.env import F16
import F16model.utils.plots as utils_plots
import F16model.utils.control_metrics as utils_metrics
from utils import parse_args

from ppo_train_gsde import make_env
from ppo_model_gsde import Agent

model_name = "runs/old_reward/F16__1__1709496234__d000"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_sim():
    args = parse_args()
    # args.seed = 289  # 7619 -18 reward OMG
    # args.seed = 251
    args.seed = random.randint(1, 999)
    print(f"Run with seed = {args.seed}")
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.seed + i, ENV_CONFIG) for i in range(1)]
    )

    agent = Agent(envs, args)
    agent.load(model_name)
    state, _ = agent.env.reset()
    print(f"Start state: {agent.env.call('init_state')[0]}")

    state = torch.Tensor(state).to(device)
    agent.sample_theta_gsde(state)

    actions = []
    states = []
    rewards = []
    clock = []
    ref_signal = agent.env.call("ref_signal")[0].theta_ref[:-1]
    print(
        f"INIT THETA {np.degrees(F16.denormalize(state[0])[2])} | REF THETA {np.degrees(ref_signal[15])}"
    )
    for _ in range(0, 2048):
        state = torch.Tensor(state).to(device)
        action, _, _, _ = agent.get_action_and_value(state)
        action = action.cpu().detach().numpy()
        state, reward, done, _, info = agent.env.step(action)

        actions.append(F16.rescale_action(action[0]))
        rewards.append(reward[0])
        if done:
            clock.append(info["final_info"][0]["clock"])
            states.append(F16.denormalize(info["final_observation"][0]))
            break
        else:
            clock.append(info["clock"][0])
            states.append(F16.denormalize(state[0]))
    if len(ref_signal) < len(states):
        cut_index = len(ref_signal)
    else:
        cut_index = len(states)
    return (
        states[:cut_index],
        actions[:cut_index],
        ref_signal[:cut_index],
        rewards[:cut_index],
        clock[:cut_index],
    )


if __name__ == "__main__":
    for i in range(1, 10):
        #        init_state = np.array([0, 3000, 0, 0, 200, 0])
        #        init_control = np.array([0, 0])
        ENV_CONFIG = {
            "dt": 0.01,
            "tn": 10,
            "debug_state": False,
            "determenistic_ref": True,
            "scenario": None,
        }
        #        ENV_CONFIG["init_state"] = init_state
        #        ENV_CONFIG["init_control"] = init_control
        states, actions, ref_signal, r, t = run_sim()
        print(f"total reward {sum(r)}")
        print(f"nMAE: { utils_metrics.nMAE(ref_signal, [i[2] for i in states])}")
        utils_plots.result(states, actions, t, ref_signal=ref_signal, reward=r)
#   utils_plots.algo(r, t)
