import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from F16model.env import F16, get_trimmed_state_control
import F16model.utils.plots as utils_plots
import F16model.utils.control_metrics as metrics

CONST_STEP = True
model_name = "runs/models/F16__utils__sb__1__1702210304_07e9.zip"
ENV_CONFIG = {
    "dt": 0.01,
    "tn": 10,
    "norm_state": True,
    "debug_state": False,
    "determenistic_ref": True,
}


def env_wrapper():
    env = F16(ENV_CONFIG)
    return env


def run_sim():
    vec_env = make_vec_env(env_wrapper, n_envs=1)
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        use_sde=False,
    )
    model = PPO.load(model_name)
    state = vec_env.reset()
    n = 2048

    actions = []
    states = []
    rewards = []
    clock = []
    done = False
    for _ in range(0, n):
        action, _ = model.predict(state)
        (state, reward, done, info) = vec_env.step(action)
        if state[0].all():
            states.append(state[0])
            rewards.append(reward[0])
            actions.append(action[0])
            clock.append(info[0]["clock"])
        if done[0]:
            print(f"Done! {done} {reward} {state}|{states[0]}")
            break
    states = list(map(F16.denormalize, states))
    actions = list(map(F16.rescale_action, actions))
    ref_signal = vec_env.get_attr("ref_signal")
    return (
        states[:-1],
        actions[:-1],
        ref_signal[0].theta_ref[:-1],
        rewards[:-1],
        clock[:-1],
    )


if __name__ == "__main__":
    seed = 1
    random.seed(seed)
    np.random.seed(seed)

    states, actions, ref_signal, r, t = run_sim()
    print(f"total reward {sum(r)}")
    utils_plots.result(
        states, actions, t, "agent_control_last_50", ref_signal, cut_index=-50
    )
    utils_plots.result(states, actions, t, "agent_control", ref_signal)
    utils_plots.algo(r, t, plot_name="agent_reward")
    theta = np.degrees([i[2] for i in states])
    theta_ref = np.degrees(ref_signal)
    print(f"theta nMAE = {metrics.nMAE(theta_ref, theta) * 100}%")
