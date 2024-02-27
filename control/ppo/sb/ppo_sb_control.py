import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from F16model.env import F16, get_trimmed_state_control
import F16model.utils.plots as utils_plots
import F16model.utils.control_metrics as metrics

CONST_STEP = True
model_name = "runs/models/F16__sb__1__1708971602_6628.zip"
ENV_CONFIG = {
    "dt": 0.01,
    "tn": 10,
    "norm_state": True,
    "debug_state": False,
    "determenistic_ref": False,
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
        use_sde=True,
    )
    model = PPO.load(model_name)
    state = vec_env.reset()
    n = 2048

    actions = []
    states = []
    rewards = []
    clock = []
    done = False
    ref_signal = vec_env.get_attr("ref_signal")
    for _ in range(0, n):
        action, _ = model.predict(state)
        (state, reward, done, info) = vec_env.step(action)
        if done[0]:
            print(f"Done! {done} {reward} {state}|{states[0]}")
            break
        states.append(state[0])
        rewards.append(reward[0])
        actions.append(action[0])
        clock.append(info[0]["clock"])
    states = list(map(F16.denormalize, states))
    actions = list(map(F16.rescale_action, actions))
    return (
        states,
        actions,
        ref_signal[0].theta_ref[:-1],
        rewards,
        clock,
    )


if __name__ == "__main__":
    seed = random.randint(1, 999)
    random.seed(seed)
    np.random.seed(seed)

    states, actions, ref_signal, r, t = run_sim()
    print(f"total reward {sum(r)}")
    #    utils_plots.result(
    #        states, actions, t, "agent_control_last_50", ref_signal, cut_index=-50
    #    )
    utils_plots.result(states, actions, t, ref_signal=ref_signal, reward=r)

    theta = np.degrees([i[2] for i in states])
    theta_ref = np.degrees(ref_signal)
    print(f"theta nMAE = {metrics.nMAE(theta_ref, theta) * 100}%")
