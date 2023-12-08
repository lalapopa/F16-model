import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


from F16model.env import F16, get_trimmed_state_control
import F16model.utils.plots as utils_plots

CONST_STEP = True
model_name = "runs/models/F16__utils__sb__1__1701981653_0b70.zip"
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


def run_sim(x0, u0, max_episode=2000):
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
    print(state)
    for _ in range(0, n):
        action, s = model.predict(state)
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
    return states, actions, ref_signal[0].theta_ref, sum(rewards), clock


if __name__ == "__main__":
    x0, u0 = get_trimmed_state_control()
    states, actions, ref_signal, r, t = run_sim(x0, u0)
    print(f"total reward {r}")
    utils_plots.result(states, actions, t, "test_agent", ref_signal)
