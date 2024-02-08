import numpy as np
import random

from F16model.env import F16, get_trimmed_state_control
import F16model.utils.plots as utils_plots
import F16model.utils.control as utils_control

CONST_STEP = False
ENV_CONFIG = {
    "dt": 0.01,
    "tn": 10,
    "norm_state": True,
    "debug_state": False,
    "determenistic_ref": False,
}


def run_sim(x0, u0, max_episode=2000):
    env = F16(ENV_CONFIG)
    env.reset()
    t0 = 0
    dt = env.dt
    tn = env.tn
    t = np.arange(t0, tn + dt, dt)

    # Control Define
    if CONST_STEP:
        stab_act = utils_control.step_function(t0, dt, tn, 0, 0, bias=u0[0])
        throttle_act = utils_control.step_function(t0, dt, tn, 0, 0, bias=u0[1])
    else:
        stab_act = utils_control.make_step_series(
            t0, dt, tn, 1, step_time=1, hold_time=1, bias=u0[0]
        )
        throttle_act = utils_control.step_function(t0, dt, tn, 5, 1, bias=u0[1])

    actions = []
    states = []
    rewards = []
    times = []
    ref_signal = env.ref_signal.theta_ref[:-1]
    for i, _ in enumerate(t):
        action = np.array([stab_act[i], throttle_act[i]])
        state, reward, done, _, info = env.step(action)
        if state.all():
            states.append(state)
            rewards.append(reward)
            actions.append(action[0])
            times.append(info["clock"])
        if done:
            states = states[:i]
            actions = actions[:i]
            times = times[:i]
            ref_signal = ref_signal[:i]
            break
    return states, actions, sum(rewards), times, ref_signal


if __name__ == "__main__":
    x0, u0 = get_trimmed_state_control()
    states, actions, r, t, ref = run_sim(x0, u0)
    print(f"total reward {r}")
    utils_plots.result(states, actions, t, ref_signal=ref)
