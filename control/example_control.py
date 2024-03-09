import numpy as np

from F16model.env import F16, get_trimmed_state_control
import F16model.utils.plots as utils_plots
import F16model.utils.control as utils_control
from F16model.utils.calc import normalize_value
import F16model.data.plane as plane

CONST_STEP = True


def run_sim(u0):
    env = F16(ENV_CONFIG)
    env.reset()
    t0 = 0
    dt = env.dt
    tn = env.tn
    t = np.arange(t0, tn + dt, dt)

    # Control Define
    if CONST_STEP:
        stab_act = utils_control.step_function(t0, dt, tn, 5, np.radians(-10), bias=0)
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
        action = normalize_value(
            np.array([stab_act[i]]), -plane.maxabsstab, plane.maxabsstab
        )  # radians -> normalized action
        state, reward, done, _, info = env.step(
            action
        )  # plz put action as normalized value
        if done:
            states = states[:i]
            actions = actions[:i]
            times = times[:i]
            ref_signal = ref_signal[:i]
            break
        states.append(F16.denormalize(state))
        rewards.append(reward)
        actions.append(F16.rescale_action(action))
        times.append(info["clock"])
    return states, actions, rewards, times, ref_signal


if __name__ == "__main__":
    init_state = np.array([0, 3000.0, 0, 0, 195.0, 0])
    init_control = np.array([0, 0])
    ENV_CONFIG = {
        "dt": 0.01,
        "tn": 10,
        "debug_state": False,
        "determenistic_ref": True,
        "scenario": "pure_step",
    }
    ENV_CONFIG["init_state"] = init_state
    ENV_CONFIG["init_control"] = init_control

    _, u0 = get_trimmed_state_control()
    states, actions, r, t, ref = run_sim(u0)
    print(f"Total reward {sum(r)}")
    utils_plots.result(states, actions, t, ref_signal=ref, reward=r)
