import numpy as np

from make_enviroment import Env, get_trimmed_state_control
import F16model.utils.plots as utils_plots
import F16model.utils.control as utils_control


CONST_STEP = True


def run_sim(x0, u0, max_episode=2000):
    env = Env(x0, u0)
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
        action = np.array([stab_act[i], throttle_act[i]])
        state, reward, done, current_time, _ = env.step(action)
        if state.all():
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            times.append(current_time)
        if done:
            break
    return states, actions, sum(rewards), times


if __name__ == "__main__":
    x0, u0 = get_trimmed_state_control()
    states, actions, r, t = run_sim(x0, u0)
    print(f"total reward {r}")
    utils_plots.result(states, actions, t, "test_name")
