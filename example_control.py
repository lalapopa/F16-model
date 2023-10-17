import numpy as np

from make_enviroment import Env, get_trimmed_state_control
import F16model.utils.plots as utils_plots
import F16model.utils.control as utils_control


CONST_STEP = False 

def run_sim(x0, u0, max_episode=2000):
    env = Env(x0, u0)
    t0 = 0
    dt = env.dt 
    tn = env.tn
    t = np.arange(t0, tn + dt, dt)

    # Control Define
    if CONST_STEP:
        stab_act = utils_control.step_function(t0, dt, tn, 1, u0[0])
        throttle_act = utils_control.step_function(t0, dt, tn, 1, u0[1])
    else:
        stab_act = utils_control.make_step_series(t0, dt, tn, np.radians(4), step_time=2, hold_time=5, bias=u0[0])
        throttle_act = utils_control.step_function(t0, dt, tn, 5, 1, bias=u0[1])

    u = [np.array([stab_act[i], throttle_act[i]]) for i in range(len(t))]

    actions = []
    states = []
    rewards = []
    times = []
    for action in u:
        action = np.clip(action, [-np.radians(25), 0], [np.radians(25), 1])
        state, reward, done, current_time = env.step(action)  
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