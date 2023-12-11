import numpy as np
import random
import ast
import time

from F16model.model import States
from F16model.env import F16, get_trimmed_state_control
from F16model.data import plane
from F16model.utils.calc import normalize_value
import F16model.utils.plots as utils_plots

SEED = 322
ENV_CONFIG = {
    "dt": 0.01,
    "tn": 5,
    "norm_state": False,
    "debug_state": False,
    "determenistic_ref": True,
}


def test_F16():
    x0, u0 = get_trimmed_state_control()
    ENV_CONFIG["init_state"] = x0
    ENV_CONFIG["init_control"] = u0
    n = 2048
    env = F16(ENV_CONFIG)
    #    env.reset(SEED)
    actions = []
    states = []
    rewards = []
    times = []
    done = False
    start_time = time.time()

    for _ in range(0, n):
        # action = get_action()
        # u0 = np.array(
        #    [np.radians(random.uniform(-10, 10)), np.radians(random.uniform(0, 1))]
        # )
        stab_norm = normalize_value(u0[0], np.radians(-25), np.radians(25))
        action = np.array([stab_norm])
        state, reward, done, _, info = env.step(action)  # give as numpy array
        if state.all():
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            times.append(info["clock"])
        if done:
            break
    print(f"TOTAL REWARD = {round(sum(rewards), 4)}, TOTAL TIME = {times[-1]}")
    print("--- %s seconds ---" % (time.time() - start_time))
    denorm_states = list(map(F16.denormalize, states))
    actions = list(map(F16.rescale_action, actions))
    print(len(times), len(env.ref_signal.theta_ref))
    utils_plots.result(
        denorm_states,
        actions,
        times,
        plot_name="test_F16_gym",
        ref_signal=env.ref_signal.theta_ref,
    )
    utils_plots.algo(rewards, times, plot_name="test_F16_gym_algo")


def test_failed_run():
    file_name = "./logs/F16__ppo_train__1__1700557367_93b4.txt"
    data = []
    with open(file_name, "r") as f:
        for line in f:
            data.append(ast.literal_eval(line))
    init_state = States(*data[0])
    actions = data[1:]
    clipped_actions = []
    for action in actions:
        clipped_actions.append(
            np.clip(action, [-plane.maxabsstab, 0], [plane.maxabsstab, 1])
        )
    env = F16(init_state=init_state, debug_state=True)
    states = []
    rewards = []
    times = []
    done = False
    i = 0
    for action in clipped_actions:
        print(action)
        state, reward, done, current_time, _ = env.step(action)  # give as numpy array
        i += 1
        if state.all():
            states.append(state)
            rewards.append(reward)
            times.append(current_time)
        if done:
            break

    print(f"TOTAL REWARD = {round(sum(rewards), 4)}, TOTAL TIME = {times[-1]}")
    print(f"|{states[0]}|{len(rewards) = }|{done = }|")
    utils_plots.result(
        states[: i - 10],
        clipped_actions[: i - 10],
        times[: i - 10],
        plot_name=f"fail_test_{file_name[-8:-4]}",
    )
    utils_plots.algo(rewards, times, plot_name=f"fail_test_{file_name[-8:-4]}_reward")


if __name__ == "__main__":
    test_F16()
#    check_dispertion_reward()
#    parallel_env_runner()
#    with Profile() as profile:
#        for i in range(0, 1):
#            test_F16()
#        (Stats(profile).strip_dirs().sort_stats(SortKey.TIME).print_stats())
