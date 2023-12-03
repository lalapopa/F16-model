import numpy as np
import random
import ast
import time
from cProfile import Profile
from pstats import SortKey, Stats

from F16model.model import States
from F16model.env import F16, get_trimmed_state_control
from F16model.data import plane
import F16model.utils.plots as utils_plots

# random.seed(322)

ENV_CONFIG = {
    "dt": 0.01,
    "tn": 20,
    "norm_state": False,
    "debug_state": False,
}


def test_F16():
    ENV_CONFIG["norm_state"] = True
    x0, u0 = get_trimmed_state_control()
    ENV_CONFIG["init_state"] = x0
    ENV_CONFIG["init_control"] = u0
    n = 1000 
    env = F16(ENV_CONFIG)
#    env.reset()
    actions = []
    states = []
    rewards = []
    times = []
    done = False
    start_time = time.time()

    for _ in range(0, n):
        # action = get_action()
        action = np.array([np.radians(random.uniform(-10, 10)), 0.3])
        # action = u0
        state, reward, done, current_time, _ = env.step(action)  # give as numpy array
        if state.all():
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            times.append(current_time)
        if done:
            break
    print(f"TOTAL REWARD = {round(sum(rewards), 4)}, TOTAL TIME = {times[-1]}")
#    print(f"|{states[0]}|{len(rewards) = }|{done = }|")
    print("--- %s seconds ---" % (time.time() - start_time))
    denorm_states = list(map(F16.denormalize, states))
    utils_plots.result(denorm_states, actions, times, plot_name="test_F16", ref_signal=env.ref_signal.theta_ref[:n])
    utils_plots.algo(rewards, times, plot_name="test_F16_algo")


def check_dispertion_reward():
    ep_rewards = []
    for i in range(100):
        env = F16(ENV_CONFIG)
        actions = []
        states = []
        rewards = []
        times = []
        done = False
        for _ in range(0, 512):
            # action = np.array([0, 0])
            action = np.array([np.radians(random.uniform(-10, 10)), 0.3])
            state, reward, done, current_time, _ = env.step(
                action
            )  # give as numpy array
            if state.all():
                states.append(state)
                rewards.append(reward)
                actions.append(action)
                times.append(current_time)
            if done:
                break
        ep_rewards.append(sum(rewards))
        print(min(ep_rewards), max(ep_rewards), np.mean(ep_rewards))


def test_trim_state_value():
    u0 = [np.radians(-4.3166), 0.5449]
    x0 = [
        0,
        6000,
        0,
        np.radians(2.4839),
        250,
        np.radians(2.4839),
    ]
    #    env = F16(x0, u0, norm_state=True)
    env = F16(norm_state=True)
    actions = []
    states = []
    rewards = []
    times = []
    done = False
    while not done:
        action = np.random.rand(2)
        #        action = np.array(u0)
        # action = np.array([0, 1])
        state, reward, done, current_time, _ = env.step(action)  # give as numpy array
        if state.all():
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            times.append(current_time)
        if done:
            break
    print(f"TOTAL REWARD = {round(sum(rewards), 4)}, TOTAL TIME = {times[-1]}")
    print(f"|{states[0]}|{len(rewards) = }|{done = }|")
    denorm_states = list(map(F16.denormalize, states))
    utils_plots.result(denorm_states, actions, times, plot_name="test_F16_trim_value")
    utils_plots.algo(rewards, times, plot_name="test_F16_trim_value_algo")


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

    print(
        f"TOTAL REWARD = {round(sum(rewards), 4)}, TOTAL TIME = {times[-1]}"
    )
    print(f"|{states[0]}|{len(rewards) = }|{done = }|")
    utils_plots.result(
        states[: i - 10],
        clipped_actions[: i - 10],
        times[: i - 10],
        plot_name=f"fail_test_{file_name[-8:-4]}",
    )
    utils_plots.algo(rewards, times, plot_name=f"fail_test_{file_name[-8:-4]}_reward")


if __name__ == "__main__":
    # test_run_episode()
    # test_trim_state_value()
    # test_failed_run()
    test_F16()
    # check_dispertion_reward()
#    parallel_env_runner()
#    with Profile() as profile:
#        for i in range(0, 1):
#            test_F16()
#        (Stats(profile).strip_dirs().sort_stats(SortKey.TIME).print_stats())
