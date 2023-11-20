import numpy as np
import random
import ast

from F16model.model import F16, States
from F16model.data import plane
from F16model.model.env import run_episode, get_trimmed_state_control
import F16model.utils.plots as utils_plots

approx_reward = 2480.2364
random.seed(322)


def test_run_episode():
    x0, u0 = get_trimmed_state_control()
    states, actions, reward, t = run_episode(x0, u0)
    utils_plots.result(states, actions, t, plot_name="test_run_episode")
    print(f"TOTAL REWARD = {reward}/{approx_reward}, TOTAL TIME = {t[-1]}")


def test_F16():
    x0, u0 = get_trimmed_state_control()
    env = F16(x0, u0, norm_state=True)
    actions = []
    states = []
    rewards = []
    times = []
    done = False
    while not done:
        # action = get_action()
        action = np.array([np.radians(random.uniform(-10, 10)), 0.3])
        state, reward, done, current_time, _ = env.step(action)  # give as numpy array
        if state.all():
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            times.append(current_time)
        if done:
            break
    print(
        f"TOTAL REWARD = {round(sum(rewards), 4)}/{approx_reward}, TOTAL TIME = {times[-1]}"
    )
    print(f"|{states[0]}|{len(rewards) = }|{done = }|")
    denorm_states = list(map(F16.denormalize, states))
    utils_plots.result(denorm_states, actions, times, plot_name="test_F16")


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
    env = F16(x0, u0, norm_state=True)
    actions = []
    states = []
    rewards = []
    times = []
    done = False
    while not done:
        # action = np.random.rand(2)
        # action = np.array(u0)
        action = np.array([0, 1])
        state, reward, done, current_time, _ = env.step(action)  # give as numpy array
        if state.all():
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            times.append(current_time)
        if done:
            break
    print(
        f"TOTAL REWARD = {round(sum(rewards), 4)}/{approx_reward}, TOTAL TIME = {times[-1]}"
    )
    print(f"|{states[0]}|{len(rewards) = }|{done = }|")
    denorm_states = list(map(F16.denormalize, states))
    utils_plots.result(denorm_states, actions, times, plot_name="test_F16_trim_value")
    utils_plots.algo(rewards, times, plot_name="test_F16_trim_value_algo")


def test_failed_run():
    file_name = "./logs/F16__ppo_train__1__1699893040_2f26.txt"
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
        state, reward, done, current_time, _ = env.step(action)  # give as numpy array
        i += 1
        if state.all():
            states.append(state)
            rewards.append(reward)
            times.append(current_time)
        if done:
            break

    print(
        f"TOTAL REWARD = {round(sum(rewards), 4)}/{approx_reward}, TOTAL TIME = {times[-1]}"
    )
    print(f"|{states[0]}|{len(rewards) = }|{done = }|")
    utils_plots.result(
        states[: i - 10],
        clipped_actions[: i - 10],
        times[: i - 10],
        plot_name="test_F16_trim_value",
    )
    utils_plots.algo(rewards, times, plot_name="test_F16_trim_value_algo")


if __name__ == "__main__":
    # test_run_episode()
    test_F16()
    # test_trim_state_value()
    # test_failed_run()
