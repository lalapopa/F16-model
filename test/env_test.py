import numpy as np

from F16model.model import F16
from F16model.model.env import run_episode, get_trimmed_state_control
import F16model.utils.plots as utils_plots

approx_reward = 2480.2364


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
        action = np.array([0, 0])
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
        # action = get_action()
        action = np.random.rand(2)
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


if __name__ == "__main__":
    # test_run_episode()
    # test_F16()
    test_trim_state_value()
