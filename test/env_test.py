from F16model.model import F16
from F16model.model.env import run_episode, get_trimmed_state_control
import F16model.utils.plots as utils_plots


def main():
    x0, u0 = get_trimmed_state_control()
    states, actions, reward, t = run_episode(x0, u0)
    print(f"TOTAL REWARD = {reward}, TOTAL TIME = {t[-1]}")
    utils_plots.result(states, actions, t)


if __name__ == "__main__":
    main()
