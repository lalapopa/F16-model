import string
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def result(
    x_array,
    u_array,
    time,
    plot_name=None,
    ref_signal=None,
    reward=None,
    cut_index=None,
    control="theta",
):
    cut_index = _get_cut_index(cut_index)
    if control == "theta":
        max_plots = 6
        error_plot_pos = 4
        error_label = "$\vartheta_{err}$"
    elif control == "omega":
        max_plots = 5
        error_plot_pos = 3
        error_label = "$\omega_{z \, {err}}$"

    plt.subplot(max_plots, 1, 1)
    plt.plot(time[cut_index:], np.degrees([i for i in u_array])[cut_index:], "-r")
    plt.grid()
    plt.ylabel(r"$\varphi$, deg")

    plt.subplot(max_plots, 1, 2)
    plt.plot(time[cut_index:], np.degrees([i[1] for i in x_array])[cut_index:], "-b")
    plt.grid()
    plt.ylabel(r"$\omega_{z}$, deg/sec")
    if control == "omega":
        if ref_signal is not None:
            plt.plot(
                time[cut_index:],
                np.degrees(ref_signal)[cut_index:],
                ":",
                label=r"$\omega_{ref}$",
            )

    if control == "theta":
        plt.subplot(max_plots, 1, 3)
        plt.plot(
            time[cut_index:],
            np.degrees([i[2] for i in x_array])[cut_index:],
            "-b",
            label=r"$\vartheta$",
        )
        if ref_signal is not None:
            plt.plot(
                time[cut_index:],
                np.degrees(ref_signal)[cut_index:],
                ":",
                label=r"$\vartheta_{ref}$",
            )
        plt.legend()
        plt.grid()
        plt.ylabel(r"$\vartheta \, deg$")

    plt.subplot(max_plots, 1, error_plot_pos)
    plt.plot(
        time[cut_index:],
        np.degrees([i[1] for i in x_array][cut_index:])
        - np.degrees(ref_signal)[cut_index:],
        "-b",
        label=r"%s" %error_label,
    )
    plt.ylabel(r"Error")
    plt.legend()
    plt.grid()

    plt.subplot(max_plots, 1, error_plot_pos + 1)
    plt.plot(time[cut_index:], reward[cut_index:], "-b")
    plt.ylabel(r"Reward")
    plt.grid()

    plt.subplot(max_plots, 1, error_plot_pos + 2)
    plt.plot(time[cut_index:], [i[0] for i in x_array][cut_index:], "-b")
    plt.ylabel("$H$, m")
    plt.grid()
    plt.xlabel("t, sec")

    if plot_name:
        plot_name = plot_name + ".svg"
    else:
        RUN_TIME = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        plot_name = RUN_TIME + ".png"
    plt.gcf().set_size_inches(8, 10)
    plt.tight_layout()
    plt.savefig(f"./logs/{plot_name}", dpi=300)
    print(f"Plots saved in: ./logs/{plot_name}")
    plt.clf()


def algo(rewards, time, plot_name=None):
    plt.subplot(1, 1, 1)
    plt.plot(time, rewards)
    plt.ylabel(r"Reward")
    plt.xlabel(r"time, sec")
    plt.grid()
    if plot_name:
        plot_name = plot_name + ".svg"
    else:
        RUN_TIME = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        plot_name = RUN_TIME + "_algo.png"
    plt.gcf().set_size_inches(8, 5)
    plt.tight_layout()
    plt.savefig(f"./logs/{plot_name}", dpi=300)
    plt.clf()


def _get_cut_index(cut_index):
    if cut_index:
        return cut_index
    else:
        return 0
