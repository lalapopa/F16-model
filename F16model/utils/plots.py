import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def result(x_array, u_array, time, plot_name=None, ref_signal=None, cut_index=None):
    cut_index = _get_cut_index(cut_index)
    plt.subplot(5, 1, 1)
    plt.plot(time[cut_index:], np.degrees([i for i in u_array])[cut_index:], "-r")
    plt.grid()
    plt.ylabel(r"$\varphi$, deg")
    #    plt.subplot(6, 1, 2)
    #    plt.plot(time, [i[1] for i in u_array], "-r")
    #    plt.grid()
    #    plt.xlim(time[0], time[-1])
    #    plt.ylabel(r"$P$")

    plt.subplot(5, 1, 2)
    plt.plot(time[cut_index:], np.degrees([i[1] for i in x_array])[cut_index:], "-b")
    plt.grid()
    plt.ylabel(r"$\omega_{z}$, deg/sec")

    plt.subplot(5, 1, 3)
    plt.plot(
        time[cut_index:],
        np.degrees([i[2] for i in x_array])[cut_index:],
        "-b",
        label=r"$\vartheta$",
    )
    #     plt.plot(
    #         time[cut_index:],
    #         np.degrees([i[3] for i in x_array])[cut_index:],
    #         "--m",
    #         label=r"$\alpha$",
    #     )
    if ref_signal is not None:
        plt.plot(
            time[cut_index:],
            np.degrees(ref_signal)[cut_index:],
            ":",
            label=r"$\vartheta_{ref}$",
        )
    plt.legend()
    plt.grid()
    plt.ylabel(r"$\vartheta\, deg")

    #    plt.subplot(6, 1, 5)
    #    plt.plot(time, [i[3] for i in x_array], "-b")
    #    if ref_signal is not None:
    #        ref_speed = x_array[0][3]
    #        plt.plot(time, [ref_speed for i in time], ":")
    #    plt.ylabel("$V$, m/s")
    #    plt.grid()
    #    plt.xlim(time[0], time[-1])
    #    plt.xlabel("t, sec")
    plt.subplot(5, 1, 4)
    plt.plot(
        time[cut_index:],
        np.degrees([i[3] for i in x_array][cut_index:])
        - np.degrees([i[2] for i in x_array][cut_index:]),
        "-b",
        label=r"$\theta_{err}$",
    )

    plt.ylabel(r"Reward specific signals")
    plt.legend()
    plt.grid()

    plt.subplot(5, 1, 5)
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
