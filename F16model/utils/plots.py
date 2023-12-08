import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def result(x_array, u_array, time, plot_name=None, ref_signal=None):
    plt.subplot(6, 1, 1)
    plt.plot(time, np.degrees([i for i in u_array]), "-r")
    plt.grid()
    plt.xlim(time[0], time[-1])
    plt.ylabel(r"$stab_{act}$, deg")

    #     plt.subplot(7, 1, 2)
    #     plt.plot(time, [i[1] for i in u_array], "-r")
    #     plt.grid()
    #     plt.xlim(time[0], time[-1])
    #     plt.ylabel(r"$P$")

    plt.subplot(6, 1, 2)
    plt.plot(time, np.degrees([i[1] for i in x_array]), "-b")
    plt.grid()
    plt.xlim(time[0], time[-1])
    plt.ylabel(r"$\omega_{z}$, deg/sec")

    plt.subplot(6, 1, 3)
    plt.plot(time, np.degrees([i[2] for i in x_array]), "-b", label=r"$\theta$")
    plt.plot(time, np.degrees([i[3] for i in x_array]), "--m", label=r"$\alpha$")
    if ref_signal is not None:
        plt.plot(time, np.degrees(ref_signal), ":")
    plt.legend()
    plt.grid()
    plt.xlim(time[0], time[-1])
    plt.ylabel(r"$\theta\, \alpha$, deg")

    #    plt.subplot(7, 1, 5)
    #    plt.plot(time, [i[3] for i in x_array], "-b")
    #    if ref_signal is not None:
    #        ref_speed = x_array[0][3]
    #        plt.plot(time, [ref_speed for i in time], ":")
    #    plt.ylabel("$V$, m/s")
    #    plt.grid()
    #    plt.xlim(time[0], time[-1])
    #    plt.xlabel("t, sec")

    plt.subplot(6, 1, 4)
    plt.plot(time, [i[0] for i in x_array], "-b")
    plt.ylabel("$H$, m")
    plt.grid()
    plt.xlim(time[0], time[-1])

    plt.subplot(6, 1, 5)
    plt.plot(time, [i[4] for i in x_array], ":", label=r"$\theta_{err}$")
    #    plt.plot(time, [i[6] for i in x_array], ":", label=r"$V_{err}$")
    plt.legend()
    plt.grid()
    plt.xlim(time[0], time[-1])

    plt.xlabel("t, sec")

    if plot_name:
        plot_name = plot_name + ".png"
    else:
        RUN_TIME = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        plot_name = RUN_TIME + ".png"
    plt.gcf().set_size_inches(8, 10)
    plt.tight_layout()
    plt.savefig(f"./logs/{plot_name}", dpi=300)
    plt.clf()


def algo(rewards, time, plot_name):
    plt.subplot(1, 1, 1)
    plt.plot(time, rewards)
    plot_name = plot_name + ".png"
    plt.gcf().set_size_inches(8, 10)
    plt.tight_layout()
    plt.savefig(f"./logs/{plot_name}", dpi=300)
    plt.clf()
