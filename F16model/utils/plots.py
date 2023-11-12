import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def result(x_array, u_array, time, plot_name=None):
    plt.subplot(5, 1, 1)
    plt.plot(time, np.degrees([i[0] for i in u_array]), "-r")
    plt.grid()
    plt.xlim(time[0], time[-1])
    plt.ylabel(r"$stab_{act}$, deg")

    plt.subplot(5, 1, 2)
    plt.plot(time, [i[1] for i in u_array], "-r")
    plt.grid()
    plt.xlim(time[0], time[-1])
    plt.ylabel(r"$P$")

    plt.subplot(5, 1, 3)
    plt.plot(time, np.degrees([i[2] for i in x_array]), "-b", label=r"$\theta$")
    plt.plot(time, np.degrees([i[4] for i in x_array]), "--m", label=r"$\alpha$")
    plt.legend()
    plt.grid()
    plt.xlim(time[0], time[-1])
    plt.ylabel(r"$\theta\, \alpha$, deg")

    plt.subplot(5, 1, 4)
    plt.plot(time, np.degrees([i[1] for i in x_array]), "-b")
    plt.grid()
    plt.xlim(time[0], time[-1])
    plt.ylabel(r"$\omega_{z}$, deg/sec")

    plt.subplot(5, 1, 5)
    plt.plot(time, [i[3] for i in x_array], "-b")
    plt.ylabel("$V$, m/s")
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
