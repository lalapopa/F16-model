import numpy as np
import time

import F16model.utils.control as utils_control
import F16model.utils.plots as utils_plots
from F16model.model import States, Control, interface
from F16model.model.engine import find_correct_thrust_position


CONST_STEP = True


def run_sim(x0, u0):
    t0 = 0
    dt = 0.02
    tn = 10
    t = np.arange(t0, tn + dt, dt)

    # Control Define
    if CONST_STEP:
        stab_act = utils_control.step_function(t0, dt, tn, 1, u0.stab)
        throttle_act = utils_control.step_function(t0, dt, tn, 1, u0.throttle)
    else:
        stab_act = np.radians(
            utils_control.make_step_series(
                t0, dt, tn, 5, step_time=3, hold_time=2, bias=u0.stab
            )
        )
        throttle_act = utils_control.step_function(
            t0, dt, tn, 0, u0.throttle, bias=u0.throttle
        )

    u = [Control(stab_act[i], throttle_act[i]) for i in range(len(t))]

    # Calculate states
    start = time.time()
    states = []
    model = interface.F16model(x0, dt)

    for i in range(len(t)):
        x_i = model.step(u[i])
        states.append(x_i)
        if not x_i:
            u = u[:i]
            t = t[:i]
            break
    print(f"Simulation FNISED IN {time.time() - start }s")
    print(len(states), len(u), len(t))
    return states, u, t


if __name__ == "__main__":
    u_trimed = Control(np.radians(-4.3674), 0.3767)
    Ox0 = 0
    Oy0 = 3000
    V0 = 200
    alpha0 = np.radians(2.7970)
    wz0 = np.radians(0)
    theta0 = alpha0
    dstab0 = np.radians(0)
    x0 = States(
        Ox0,
        Oy0,
        wz0,
        theta0,
        V0,
        alpha0,
        u_trimed.stab,
        dstab0,
        find_correct_thrust_position(u_trimed.throttle),
    )
    x, u, t = run_sim(x0, u_trimed)
    utils_plots.result(x, u, t)
