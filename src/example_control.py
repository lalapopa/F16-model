import numpy as np
import time

import utils.control, utils.plots
from model import States, Control, ODE_3DoF
from model.engine import find_correct_thrust_position


RUN_CSAPS = False
CONST_STEP = True


def run_sim(x0, u0):
    t0 = 0
    dt = 0.02
    tn = 5
    t = np.arange(t0, tn + dt, dt)

    # Control Define
    if CONST_STEP:
        stab_act = utils.control.step_function(t0, dt, tn, 0, u0.stab)
        throttle_act = utils.control.step_function(t0, dt, tn, 0, u0.throttle)
    else:
        stab_act = np.radians(
            utils.control.make_step_series(t0, dt, tn, 5, step_time=3, hold_time=1)
        )
        throttle_act = utils.control.step_function(t0, dt, tn, 0, 0)

    u = np.zeros(len(t), dtype=object)
    u = [Control(stab_act[i], throttle_act[i]) for i, _ in enumerate(u)]

    # Calculate states
    start = time.time()

    x_out = np.zeros(len(t), dtype=object)
    x_out[0] = x0
    for i in range(1, len(t)):
        if RUN_CSAPS:
            x_out[i] = x_out[i - 1] + dt * ODE_3DoF.solve(
                x_out[i - 1], u[i - 1], interp_method="csaps"
            )
        else:
            x_out[i] = x_out[i - 1] + dt * ODE_3DoF.solve(x_out[i - 1], u[i - 1])
        if np.isnan(x_out[i].Ox):
            x_out[i] = States(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            break_index = np.where(x_out == 0)[0][0]
            x_out = x_out[:break_index]
            u = u[:break_index]
            t = t[:break_index]
            break
    print(f"Simulation FNISED IN {time.time() - start }s")
    return x_out, u, t


if __name__ == "__main__":
    u_trimed = Control(np.radians(-4.41636648174007), 0.24075439651111258)
    Ox0 = 0
    Oy0 = 3000
    V0 = 125
    alpha0 = np.radians(3.1)
    wz0 = np.radians(0)
    theta0 = np.radians(3.1)
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
    x_result, u_result, t = run_sim(x0, u_trimed)
    utils.plots.result(x_result, u_result, t)
