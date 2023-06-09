import numpy as np
import time

import F16model.utils.control, F16model.utils.plots
from F16model.model import States, Control, runner
from F16model.model.engine import find_correct_thrust_position


CONST_STEP = False
def run_sim(x0, u0):
    t0 = 0
    dt = 0.02
    tn = 20
    t = np.arange(t0, tn + dt, dt)

    # Control Define
    if CONST_STEP:
        stab_act = utils.control.step_function(t0, dt, tn, 0, u0.stab)
        throttle_act = utils.control.step_function(t0, dt, tn, 0, u0.throttle)
    else:
        stab_act = np.radians(
            utils.control.make_step_series(t0, dt, tn, 1, step_time=3, hold_time=2)
        )
        throttle_act = utils.control.step_function(t0, dt, tn, 0, u0.throttle)

    u = [Control(stab_act[i], throttle_act[i]) for i in range(len(t))]

    # Calculate states
    start = time.time()
    env = runner.Env(x0, dt)

    for i in range(len(t)):
        state = env.step(u[i])
        if not state:
            print(i)
            u = u[:i]
            t = t[:i]
            break
    print(f"Simulation FNISED IN {time.time() - start }s")
    print(len(env.get_states()), len(u), len(t))
    return env.get_states(), u, t


if __name__ == "__main__":
    u_trimed = Control(np.radians(0), 0.7570899485191026)
    Ox0 = 0
    Oy0 = 9000
    V0 = 325
    alpha0 = 0.0366848418846759
    wz0 = np.radians(0)
    theta0 = 0.0366848418846759
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
