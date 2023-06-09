import numpy as np


def step_function(t0, dt, tn, tstep, ampl):
    t = np.arange(t0, tn + dt, dt)
    return (t >= tstep) * ampl


def make_step_series(t0, dt, tn, amplitude_value, step_time, hold_time):
    step_n = (tn - t0) // (step_time + hold_time)
    out_sig = np.arange(t0, tn + dt, dt) * 0
    for i in range(1, step_n + 1):
        step = step_function(
            t0, dt, tn, (hold_time * i) + step_time * (i - 1), amplitude_value
        )
        neg_step = step_function(
            t0, dt, tn, (hold_time * i) + step_time * i, -amplitude_value
        )
        out_sig += step + neg_step
    return out_sig
