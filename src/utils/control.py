import numpy as np


def step_function(t0, dt, tn, tstep, ampl):
    t = np.arange(t0, tn+dt, dt)
    return (t >= tstep) * ampl
