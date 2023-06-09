import scipy.optimize
import numpy as np
import logging
import datetime
import matplotlib.pyplot as plt

from model import States, Control, ODE_3DoF
from model.engine import find_correct_thrust_position
import utils.plots
from example_control import run_sim


def cost_function(u):
    weight = States(
        Ox=0,
        Oy=10,
        wz=10,
        theta=15,
        V=30,
        alpha=20,
        stab=0,
        dstab=0,
        Pa=0,
    )
    u_control = Control(u[0], u[1])

    x0 = States(
        Ox=Ox0,
        Oy=Oy0,
        wz=wz0,
        V=V0,
        alpha=u[2],
        theta=u[2],
        stab=u_control.stab,
        dstab=dstab0,
        Pa=find_correct_thrust_position(u_control.throttle),
    )
    step = ODE_3DoF.solve(x0, u_control)
    state_in_power = np.power(step.to_array(), 2)
    return np.matmul(weight.to_array(), state_in_power)


def get_level_trim_value(u):
    run_trim = True
    last_cost = 0
    max_iter = 5
    current_iter = 1
    while run_trim:
        out = scipy.optimize.fmin(
            func=cost_function,
            x0=u,
            xtol=1e-10,
            ftol=1e-10,
            maxfun=5e03,
            maxiter=1e03,
        )
        cost = cost_function(out)

        if cost == last_cost or current_iter > max_iter:
            run_trim = False

        current_iter += 1
        last_cost = cost
    print(f"RESULT FINDED = {out}")
    return u, last_cost, Control(out[0], out[1]), out[2], out[2]


def get_error(first_state, second_state):
    theta_err = np.degrees(abs(first_state.theta - second_state.theta))
    wz_err = np.degrees(abs(first_state.wz - second_state.wz))
    return theta_err, wz_err


def make_combination(arr1, arr2, arr3):
    result = []
    for i in arr2:
        for j in arr1:
            for m in arr3:
                result.append(np.array([j, i, m]))
    return np.array(result)


def trim_find(array):
    eps = 10e-6
    all_costs = []
    all_u_trimed = []
    combinations = []
    all_alphas = []
    all_thetas = []
    for i in array:
        u0, cost, u_trim, alpha, theta = get_level_trim_value(i)
        all_costs.append(cost)
        combinations.append(u0)
        all_u_trimed.append(u_trim)
        all_alphas.append(alpha)
        all_thetas.append(theta)
        logging.info(f"combination = {u0}; cost = {cost}; trim = {u_trim}")
        if cost <= eps:
            return (
                combinations,
                all_costs,
                all_u_trimed,
                all_alphas,
                all_thetas,
            )
    return (
        combinations,
        all_costs,
        all_u_trimed,
        all_alphas,
        all_thetas,
    )


if __name__ == "__main__":
    current_time = datetime.datetime.now().strftime("%y%m%d")
    logging.basicConfig(
        level=logging.DEBUG,
        filename=f"logs/{current_time}.log",
        filemode="a+",
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y%m%d%H%M%S",
    )
    # Initial condions
    Ox0 = 0
    Oy0 = 9000
    V0 = 325
    wz0 = np.radians(0)
    dstab0 = np.radians(0)

    stab_range = np.radians(np.arange(-20, 20 + 0.1, 1))
    thrust_range = np.arange(0.2, 1 + 0.5, 0.25)
    alpha_range = np.radians(np.arange(-10, 10 + 0.1, 2))

    combinations = make_combination(stab_range, thrust_range, alpha_range)
    logging.info(f"Total combinations = {len(combinations)}")
    (
        combinations,
        all_costs,
        all_u_trimed,
        all_alphas,
        all_thetas,
    ) = trim_find(combinations)
    min_cost_index = np.argmin(all_costs)
    print(f"min_cost = {all_costs[min_cost_index]}")
    print(f"best init value = {combinations[min_cost_index]}")
    print(f"trimmed result = {all_u_trimed[min_cost_index]}")
    print(f"alpha, theta = {all_alphas[min_cost_index]}, {all_thetas[min_cost_index]}")
