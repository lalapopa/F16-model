import scipy.optimize
import numpy as np
import logging
import datetime
import multiprocessing
import concurrent.futures
import matplotlib.pyplot as plt


from model import States, Control, ODE_3DoF
from model.engine import find_correct_thrust_position
import utils.plots
from example_control import run_sim


def cost_function(u):
    weight = States(
        Ox=0,
        Oy=5,
        Vx=10,
        Vy=10,
        wz=10 * 57.3,
        V=10,
        alpha=10 * 57.3,
        theta=10 * 57.3,
        stab=0,
        dstab=0,
        Pa=0,
    )
    # Thrust lever limit
    if u[1] >= 1:
        u[1] = 1
    elif u[1] <= 0:
        u[1] = 0
    u_control = Control(u[0], u[1])
    t0 = 0
    dt = 0.02
    tn = 1
    t = np.arange(t0, tn + dt, dt)
    x0 = States(
        Ox=Ox0,
        Oy=Oy0,
        Vx=Vx0,
        Vy=Vy0,
        wz=wz0,
        V=V0,
        alpha=alpha0,
        theta=theta0,
        stab=u_control.stab,
        dstab=dstab0,
        Pa=find_correct_thrust_position(u_control.throttle),
    )
    x = np.zeros(len(t), dtype=object)
    x[0] = x0
    weights = np.zeros(len(t), dtype=object)
    for i in range(1, len(t)):
        step = ODE_3DoF.solve(x0, u_control)
        if np.isnan(step.wz):
            return np.inf
        x[i] = x[i - 1] + dt * step
    #        weights[i] = np.matmul(weight.to_array(), np.power(step.to_array(), 2))
    error = np.power(x[-1].to_array() - x0.to_array(), 2)
    #    error = np.power(step.to_array() - x0.to_array(), 2)
    return np.matmul(weight.to_array(), error)


def get_level_trim_value(u):
    print(u)
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
            maxfun=5e04,
            maxiter=1e04,
        )
        cost = cost_function(out)
        if cost == last_cost or current_iter > max_iter:
            run_trim = False
        print(f"iter = {current_iter}")
        current_iter += 1
        last_cost = cost
    return u, last_cost, Control(out[0], out[1])


def get_error(first_state, second_state):
    theta_err = np.degrees(abs(first_state.theta - second_state.theta))
    wz_err = np.degrees(abs(first_state.wz - second_state.wz))
    return theta_err, wz_err


def make_combination(arr1, arr2):
    result = []
    for i in arr2:
        for j in arr1:
            result.append(np.array([j, i]))
    return np.array(result)


def divide_into_chunks(array):
    cpu_number = multiprocessing.cpu_count()
    divide = int(len(array) / cpu_number)
    chunks = []
    if len(array) > cpu_number:
        for val in range(0, cpu_number):
            right_limit = divide * (val + 1)
            left_limit = val * divide
            if val == cpu_number - 1:
                right_limit = len(array)
            chunks.append(array[left_limit:right_limit])
    else:
        chunks = array
    return chunks


def paralell_trim_find(array):
    all_costs = []
    all_u_trimed = []
    combinations = []
    with concurrent.futures.ProcessPoolExecutor() as exe:
        result = exe.map(get_level_trim_value, array)
        for u0, cost, u_trim in result:
            all_costs.append(cost)
            combinations.append(u0)
            all_u_trimed.append(u_trim)
            logging.info(f"combination = {u0}; cost = {cost}; trim = {u_trim}")
    return combinations, all_costs, all_u_trimed


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
    Oy0 = 3000
    Vx0 = 125
    Vy0 = 0
    V0 = 125
    alpha0 = 0
    wz0 = np.radians(0)
    theta0 = np.radians(0)
    dstab0 = np.radians(0)

    stab_range = np.radians(np.arange(-5, 5 + 0.5, 0.5))
    thrust_range = np.arange(0.2, 1 + 0.5, 0.5)
    # thrust_val = 0.25
    # costs = []
    # for i in stab_range:
    #     cost = cost_function([i, thrust_val])
    #     costs.append(float(cost))
    #     print(cost)
    # print(stab_range)
    # print(costs)
    # plt.plot(np.degrees(stab_range), np.array(costs))
    # plt.xlabel("stab, deg")
    # plt.ylabel("cost")
    # plt.show()

    combinations = make_combination(stab_range, thrust_range)
    logging.info(f"Total combinations = {len(combinations)}")
    combinations, all_costs, all_u_trimed = paralell_trim_find(combinations)
    min_cost_index = np.argmin(all_costs)
    print(f"min_cost = {all_costs[min_cost_index]}")
    print(f"best init value = {combinations[min_cost_index]}")
    print(f"trimmed result = {all_u_trimed[min_cost_index]}")

#    u_trimed = Control(-0.08469571, 0.6016327)
#    x0 = States(
#        Ox0,
#        Oy0,
#        Vx0,
#        Vy0,
#        wz0,
#        theta0,
#        u_trimed.stab,
#        dstab0,
#        find_correct_thrust_position(u_trimed.throttle),
#    )
#    x_result, u_result, t = run_sim(x0, u_trimed)
#    final_value = x_result[-1]
