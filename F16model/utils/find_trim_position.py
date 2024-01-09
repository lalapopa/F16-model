import scipy.optimize
import numpy as np
import logging
import datetime
import os

from F16model.model import States, Control, ODE_3DoF
from F16model.model.engine import find_correct_thrust_position


class Cost:
    def __init__(self, V, H):
        self.V = V
        self.H = H
        self.weight = States(
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

    def calculate(self, u):
        u_control = Control(u[0], u[1])  # TODO: clean up work with custom class
        x0 = States(
            Ox=0,
            Oy=self.H,
            wz=0,
            V=self.V,
            alpha=u[2],
            theta=u[2],
            stab=u_control.stab,
            dstab=0,
            Pa=find_correct_thrust_position(u_control.throttle),
        )
        step = ODE_3DoF.solve(x0, u_control)
        state_in_power = np.power(step.to_array(), 2)
        return np.matmul(self.weight.to_array(), state_in_power)


def optimize_loop(u, init_condition):
    run_trim = True
    last_cost = 0
    max_iter = 5
    current_iter = 1
    cost = Cost(*init_condition)
    while run_trim:
        out, eval_cost, _, _, _ = scipy.optimize.fmin(
            func=cost.calculate,
            x0=u,
            xtol=1e-10,
            ftol=1e-10,
            maxfun=5e03,
            maxiter=1e03,
            disp=False,
            full_output=True,
        )
        if eval_cost == last_cost or current_iter > max_iter:
            run_trim = False
        current_iter += 1
        last_cost = eval_cost
    return last_cost, Control(out[0], out[1]), out[2], out[2]


def trim_find(array, init_condition):
    eps = 10e-6
    all_costs = []
    all_u_trimed = []
    all_alphas = []
    all_thetas = []
    for i in array:
        cost, u_trim, alpha, theta = optimize_loop(i, init_condition)
        all_costs.append(cost)
        all_u_trimed.append(u_trim)
        all_alphas.append(alpha)
        all_thetas.append(theta)
        logging.info(f"combination = {i}; cost = {cost} / {eps}; trim = {u_trim}")
        if cost <= eps:
            break
    return (
        all_costs,
        all_u_trimed,
        all_alphas,
        all_thetas,
    )


def run(V0, Oy0):
    stab_range = np.radians(np.arange(-20, 20 + 0.1, 1))
    thrust_range = np.arange(0.2, 1 + 0.5, 0.25)
    alpha_range = np.radians(np.arange(-10, 10 + 0.1, 2))
    combinations = np.array(
        np.meshgrid(stab_range, thrust_range, alpha_range)
    ).T.reshape(-1, 3)
    np.random.shuffle(combinations)
    logging.info(f"Total combinations = {len(combinations)}")
    init_condition = (V0, Oy0)
    (
        all_costs,
        all_u_trimed,
        all_alphas,
        all_thetas,
    ) = trim_find(combinations, init_condition)
    min_cost_index = np.argmin(all_costs)
    return (
        all_u_trimed[min_cost_index],
        all_alphas[min_cost_index],
        all_thetas[min_cost_index],
    )


if __name__ == "__main__":
    current_time = datetime.datetime.now().strftime("%y%m%d")
    log_file_name = str(f"./logs/{current_time}.log")
    os.makedirs(os.path.dirname(log_file_name), exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        filename=log_file_name,
        filemode="a+",
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y%m%d%H%M%S",
    )
    # Initial condions
    H = 8000
    V = 80
    print(run(V, H))
