import os
import time
import random
import gymnasium as gym
import optuna

from ppo_model_gsde import Agent
from F16model.env import F16
from utils import (
    parse_args,
    write_python_file,
)

ENV_CONFIG = {
    "dt": 0.01,
    "tn": 10,
    "norm_state": True,
    "debug_state": False,
    "determenistic_ref": False,
}


def make_env(seed):
    def wrap_env():
        env = F16(ENV_CONFIG)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return wrap_env


def objective(trial):
    args.learning_rate = trial.suggest_float("lr", 1e-7, 1e-3, log=True)
    args.clip_coef = trial.suggest_float("clip-coef", 0.1, 2, step=0.1)
    args.max_grad_norm = trial.suggest_float("max-grad-norm", 0.1, 2, step=0.1)

    model = Agent(envs, args)
    run_name = f"F16__{args.exp_name}__{args.seed}__{str(int(time.time()))}_{('%032x' % random.getrandbits(128))[:4]}"
    accuracy = model.train(run_name)
    return accuracy


if __name__ == "__main__":
    args = parse_args()

    run_name = f"F16__{args.exp_name}__{args.seed}__{str(int(time.time()))}_{('%032x' % random.getrandbits(128))[:4]}"
    write_python_file(
        os.path.abspath(__file__), f"runs/{run_name}/{os.path.basename(__file__)}"
    )
    write_python_file(
        "control/ppo/ppo_model_gsde.py",
        f"runs/{run_name}/ppo_model_gsde.py",
    )  # stupid as shit
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.seed + i,
            )
            for i in range(args.num_envs)
        ]
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective, n_trials=100, catch=(ValueError)
    )  # When None need to continue
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
