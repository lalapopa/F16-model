import os
import time
import random
import gymnasium as gym
import optuna

from ppo_model_gsde import Agent
from ppo_train_gsde import make_env
from utils import (
    parse_args,
    write_python_file,
)


def objective(trial):
    args.learning_rate = trial.suggest_float("lr", 1e-6, 5e-3, log=True)
    args.num_minibatches = trial.suggest_int("num-minibatches", 32, 512, step=32)

    model = Agent(envs, args)
    run_name = f"F16__{args.seed}__{str(int(time.time()))}__{('%032x' % random.getrandbits(128))[:4]}"
    write_python_file(
        os.path.abspath(__file__),
        f"{args.save_dir}/{run_name}/{os.path.basename(__file__)}",
    )
    write_python_file(
        "control/ppo/ppo_model_gsde.py",
        f"{args.save_dir}/{run_name}/ppo_model_gsde.py",
    )  # stupid as shit
    accuracy = model.train(run_name)
    return accuracy


if __name__ == "__main__":

    ENV_CONFIG = {
        "dt": 0.01,
        "tn": 10,
        "norm_state": True,
        "debug_state": False,
        "determenistic_ref": False,
    }
    args = parse_args()

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.seed + i, ENV_CONFIG) for i in range(args.num_envs)]
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
