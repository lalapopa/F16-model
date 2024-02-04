import os
import time
import random
import gymnasium as gym

from ppo_model_gsde import Agent
from F16model.env import F16
from utils import (
    parse_args,
    write_python_file,
)


def make_env(seed, env_config):
    def wrap_env():
        env = F16(env_config)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return wrap_env


if __name__ == "__main__":
    ENV_CONFIG = {
        "dt": 0.01,
        "tn": 10,
        "norm_state": True,
        "debug_state": False,
        "determenistic_ref": False,
    }
    args = parse_args()

    run_name = f"F16__{args.seed}__{str(int(time.time()))}__{('%032x' % random.getrandbits(128))[:4]}"
    write_python_file(
        os.path.abspath(__file__),
        f"{args.save_dir}/{run_name}/{os.path.basename(__file__)}",
    )
    write_python_file(
        os.path.abspath(__file__).replace("train", "model"),
        f"{args.save_dir}/{run_name}/{os.path.basename(__file__).replace('train', 'model')}",
    )  # stupid as shit
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.seed + i, ENV_CONFIG) for i in range(args.num_envs)]
    )

    start_time = time.time()
    model = Agent(envs, args)
    accuracy = model.train(run_name)
    print("--- %s seconds ---" % (time.time() - start_time))
