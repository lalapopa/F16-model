import os
import gymnasium as gym

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


if __name__ == "__main__":
    args = parse_args()

#    write_python_file(
#        os.path.abspath(__file__), f"runs/{run_name}/{os.path.basename(__file__)}"
#    )
#    write_python_file(
#        os.path.abspath(__file__).replace("train", "model"),
#        f"runs/{run_name}/{os.path.basename(__file__).replace('train', 'model')}",
#    )
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.seed + i,
            )
            for i in range(args.num_envs)
        ]
    )
    model = Agent(envs, args)
    model.train()
