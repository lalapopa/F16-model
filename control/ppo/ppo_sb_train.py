import os
import random
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from utils import parse_args, state_logger, weight_histograms, write_python_file
from F16model.env.env_gym import GymF16

ENV_CONFIG = {
    "dt": 0.01,
    "tn": 10,
    "norm_state": True,
    "debug_state": False,
    "determenistic_ref": False,
}


def env_wrapper():
    env = GymF16(ENV_CONFIG)
    return env


args = parse_args()
run_name = f"F16__{args.exp_name}__sb__{args.seed}__{str(int(time.time()))}_{('%032x' % random.getrandbits(128))[:4]}"


write_python_file(
    os.path.abspath(__file__), f"runs/{run_name}/{os.path.basename(__file__)}"
)

vec_env = make_vec_env(env_wrapper, n_envs=4)
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log=f"runs/{run_name}",
    seed=4,
    use_sde=True,
)
model.learn(total_timesteps=5000000)
model.save(f"runs/model/{run_name}")

# del model  # remove to demonstrate saving and loading

# model = PPO.load(f"model/{run_name}")

# obs = vec_env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = vec_env.step(action)
#     print(reward)
#     vec_env.render("human")
