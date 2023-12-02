import numpy as np
import random
import torch
import gym

from control.ppo.ppo_model import Agent


def make_env(gym_id, seed, idx, capture_video, run_name):
    env = gym.make(gym_id, continuous=True, render_mode="human")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if capture_video:
        if idx == 0:
            env = gym.wrappers.RecordVideo(env, f"logs/{run_name}")
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = "LunarLander_test__utils__1__1701346044_3e3b"
    gym_id = "LunarLander-v2"
    total_steps = 500
    seed = 1

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(gym_id, seed, 0, True, f"{run_name}")
    agent = Agent(
        np.array(env.observation_space.shape).prod(),
        np.array(env.action_space.shape).prod(),
    ).to(device)
    agent.load(f"runs/models/{run_name}")

    total_reward = 0
    done = False
    total_fucks = []
    eps_num = 100
    for i in range(0, eps_num):
        total_reward = 0
        stop_action = False
        next_obs, _ = env.reset()
        done = False
        for step in range(0, total_steps + 1):
            next_obs = torch.Tensor(next_obs).to(device).reshape(1, 8)
            if not stop_action:
                action, _, _, _ = agent.get_action_and_value(next_obs)
                action = action.cpu().numpy()[0]
            else:
                action = np.array([0, 0])
            next_obs, reward, done, _, info = env.step(action)
            print(f"STATE: {next_obs}\nREWARD: {reward}\nACTION: {action}")
            #           env.render()
            total_reward += reward
            if next_obs[-1] == 1 and next_obs[-2] == 1:
                stop_action = True
            if done:
                break
        if total_reward < 200:
            total_fucks.append(i)
            print(f"#{i+1} {total_reward = }, {step =} :(")
        else:
            print(f"#{i+1} {total_reward = }, {step =} :)")
    print(f"{len(total_fucks)}/{eps_num} DEAD EP: {total_fucks}")
    env.close()
