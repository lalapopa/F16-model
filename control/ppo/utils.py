import os
import argparse
import numpy as np
import torch.nn as nn
from distutils.util import strtobool

import F16model.utils.control_metrics as utils_metrics
from F16model.env import F16


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0003,
        help="the learning rate of the optimizer",
    )
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=200000,
        help="total timesteps of the experiments",
    )
    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`",
    )
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, cuda will be enabled by default",
    )

    # Algorithm specific arguments
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="the number of parallel game environments",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=2048,
        help="the number of steps to run in each environment per policy rollout",
    )
    parser.add_argument(
        "--anneal-lr",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggle learning rate annealing for policy and value networks",
    )
    parser.add_argument(
        "--gae",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Use GAE for advantage computation",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="the discount factor gamma"
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="the lambda for the general advantage estimation",
    )
    parser.add_argument(
        "--num-minibatches", type=int, default=64, help="the number of mini-batches"
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=10,
        help="the K epochs to update the policy",
    )
    parser.add_argument(
        "--norm-adv",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles advantages normalization",
    )
    parser.add_argument(
        "--clip-coef",
        type=float,
        default=0.5,
        help="the surrogate clipping coefficient",
    )
    parser.add_argument(
        "--clip-vloss",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.",
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.0, help="coefficient of the entropy"
    )
    parser.add_argument(
        "--vf-coef", type=float, default=0.5, help="coefficient of the value function"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="the maximum norm for the gradient clipping",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="the target KL divergence threshold",
    )
    parser.add_argument(
        "--note", type=str, default=None, help="additional info about run"
    )
    parser.add_argument(
        "--save-dir", type=str, default="runs/", help="where to save running info"
    )
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


def weight_histograms(writer, step, model):
    # Iterate over all model layers
    for layer_number in range(len(model)):
        # Get layer
        layer = model[layer_number]
        # Compute weight histograms for appropriate layer
        if isinstance(layer, nn.Conv2d):
            weights = layer.weight
            _weight_histograms_conv2d(writer, step, weights, layer_number)
        elif isinstance(layer, nn.Linear):
            weights = layer.weight
            _weight_histograms_linear(writer, step, weights, layer_number)


def _weight_histograms_conv2d(writer, step, weights, layer_number):
    weights_shape = weights.shape
    num_kernels = weights_shape[0]
    for k in range(num_kernels):
        flattened_weights = weights[k].flatten()
        tag = f"layer_{layer_number}/kernel_{k}"
        writer.add_histogram(
            tag, flattened_weights, global_step=step, bins="tensorflow"
        )


def _weight_histograms_linear(writer, step, weights, layer_number):
    flattened_weights = weights.flatten()
    tag = f"layer_{layer_number}"
    writer.add_histogram(tag, flattened_weights, global_step=step, bins="tensorflow")


def state_logger(run_name, action=None, init_state=None):
    if isinstance(init_state, np.ndarray):
        with open("logs/" + run_name + ".txt", "w") as f:
            f.write(str(list(init_state)) + "\n")
    if isinstance(action, np.ndarray):
        with open("logs/" + run_name + ".txt", "a") as f:
            f.write(str(list(action)) + "\n")

def calculate_episode_nmae(obs_signal, done_envs, step):
    nMAE_avg = 0
    for idx_done_env in done_envs:
        obs_normalized_single= [_[idx_done_env] for _ in obs_signal]
        obs_single = list(map(F16.denormalize, obs_normalized_single))
        theta_obs = [i[1] for i in obs_single] 
        theta_ref_obs = [i[3] for i in obs_single] 
        nMAE_episode = utils_metrics.nMAE(theta_ref_obs[:step], theta_obs[:step])
        nMAE_avg += nMAE_episode / len(done_envs) 
        print(f"nMAE EP #{idx_done_env}: {nMAE_episode:.2f}")
    return nMAE_avg 


def write_to_tensorboard(writer, info, global_step):
    total_rewards = 0
    done_envs = []
    for item in info:
        if "final_info" in item:
            step_taken = 0
            log_steps = []
            log_rewards = []
            log_length = []
            for idx, final_item in enumerate(info["_final_info"]):
                if final_item:
                    ep_return = info["final_info"][idx]["total_return"]
                    ep_length = info["final_info"][idx]["episode_length"]
                    log_steps.append(global_step - step_taken)
                    log_rewards.append(ep_return)
                    log_length.append(ep_length)
                    step_taken += ep_length
                    done_envs.append(idx)
            for i_step, i_reward, i_length in zip(
                log_steps[::-1], log_rewards[::-1], log_length[::-1]
            ):
                # print(f"global_step={i_step}, episodic_return={i_reward}")
                total_rewards += i_reward
                writer.add_scalar(
                    "charts/episodic_return",
                    i_reward,
                    i_step,
                )
                writer.add_scalar(
                    "charts/episodic_length",
                    i_length,
                    i_step,
                )
            avg_returns = total_rewards / len(log_rewards)
            print(
                f"Step: {global_step} EPs: {len(log_rewards)} AVG episodes return: {avg_returns:.2f}"
            )
            return avg_returns, done_envs
    return None, done_envs


def write_python_file(filename, save_name):
    print(filename, save_name)
    with open(filename) as f:
        data = f.read()

    if not os.path.exists(os.path.dirname(save_name)):
        os.makedirs(os.path.dirname(save_name))

    with open(save_name, mode="w") as f:
        f.write(data)
