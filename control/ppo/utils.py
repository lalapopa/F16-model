import os
import argparse
import numpy as np
import torch.nn as nn
from distutils.util import strtobool


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="LunarLander-v2",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=0.0001,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-play",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=512,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--note", type=str, default=None,
        help="additional info about run")
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


def write_to_tensorboard(writer, info, global_step, args):
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

            for i_step, i_reward, i_length in zip(
                log_steps[::-1], log_rewards[::-1], log_length[::-1]
            ):
                print(f"global_step={i_step}, episodic_return={i_reward}")
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
                if args.track:
                    wandb.log({"charts/episodic_return": i_reward})
                    wandb.log({"charts/episodic_length": i_length})
            break


def write_python_file(filename, save_name):
    with open(filename) as f:
        data = f.read()
        f.close()

    if not os.path.exists(save_name):
        os.makedirs(os.path.dirname(save_name))

    with open(save_name, mode="w") as f:
        f.write(data)
        f.close()
