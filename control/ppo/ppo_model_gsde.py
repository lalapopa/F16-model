import os
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Agents for PPO algo"""

    def __init__(self, obs_shape, action_shape):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_shape), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_shape))
        self.gsde_logstd = nn.Parameter(
            torch.zeros(1, action_shape), requires_grad=True
        )
        self.gsde_mean = nn.Linear(obs_shape, action_shape)

    def get_value(self, x):
        return self.critic(x)

    def sample_theta_gsde(self, x):
        action_mean = self.gsde_mean(x)
        gsde_std = torch.exp(self.gsde_logstd.expand_as(action_mean))
        self.theta_gsde = Normal(0, gsde_std).rsample()

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        probs = Normal(action_mean, action_std)

        if action is None:
            noise = self.theta_gsde
            action = probs.sample() + noise
        log_prob = probs.log_prob(action)
        return (
            action,
            log_prob.sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )

    def save(self, path_name):
        try:
            os.makedirs(path_name)
        except FileExistsError:
            pass
        torch.save(self.actor_logstd, f"{path_name}/actor_logstd")
        torch.save(self.actor_mean, f"{path_name}/actor_mean")
        torch.save(self.critic, f"{path_name}/critic")
        torch.save(self.gsde_logstd, f"{path_name}/gsde_logstd")
        torch.save(self.gsde_mean, f"{path_name}/gsde_mean")

    def load(self, path_name):
        self.actor_logstd = torch.load(f"{path_name}/actor_logstd")
        self.actor_mean = torch.load(f"{path_name}/actor_mean")
        self.critic = torch.load(f"{path_name}/critic")
        self.gsde_logstd = torch.load(f"{path_name}/gsde_logstd")
        self.gsde_mean = torch.load(f"{path_name}/gsde_mean")
