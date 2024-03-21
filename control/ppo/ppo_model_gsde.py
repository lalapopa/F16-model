import os
import time
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from utils import (
    state_logger,
    weight_histograms,
    write_to_tensorboard,
    calculate_episode_nmae,
)
import F16model.utils.control_metrics as utils_metrics


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Agents for PPO algo"""

    def __init__(self, env, config):
        super(Agent, self).__init__()
        self.config = config
        self._setup_seed()
        self.device = self._get_device()
        self.env = env
        self.action_shape = np.array(env.single_action_space.shape).prod()
        self.obs_shape = np.array(env.single_observation_space.shape).prod()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        ).to(self.device)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(self.obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.action_shape), std=0.01),
        ).to(
            self.device
        )  # aka Policy
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_shape)).to(
            self.device
        )

        self.gsde_mean = nn.Sequential(
            layer_init(nn.Linear(self.obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.action_shape), std=0.01),
        ).to(self.device)
        self.gsde_logstd = nn.Parameter(torch.zeros(1, self.action_shape)).to(
            self.device
        )

    def get_value(self, x):
        return self.critic(x)

    def sample_theta_gsde(self, x, sample_idx=[]):
        action_mean = self.gsde_mean(x)
        gsde_std = torch.exp(self.gsde_logstd.expand_as(action_mean))
        theta_gsde_temp = Normal(0, gsde_std).rsample()
        if sample_idx:
            for idx_env in sample_idx:
                self.theta_gsde[idx_env] = theta_gsde_temp[idx_env]
        else:
            self.theta_gsde = theta_gsde_temp

    def get_action_and_value(self, x, action=None, learn_feature=False):
        action_mean = (
            self.actor_mean(x) if learn_feature else self.actor_mean(x).detach()
        )
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        probs = Normal(action_mean, action_std)

        if action is None:
            noise = self.theta_gsde * action_std
            action = probs.mean + noise
        log_prob = (
            probs.log_prob(action) if learn_feature else probs.log_prob(action.detach())
        )
        return (
            action,
            log_prob.sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )

    def save(self):
        try:
            os.makedirs(f"{self.config.save_dir}/{self.run_name}")
        except FileExistsError:
            pass
        torch.save(
            self.actor_logstd, f"{self.config.save_dir}/{self.run_name}/actor_logstd"
        )
        torch.save(
            self.actor_mean, f"{self.config.save_dir}/{self.run_name}/actor_mean"
        )
        torch.save(self.critic, f"{self.config.save_dir}/{self.run_name}/critic")
        torch.save(
            self.gsde_logstd, f"{self.config.save_dir}/{self.run_name}/gsde_logstd"
        )
        torch.save(self.gsde_mean, f"{self.config.save_dir}/{self.run_name}/gsde_mean")
        print(f"Saving model in: {self.config.save_dir}/{self.run_name}")

    def load(self, path_name):
        self.actor_logstd = torch.load(
            f"{path_name}/actor_logstd", map_location=self._get_device()
        )
        self.actor_mean = torch.load(
            f"{path_name}/actor_mean", map_location=self._get_device()
        )
        self.critic = torch.load(f"{path_name}/critic", map_location=self._get_device())
        self.gsde_logstd = torch.load(
            f"{path_name}/gsde_logstd", map_location=self._get_device()
        )
        self.gsde_mean = torch.load(
            f"{path_name}/gsde_mean", map_location=self._get_device()
        )

    def _setup_tb_log(self):
        writer = SummaryWriter(f"{self.config.save_dir}/{self.run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % (
                "\n".join(
                    [f"|{key}|{value}|" for key, value in vars(self.config).items()]
                )
            ),
        )
        return writer

    def _setup_seed(self):
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.backends.cudnn.deterministic = self.config.torch_deterministic

    def _get_device(self):
        return torch.device(
            "cuda" if torch.cuda.is_available() and self.config.cuda else "cpu"
        )

    def _init_episode_buffer(self):
        obs = torch.zeros(
            (self.config.num_steps, self.config.num_envs) + (self.obs_shape,),
            dtype=torch.float32,
        ).to(self.device)
        actions = torch.zeros(
            (self.config.num_steps, self.config.num_envs) + (self.action_shape,),
            dtype=torch.float32,
        ).to(self.device)
        logprobs = torch.zeros(
            (self.config.num_steps, self.config.num_envs), dtype=torch.float32
        ).to(self.device)
        rewards = torch.zeros(
            (self.config.num_steps, self.config.num_envs), dtype=torch.float32
        ).to(self.device)
        dones = torch.zeros(
            (self.config.num_steps, self.config.num_envs), dtype=torch.float32
        ).to(self.device)
        values = torch.zeros(
            (self.config.num_steps, self.config.num_envs), dtype=torch.float32
        ).to(self.device)
        return obs, actions, logprobs, rewards, dones, values

    def train(self, run_name):
        self.run_name = run_name
        print(f"Start running: {self.run_name}")
        writer = self._setup_tb_log()
        print(f"Traninig using {self.device}")

        weight_histograms(writer, 0, self.actor_mean)
        optimizer = optim.Adam(
            self.parameters(), lr=self.config.learning_rate, amsgrad=True
        )

        self.to(self.device)

        global_step = 0
        start_time = time.time()
        min_nMAE_metric = np.inf
        num_updates = self.config.total_timesteps // self.config.batch_size

        for update in range(1, num_updates + 1):
            state_logger(
                self.run_name, init_state=self.env.call("init_state")[0].to_array()
            )

            if self.config.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.config.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            obs, actions, logprobs, rewards, dones, values = self._init_episode_buffer()
            obs_init, _ = self.env.reset(seed=self.config.seed + update)
            next_obs = torch.Tensor(obs_init).to(self.device)
            next_done = torch.zeros(self.config.num_envs).to(self.device)
            with torch.no_grad():
                self.sample_theta_gsde(next_obs)
                init_gsde_state = next_obs
            for step in range(0, self.config.num_steps):
                global_step += 1 * self.config.num_envs
                obs[step] = next_obs
                dones[step] = next_done
                with torch.no_grad():
                    action, logprob, _, value = self.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob
                state_logger(self.run_name, action=action.cpu().numpy()[0])
                next_obs, reward, done, _, info = self.env.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(
                    self.device
                ), torch.Tensor(done).to(self.device)
                with torch.no_grad():
                    self.sample_theta_gsde(next_obs)
                if done.any():  # ONLY for perfomance monitor & resample theta_gsde
                    done_envs = []  # EXAMPLE: [0, 1, 2, 3] pick all paralel env
                    _, done_envs = write_to_tensorboard(writer, info, global_step)
                    for idx_done_env in done_envs:
                        init_gsde_state[idx_done_env] = next_obs[idx_done_env]
                    nMAE_avg = calculate_episode_nmae(obs, done_envs, step)
                    if nMAE_avg < min_nMAE_metric:
                        min_nMAE_metric = nMAE_avg
                        print(f"NEW best nMAE {nMAE_avg}")

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.get_value(next_obs).reshape(1, -1)
                if self.config.gae:
                    advantages = torch.zeros_like(rewards).to(self.device)
                    lastgaelam = 0
                    for t in reversed(range(self.config.num_steps)):
                        if t == self.config.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = (
                            rewards[t]
                            + self.config.gamma * nextvalues * nextnonterminal
                            - values[t]
                        )
                        advantages[t] = lastgaelam = (
                            delta
                            + self.config.gamma
                            * self.config.gae_lambda
                            * nextnonterminal
                            * lastgaelam
                        )
                    returns = advantages + values
                else:
                    returns = torch.zeros_like(rewards).to(self.device)
                    for t in reversed(range(self.config.num_steps)):
                        if t == self.config.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            next_return = returns[t + 1]
                        returns[t] = (
                            rewards[t]
                            + self.config.gamma * nextnonterminal * next_return
                        )
                    advantages = returns - values

            # flatten the batch
            b_obs = obs.reshape((-1,) + (self.obs_shape,))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + (self.action_shape,))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.config.batch_size)
            clipfracs = []
            for epoch in range(self.config.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(
                    0, self.config.batch_size, self.config.minibatch_size
                ):
                    end = start + self.config.minibatch_size
                    mb_inds = b_inds[start:end]
                    self.sample_theta_gsde(b_obs[mb_inds])
                    # print(
                    #     f"before Deadge\nobs:{b_obs[mb_inds]}\naction:{b_actions.long()[mb_inds]}"
                    # )
                    _, newlogprob, entropy, newvalue = self.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds], learn_feature=True
                    )
                    # print(f"{newlogprob = }\n{entropy = }\n{newvalue = }")
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()  # exp(log(p) - log(q)) = p/q

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.config.clip_coef)
                            .float()
                            .mean()
                            .item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if self.config.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )
                    # Policy loss
                    pg_loss1 = (
                        -mb_advantages * ratio
                    )  # - A(s,a) * policy_new/ policy_old
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef
                    )  # part of Objective function
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()  # L^{CLIP}(theta)

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.config.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.config.clip_coef,
                            self.config.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss
                        - self.config.ent_coef * entropy_loss
                        + v_loss * self.config.vf_coef
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.parameters(), self.config.max_grad_norm
                    )
                    optimizer.step()

                if self.config.target_kl is not None:
                    if approx_kl > self.config.target_kl:
                        break

            print(f"Total_loss = {loss:.2f}")
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )
            # logs
            log_data = {
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "charts/SPS": int(global_step / (time.time() - start_time)),
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
                "losses/explained_variance": explained_var,
                "train/step": global_step,
            }
            for name, value in log_data.items():
                writer.add_scalar(name, value, global_step)
                weight_histograms(writer, global_step, self.actor_mean)
            if round(global_step, -3) % 10000 == 0:
                print(f"Saving model after {global_step}")
                self.save()
        print(f"Saving model after {global_step}")
        self.save()
        writer.close()
        return min_nMAE_metric
