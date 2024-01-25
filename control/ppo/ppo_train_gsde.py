import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from ppo_model_gsde import Agent
from F16model.env import F16
from utils import (
    parse_args,
    state_logger,
    weight_histograms,
    write_to_tensorboard,
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

    run_name = f"F16__{args.exp_name}__{args.seed}__{str(int(time.time()))}_{('%032x' % random.getrandbits(128))[:4]}"
    print(f"Start running: {run_name}")
    write_python_file(
        os.path.abspath(__file__), f"runs/{run_name}/{os.path.basename(__file__)}"
    )
    write_python_file(
        os.path.abspath(__file__).replace("train", "model"),
        f"runs/{run_name}/{os.path.basename(__file__).replace('train', 'model')}",
    )
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        wandb.define_metric("train/step")
        wandb.define_metric("charts/*", step_metric="train/step")
        wandb.define_metric("losses/*", step_metric="train/step")

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Traninig using {device}")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.seed + i,
            )
            for i in range(args.num_envs)
        ]
    )

    action_size = np.array(envs.single_action_space.shape).prod()
    obs_size = np.array(envs.single_observation_space.shape).prod()

    agent = Agent(obs_size, action_size).to(device)
    weight_histograms(writer, 0, agent.actor_mean)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + (obs_size,)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + (action_size,)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    obs_init, _ = envs.reset(seed=args.seed)

    next_obs = torch.Tensor(obs_init).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    for update in range(1, num_updates + 1):
        state_logger(run_name, init_state=envs.call("init_state")[0].to_array())

        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                agent.sample_theta_gsde(next_obs)
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            state_logger(run_name, action=action.cpu().numpy()[0])
            next_obs, reward, done, _, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                done
            ).to(device)
            write_to_tensorboard(writer, info, global_step, args)
        print(f"|{update}|{num_updates + 1}|")
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + args.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + (obs_size,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + (action_size,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                agent.sample_theta_gsde(b_obs[mb_inds])
                print(
                    f"before Deadge\nobs:{b_obs[mb_inds]}\naction:{b_actions.long()[mb_inds]}"
                )
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                print(f"{newlogprob = }\n{entropy = }\n{newvalue = }")
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()  # exp(log(p) - log(q)) = p/q

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )
                print(f"{mb_advantages = }")
                # Policy loss
                pg_loss1 = -mb_advantages * ratio  # - A(s,a) * policy_new/ policy_old
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )  # part of Objective function
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()  # L^{CLIP}(theta)

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                print(f"{loss = }\n{entropy_loss = }")
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
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
            weight_histograms(writer, global_step, agent.actor_mean)
        if args.track:
            wandb.log(log_data)
        if round(global_step, -3) % 10000 == 0:
            print(f"Saving model after {global_step}")
            agent.save(f"runs/models/{run_name}")

    agent.save(f"runs/models/{run_name}")
    writer.close()
    if args.track:
        wandb.finish()
