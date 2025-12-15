import argparse
import collections
import random
from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    """Helper to build a 2-layer MLP with ReLU activations."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class ActorEncoder(nn.Module):
    """Encode state into a latent code."""

    def __init__(self, state_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = mlp(state_dim, hidden_dim, latent_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class Decoder(nn.Module):
    """Decode latent code + state into final action."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        action_max: float,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.action_max = action_max
        self.net = mlp(state_dim + latent_dim, hidden_dim, action_dim)

    def forward(self, latent: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        x = torch.cat([latent, state], dim=-1)
        action = torch.tanh(self.net(x)) * self.action_max
        return action


class Critic(nn.Module):
    """Standard TD3 critic network."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q = mlp(state_dim + action_dim, hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.q(x)


class ReplayBuffer:
    """Replay buffer storing state, final action, reward, next_state, done."""

    def __init__(self, state_dim: int, action_dim: int, capacity: int = 100000):
        self.capacity = capacity
        self.state_buf = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action_buf = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state_buf = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ):
        idx = self.ptr % self.capacity
        self.state_buf[idx] = state
        self.action_buf[idx] = action
        self.reward_buf[idx] = reward
        self.next_state_buf[idx] = next_state
        self.done_buf[idx] = done
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            state=torch.as_tensor(self.state_buf[idxs], device=device),
            action=torch.as_tensor(self.action_buf[idxs], device=device),
            reward=torch.as_tensor(self.reward_buf[idxs], device=device),
            next_state=torch.as_tensor(self.next_state_buf[idxs], device=device),
            done=torch.as_tensor(self.done_buf[idxs], device=device),
        )
        return batch


@dataclass
class TD3Config:
    state_dim: int
    action_dim: int
    action_max: float
    latent_dim: int = 8
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2


class TD3Latent:
    """TD3 with latent actor and decoder."""

    def __init__(self, config: TD3Config, device: torch.device):
        self.device = device
        self.config = config

        self.actor = ActorEncoder(config.state_dim, config.latent_dim).to(device)
        self.actor_target = ActorEncoder(config.state_dim, config.latent_dim).to(device)
        self.decoder = Decoder(
            config.state_dim, config.action_dim, config.latent_dim, config.action_max
        ).to(device)
        self.decoder_target = Decoder(
            config.state_dim, config.action_dim, config.latent_dim, config.action_max
        ).to(device)
        self.critic1 = Critic(config.state_dim, config.action_dim).to(device)
        self.critic2 = Critic(config.state_dim, config.action_dim).to(device)
        self.critic1_target = Critic(config.state_dim, config.action_dim).to(device)
        self.critic2_target = Critic(config.state_dim, config.action_dim).to(device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        # Decoder shares the actor update step, but we keep its own optimizer for clarity.
        self.decoder_opt = torch.optim.Adam(self.decoder.parameters(), lr=config.actor_lr)
        critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self.critic_opt = torch.optim.Adam(critic_params, lr=config.critic_lr)

        self._hard_update_all()

        self.total_it = 0

    def _hard_update_all(self):
        for target, src in [
            (self.actor_target, self.actor),
            (self.decoder_target, self.decoder),
            (self.critic1_target, self.critic1),
            (self.critic2_target, self.critic2),
        ]:
            target.load_state_dict(src.state_dict())

    def select_action(self, state: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        state_t = torch.as_tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            latent = self.actor(state_t)
            action = self.decoder(latent, state_t)
        action = action.cpu().numpy()[0]
        if noise_std > 0.0:
            noise = np.random.normal(0, noise_std, size=action.shape)
            action = action + noise
        return np.clip(action, -self.config.action_max, self.config.action_max)

    def train_step(self, replay_buffer: ReplayBuffer, batch_size: int):
        if replay_buffer.size < batch_size:
            return {}

        self.total_it += 1
        batch = replay_buffer.sample(batch_size, self.device)
        state = batch["state"]
        action = batch["action"]
        reward = batch["reward"]
        next_state = batch["next_state"]
        done = batch["done"]

        with torch.no_grad():
            next_latent = self.actor_target(next_state)
            noise = (
                torch.randn_like(action) * self.config.policy_noise
            ).clamp(-self.config.noise_clip, self.config.noise_clip)
            next_action = self.decoder_target(next_latent, next_state)
            next_action = (next_action + noise).clamp(
                -self.config.action_max, self.config.action_max
            )
            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target = reward + (1 - done) * self.config.gamma * target_q

        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        info = {"critic_loss": critic_loss.item()}

        if self.total_it % self.config.policy_delay == 0:
            latent = self.actor(state)
            current_action = self.decoder(latent, state)
            actor_loss = -self.critic1(state, current_action).mean()

            self.actor_opt.zero_grad()
            self.decoder_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            self.decoder_opt.step()

            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.decoder_target, self.decoder)
            self._soft_update(self.critic1_target, self.critic1)
            self._soft_update(self.critic2_target, self.critic2)
            info["actor_loss"] = actor_loss.item()

        return info

    def _soft_update(self, target: nn.Module, src: nn.Module):
        tau = self.config.tau
        for t_param, param in zip(target.parameters(), src.parameters()):
            t_param.data.mul_(1 - tau).add_(tau * param.data)


def evaluate(env: gym.Env, agent: TD3Latent, episodes: int = 10, noise: float = 0.0):
    returns = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action = agent.select_action(state, noise_std=noise)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_ret += reward
        returns.append(ep_ret)
    return float(np.mean(returns))


def train(args):
    env = gym.make("Pendulum-v1")
    eval_env = gym.make("Pendulum-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_max = float(env.action_space.high[0])

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    config = TD3Config(
        state_dim=state_dim,
        action_dim=action_dim,
        action_max=action_max,
        latent_dim=args.latent_dim,
        policy_delay=args.policy_delay,
    )
    agent = TD3Latent(config, device)
    buffer = ReplayBuffer(state_dim, action_dim, capacity=args.buffer_size)

    state, _ = env.reset(seed=args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    episode_reward = 0.0
    log_rewards = []

    for t in range(1, args.total_steps + 1):
        if t < args.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, noise_std=args.exploration_noise)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(state, action, reward, next_state, float(done))

        state = next_state
        episode_reward += reward

        if done:
            log_rewards.append(episode_reward)
            state, _ = env.reset()
            episode_reward = 0.0

        if t >= args.update_after:
            agent.train_step(buffer, args.batch_size)

        if t % args.eval_interval == 0:
            avg_ret = evaluate(eval_env, agent, episodes=args.eval_episodes)
            print(f"Step {t}: eval_return={avg_ret:.2f}")

    return agent, log_rewards


def parse_args():
    parser = argparse.ArgumentParser(description="TD3-Latent for Pendulum-v1")
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--total_steps", type=int, default=200_000)
    parser.add_argument("--start_steps", type=int, default=10_000)
    parser.add_argument("--update_after", type=int, default=1_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--buffer_size", type=int, default=100_000)
    parser.add_argument("--exploration_noise", type=float, default=0.1)
    parser.add_argument("--policy_delay", type=int, default=2)
    parser.add_argument("--eval_interval", type=int, default=1_000)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
