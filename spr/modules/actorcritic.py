# Modules for Online PPO Training
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from .modules import MLP


class ActorMLP(MLP):
    """Regular MLP actor network, pi(a|s)"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        noise_std_type: str = "log",
        init_noise_std: float = 1.0,
        min_log_noise_std: float = -20.0,
        max_log_noise_std: float = 2.0,
        activation: str = "relu",
        deterministic: bool = False,
    ):
        super().__init__(state_dim, action_dim, hidden_dims, activation)
        self.actor_type = "mlp"

        self.distribution = Normal(torch.zeros(action_dim), init_noise_std * torch.ones(action_dim))
        self.init_noise_std = init_noise_std
        self.min_noise_std = np.exp(min_log_noise_std)
        self.max_noise_std = np.exp(max_log_noise_std)
        self.noise_std_type = noise_std_type
        self.deterministic = deterministic

        if not self.deterministic:
            # Action noise
            if noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(action_dim))
            elif noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(action_dim)))
            else:
                raise ValueError(f"Unknown standard deviation type: {noise_std_type}. Should be 'scalar' or 'log'")

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        # compute mean
        mean = self.forward(observations)
        if self.deterministic:
            self.distribution = Normal(
                mean, torch.clamp(self.init_noise_std * torch.ones(mean.shape), self.min_noise_std, self.max_noise_std)
            )
        elif not self.deterministic:
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            # update distribution
            self.distribution = Normal(mean, torch.clamp(std, self.min_noise_std, self.max_noise_std))

    def act(self, observations):
        self.update_distribution(observations)
        if self.deterministic:
            return self.distribution.mean
        else:
            return self.distribution.rsample()  # reparametrize trick

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    @torch.no_grad()
    def act_inference(self, observations):
        actions_mean = self.forward(observations)
        return actions_mean


class CriticMLP(MLP):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], activation: str = "elu"):
        super().__init__(input_dim, output_dim, hidden_dims, activation)
        self.critic_type = "mlp"
