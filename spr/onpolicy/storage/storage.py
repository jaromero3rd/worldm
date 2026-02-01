from __future__ import annotations

from abc import ABC

import torch


class RolloutBuffer(ABC):
    """Implementation refers to https://github.com/leggedrobotics/rsl_rl"""

    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None  # privileged observations
            self.actions = None
            self.rewards = None
            self.dones = None  # terminated | truncated
            self.timeouts = None  # truncated
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, buffer_size, obs_shape, actions_shape, privileged_obs_shape=None, device="cpu"):
        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.total_samples = buffer_size * num_envs
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape
        self.device = device
        self.step = 0

        # environment
        self.observations = torch.zeros(buffer_size, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape is not None:
            self.privileged_observations = torch.zeros(buffer_size, num_envs, *privileged_obs_shape, device=self.device)
        else:
            self.privileged_observations = None
        self.actions = torch.zeros(buffer_size, num_envs, *actions_shape, device=self.device)
        self.rewards = torch.zeros(buffer_size, num_envs, 1, device=self.device)
        self.dones = torch.zeros(buffer_size, num_envs, 1, device=self.device).byte()
        self.timeouts = torch.zeros(buffer_size, num_envs, 1, device=self.device).byte()

        # estimation
        self.actions_log_prob = torch.zeros(buffer_size, num_envs, 1, device=self.device)
        self.mu = torch.zeros(buffer_size, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(buffer_size, num_envs, *actions_shape, device=self.device)
        self.values = torch.zeros(buffer_size, num_envs, 1, device=self.device)
        self.returns = torch.zeros(buffer_size, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(buffer_size, num_envs, 1, device=self.device)

    def add_transitions(self, transition: Transition):
        # check if the transition is valid
        if self.step >= self.buffer_size:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.timeouts[self.step].copy_(transition.timeouts.view(-1, 1))
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self.values[self.step].copy_(transition.values.view(-1, 1))
        self.step += 1

    def compute_returns(self, last_values, gamma, lam, normalize_advantage: bool = True):
        """GAE"""
        advantage = 0
        for step in reversed(range(self.buffer_size)):
            # if we are at the last step, bootstrap the return value
            if step == self.buffer_size - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            self.returns[step] = advantage + self.values[step]

        # Compute the advantages
        self.advantages = self.returns - self.values
        # Normalize the advantages if flag is set
        # This is to prevent double normalization (i.e. if per minibatch normalization is used)
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def mini_batch_generator(self, num_mini_batches, num_epochs):
        batch_size = self.num_envs * self.buffer_size
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Core
        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        # For PPO
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                # Create the mini-batch
                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                # Yield the mini-batch
                yield obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch

    def clear(self):
        self.step = 0
