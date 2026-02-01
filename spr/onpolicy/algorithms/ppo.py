from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from spr.modules import *
from spr.onpolicy.storage import RolloutBuffer


class PPO:
    """Proximal Policy Optimization algorithm based on (https://arxiv.org/abs/1707.06347)
    with multi-objective support."""

    def __init__(self, env, alg_cfg: dict, device: str = "cuda:0"):
        self.device = device
        self.alg_cfg = alg_cfg
        self.env = env
        self.device_type = torch.device(device).type
        self._init_hyperparameters()
        self._init_models()
        self._init_storage()
        self._init_normalization()

    # ----------------------------- Initialization --------------------------------
    def _init_hyperparameters(self):
        # learning parameters
        self.num_learning_epochs = self.alg_cfg.get("num_learning_epochs", 5)
        self.num_mini_batches = self.alg_cfg.get("num_mini_batches", 4)
        self.num_steps_per_env = self.alg_cfg.get("num_steps_per_env", 128)
        self.gamma = self.alg_cfg.get("gamma", 0.99)
        self.actor_lr = self.alg_cfg.get("actor_learning_rate", 3e-4)
        self.critic_lr = self.alg_cfg.get("critic_learning_rate", 3e-4)
        self.max_grad_norm = self.alg_cfg.get("max_grad_norm", 1.0)
        self.value_loss_coef = self.alg_cfg.get("value_loss_coef", 0.5)
        self.entropy_coef = self.alg_cfg.get("entropy_coef", 0.0)
        self.mixed_precision = self.alg_cfg.get("mixed_precision", False)
        self.obs_normalization = self.alg_cfg.get("obs_normalization", False)
        self.reward_normalization = self.alg_cfg.get("reward_normalization", False)
        self.normalize_advantage = self.alg_cfg.get("normalize_advantage", True)
        self.action_bound_method = self.alg_cfg.get("action_bound_method", "clip")
        self.action_scaling = self.alg_cfg.get("action_scaling", False)
        self.init_actor_weights = self.alg_cfg.get("init_actor_weights", False)

        # PPO parameters
        self.lam = self.alg_cfg.get("lam", 0.95)
        self.clip_param = self.alg_cfg.get("clip_param", 0.2)
        self.use_clipped_value_loss = self.alg_cfg.get("use_clipped_value_loss", False)

        # Multi-objective training
        self.n_objs = getattr(self.env, "num_rewards", 1)
        self.w_objs = torch.ones((self.n_objs, 1), device=self.device)

    def _init_models(self):
        # initialize actor and critic
        self.policy_cfg = self.alg_cfg.get("policy_cfg", {})
        self.actor_cfg = self.policy_cfg.get("actor_cfg", {})
        self.critic_cfg = self.policy_cfg.get("critic_cfg", {})
        self._init_actor()
        self._init_critic()

    def _init_actor(self):
        self.actor = ActorMLP(
            state_dim=self.env.num_obs,
            action_dim=self.env.num_actions,
            hidden_dims=self.actor_cfg.get("hidden_dims", [256, 256]),
            noise_std_type=self.actor_cfg.get("noise_std_type", "log"),
            init_noise_std=self.actor_cfg.get("init_noise_std", 0.3),
            min_log_noise_std=self.actor_cfg.get("min_log_noise_std", -20.0),
            max_log_noise_std=self.actor_cfg.get("max_log_noise_std", 2.0),
            activation=self.actor_cfg.get("activation", "relu"),
            deterministic=self.actor_cfg.get("deterministic", False),
        ).to(self.device)

        if self.init_actor_weights:
            for m in list(self.actor.modules()):
                if isinstance(m, torch.nn.Linear):
                    # orthogonal initialization
                    torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    torch.nn.init.zeros_(m.bias)

            # do last policy layer scaling, this will make initial actions have (close to)
            # 0 mean and std, and will help boost performances,
            # see https://arxiv.org/abs/2006.05990, Fig.24 for details
            for layer in reversed(self.actor.layers):
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.zeros_(layer.bias)
                    layer.weight.data.copy_(0.01 * layer.weight.data)
                    break

        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=self.actor_lr)

    def _init_critic(self):
        self.critic = CriticMLP(
            input_dim=self.env.num_obs,
            output_dim=1,
            hidden_dims=self.critic_cfg.get("hidden_dims", [256, 256]),
            activation=self.critic_cfg.get("activation", "relu"),
        ).to(self.device)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=self.critic_lr)

    def _init_storage(self):
        self.storage = RolloutBuffer(
            num_envs=self.env.num_envs,
            buffer_size=self.num_steps_per_env,
            obs_shape=[self.env.num_obs],
            actions_shape=[self.env.num_actions],
            device=self.device,
        )
        self.transition = RolloutBuffer.Transition()

    def _init_normalization(self):
        # normalize observation
        if self.obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[self.env.num_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

        # normalize reward
        if self.reward_normalization:
            self.reward_normalizer = EmpiricalDiscountedVariationNormalization(shape=[1], until=1.0e8).to(self.device)
        else:
            self.reward_normalizer = torch.nn.Identity().to(self.device)  # no normalization

    # ----------------------------- Interaction --------------------------------
    def act(self, obs):
        """Act based on current obs."""
        # Compute the actions and values
        normed_obs = self.obs_normalizer(obs)
        with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16, enabled=self.mixed_precision):
            actions = self.actor.act(normed_obs).detach()
            self.transition.actions = actions
            self.transition.actions_log_prob = self.actor.get_actions_log_prob(actions).detach()
            self.transition.values = self.critic.forward(normed_obs).detach()
        self.transition.action_mean = self.actor.action_mean.detach()
        self.transition.action_sigma = self.actor.action_std.detach()
        return actions

    def store_transition(self, obs, actions, next_obs, rewards, dones, infos):
        """Store the environment step"""
        # Record the transition
        if self.obs_normalization:
            self.obs_normalizer.update(obs)
        self.transition.observations = obs

        # scalarize rewards
        rewards = rewards.matmul(self.w_objs)  # [num_envs, 1]

        # this should be the only place that updates reward normalizer, clone here because
        # we bootstrapped the rewards
        if self.reward_normalization:
            self.reward_normalizer.update(rewards)
        self.transition.rewards = self.reward_normalizer(rewards.clone())

        # bootstrap reward for timeouts episodes
        if "time_outs" in infos:
            self.transition.rewards += (
                self.gamma * self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device)
            )

        self.transition.dones = dones
        self.transition.timeouts = infos["time_outs"]

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()

    def postprocess_actions(self, raw_action):
        """Map raw action to the action space. First clip or tanh the raw action, then scale it to the action space."""
        if self.action_bound_method is None:
            action = raw_action.clone()
        elif self.action_bound_method == "clip":
            action = torch.clamp(raw_action, -1.0, 1.0)
        elif self.action_bound_method == "tanh":
            action = torch.tanh(raw_action)
        else:
            raise ValueError(f"Invalid action bound method: {self.action_bound_method}")

        if self.action_scaling:
            assert (
                torch.min(action) >= -1.0 and torch.max(action) <= 1.0
            ), "action scaling only accepts raw action range = [-1, 1]"
            low, high = torch.tensor(self.env.action_space.low).to(self.device), torch.tensor(
                self.env.action_space.high
            ).to(self.device)
            action = low + (high - low) * (action + 1.0) / 2.0  # type: ignore
        return action

    # ----------------------------- Training --------------------------------
    def compute_returns(self, normed_last_critic_obs):
        with torch.no_grad(), torch.autocast(
            device_type=self.device_type, dtype=torch.bfloat16, enabled=self.mixed_precision
        ):
            last_values = self.critic.forward(normed_last_critic_obs).detach()
            self.storage.compute_returns(
                last_values,
                self.gamma,
                self.lam,
                normalize_advantage=self.normalize_advantage,
            )

    def update(self, last_critic_obs):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_explained_variance = 0

        # compute values, advantages, and returns
        self.compute_returns(self.obs_normalizer(last_critic_obs))

        # generator for mini batches
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:

            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16, enabled=self.mixed_precision):
                # Recompute actions log prob and entropy for current batch of transitions
                # -- actor
                normed_obs_batch = self.obs_normalizer(obs_batch)
                self.actor.act(normed_obs_batch)
                actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)

                # -- critic
                value_batch = self.critic(normed_obs_batch)

                # -- entropy
                entropy_batch = self.actor.entropy

                # policy loss
                policy_loss = self._compute_policy_loss(
                    actions_log_prob_batch, old_actions_log_prob_batch, advantages_batch
                )

                # value loss
                value_loss = self._compute_value_loss(returns_batch, value_batch, target_values_batch)

                # record explained variance
                mean_explained_variance += (
                    (1 - torch.var(returns_batch - value_batch) / torch.var(returns_batch)).item()
                    / self.num_learning_epochs
                    / self.num_mini_batches
                )

                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += policy_loss.item()
            mean_entropy_loss += entropy_batch.mean().item()

        # compute mean loss
        num_updates = self.num_mini_batches * self.num_learning_epochs
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates

        # clear storage
        self.storage.clear()

        return (
            mean_value_loss,
            mean_surrogate_loss,
            {
                "mean_entropy_loss": mean_entropy_loss,
                "mean_explained_variance": mean_explained_variance,
            },
        )

    def _compute_policy_loss(self, actions_log_prob_batch, old_actions_log_prob_batch, advantages_batch):
        """Clipped PPO surrogate loss"""
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        return torch.max(surrogate, surrogate_clipped).mean()

    def _compute_value_loss(self, returns_batch, value_batch, target_values_batch):
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -self.clip_param, self.clip_param
            )
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            return torch.max(value_losses, value_losses_clipped).mean()
        else:
            return (returns_batch - value_batch).pow(2).mean()

    def update_objective_weights(self, objective_weights):
        """Once objective weights are updated, reinitialize actor and critic"""
        self.w_objs.copy_(torch.FloatTensor(objective_weights).view(self.n_objs, 1).to(device=self.device))
        self._init_actor()
        self._init_critic()
        self._init_normalization()

    def train_mode(self):
        self.actor.train()
        self.critic.train()
        self.obs_normalizer.train()
        self.reward_normalizer.train()

    def eval_mode(self):
        self.actor.eval()
        self.critic.eval()
        self.obs_normalizer.eval()
        self.reward_normalizer.eval()
