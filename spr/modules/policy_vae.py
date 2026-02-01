from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from spr.dataset import TrajectoryDataset
from spr.utils.config import PolicyDecoderConfig, PolicyEncoderConfig, VAEConfig

from .modules import MLP, Transformer
from .normalizer import EmpiricalNormalization


class PolicyVAE(nn.Module):
    """
    Policy VAE that encapsulates PolicyEncoder, PolicyDecoder and ValueHead.
    """

    def __init__(self, state_dim: int, action_dim: int, n_objs: int, config: VAEConfig):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_objs = n_objs

        # Initialize Normalizers
        self.use_obs_norm = config.obs_norm
        self.use_value_norm = config.value_norm
        self.obs_normalizer = EmpiricalNormalization(shape=state_dim) if self.use_obs_norm else nn.Identity()
        self.value_normalizer = EmpiricalNormalization(shape=n_objs) if self.use_value_norm else nn.Identity()

        # Initialize Encoder
        if config.encoder.encoder_type == "mlp":
            self.encoder = PolicyEncoderMLP(state_dim, action_dim, n_objs, config.encoder)
        else:
            self.encoder = PolicyEncoder(state_dim, action_dim, n_objs, config.encoder)

        # Initialize Value Head
        self.value_head = PerObjectiveValueHead(
            config.encoder.obj_n_embd, n_objs, hidden_dims=config.value_head_hidden_dims
        )

        # Initialize Decoder (MetaActor)
        self.decoder = PolicyDecoder(
            state_dim=state_dim,
            action_dim=action_dim,
            policy_repr_dim=config.encoder.n_embd,
            config=config.decoder,
        )

    # ----------------------------- Unified Interface -----------------------------
    def encode(self, inputs: dict, return_distribution: bool = True):
        inputs = inputs.copy()  # copy to avoid modifying original dict
        inputs["context_states"] = self.obs_normalizer(inputs["context_states"])
        return self.encoder(inputs, return_distribution)

    def decode(self, observations: torch.Tensor, policy_repr: torch.Tensor) -> Normal:
        self.decoder.update_distribution(self.obs_normalizer(observations), policy_repr)
        return self.decoder.distribution

    def sample(self, mean: torch.Tensor, log_std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder.sample(mean, log_std)

    def get_obj_reprs(self, base_repr: torch.Tensor) -> torch.Tensor:
        return self.encoder.get_obj_reprs(base_repr)

    def act_inference(self, observations: torch.Tensor, policy_repr: torch.Tensor) -> torch.Tensor:
        return self.decoder.act_inference(self.obs_normalizer(observations), policy_repr)

    def get_value(self, repr: torch.Tensor) -> torch.Tensor:
        if repr.dim() == 3:  # [B, n_objs, obj_n_embd]
            obj_reprs = repr
        else:  # base_repr, [B, n_embd]
            obj_reprs = self.get_obj_reprs(repr)
        return self.value_head(obj_reprs).squeeze(-1)  # [B, n_objs]

    def get_orthogonality_loss(self) -> torch.Tensor:
        return self.encoder.get_orthogonality_loss()

    def weighted_gradients(
        self, base_features: np.ndarray, weights: np.ndarray, device: torch.device
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradients of weighted value w.r.t. base features."""
        z = torch.tensor(base_features, device=device, dtype=torch.float32, requires_grad=True)
        weights_t = torch.tensor(weights, device=device, dtype=torch.float32)
        values = self.value_normalizer.inverse(self.get_value(z))  # [B, n_objs]
        weighted_value = (values * weights_t).sum(-1)  # [B]
        grads = torch.autograd.grad(weighted_value.sum(), z)[0]  # [B, D]
        return grads.detach().cpu().numpy(), weighted_value.detach().cpu().numpy()

    def update_normalization(self, dataset: TrajectoryDataset):
        if self.use_obs_norm:
            self.obs_normalizer.update(dataset.get_all_observations())
            self.obs_normalizer.eval()
        if self.use_value_norm:
            self.value_normalizer.update(dataset.get_all_returns())
            self.value_normalizer.eval()

    def train(self):
        self.encoder.train()
        self.decoder.train()
        self.value_head.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        self.value_head.eval()


class PolicyEncoder(nn.Module):
    """Policy encoder with transformer encoder for base policy representation (a distribution) and
    orthogonal linear layers for per-objective policy representations."""

    def __init__(self, state_dim: int, action_dim: int, n_objs: int, config: PolicyEncoderConfig):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_objs = n_objs
        self.config = config

        # Shared Base Policy Representation Encoder
        self._init_base_encoder()
        self.base_head = nn.Linear(self.config.n_embd, 2 * self.config.n_embd)  # mean and log_std of base repr

        # Per-Objective Policy Representation Encoders
        self.obj_heads = nn.ModuleList(
            [
                MLP(
                    config.n_embd,
                    config.obj_n_embd,
                    hidden_dims=config.obj_hidden_dims,
                    activation=config.obj_activation,
                )
                for _ in range(n_objs)
            ]
        )

        # Optional Pretrained Embeddings
        self.states_embd = nn.Identity()
        self.actions_embd = nn.Identity()

    def forward(self, inputs: dict, return_distribution: bool = True):
        """Encode per-objective policy representations using separate encoders. Only the base policy representation comes from a distribution. Per-objective representations
        are deterministic linear projections of the sampled policy representation.

        Returns:
            If return_distribution=True:
                base_repr_mean: [B, n_embd] - base policy representation mean
                base_repr_logstd: [B, n_embd] - base policy representation log_std
            If return_distribution=False:
                base_repr: [B, n_embd] - sampled base policy representation
                obj_reprs: [B, n_objs, n_embd] - per-objective representations
        """
        context_states = inputs.get("context_states")  # [B, T_ctx, obs_dim] or [T_ctx, obs_dim]
        context_actions = inputs.get("context_actions")  # [B, T_ctx, action_dim] or [T_ctx, action_dim]
        if context_states.dim() == 2:  # [T_ctx, obs_dim]
            context_states = context_states.unsqueeze(0)  # [1, T_ctx, obs_dim]
            context_actions = context_actions.unsqueeze(0)  # [1, T_ctx, action_dim]

        # Preprocess context
        context = torch.cat([self.states_embd(context_states), self.actions_embd(context_actions)], dim=-1)

        # Policy representation distribution
        cls_embd = self._get_base_repr(context)
        base_repr_mean, base_repr_logstd = self.base_head(cls_embd).chunk(2, dim=-1)  # [B, n_embd], [B, n_embd]
        base_repr_logstd = torch.clamp(base_repr_logstd, min=-10, max=2)  # clamp for numerical stability

        if return_distribution:
            return base_repr_mean, base_repr_logstd
        else:
            return self.sample(base_repr_mean, base_repr_logstd)

    def _init_base_encoder(self):
        self.base_encoder = Transformer(self.config, token_dim=self.state_dim + self.action_dim)
        self.cls_token = nn.ParameterList([nn.Parameter(torch.randn(1, 1, self.state_dim + self.action_dim) * 0.02)])
        print(f"Num params in token embedding: {sum(p.numel() for p in self.base_encoder.embedding.parameters())}")
        print(f"Num params in self attention: {sum(p.numel() for p in self.base_encoder.encoder.parameters())}")

    def _get_base_repr(self, context: torch.Tensor) -> torch.Tensor:
        """Get base policy representation from the context."""
        sequence = torch.cat(
            [
                self.cls_token[0].expand(context.shape[0], -1, -1),  # [B, 1, token_dim] - policy summary
                context,  # [B, T_ctx, token_dim] - policy context
            ],
            dim=1,
        )  # [B, 1 + T_ctx, token_dim]
        return self.base_encoder(sequence)[:, 0, :]  # [B, n_embd] - encoded [CLS] token

    def get_obj_reprs(self, base_repr: torch.Tensor) -> torch.Tensor:
        """Get per-objective representations from the base representation."""
        return torch.stack([obj_head(base_repr) for obj_head in self.obj_heads], dim=1)  # [B, n_objs, obj_n_embd]

    def sample(self, base_repr_mean: torch.Tensor, base_repr_logstd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from the policy distribution and compute deterministic per-objective representations."""
        # Reparameterization trick
        base_repr_eps = torch.randn_like(base_repr_mean)
        base_repr = base_repr_mean + torch.exp(base_repr_logstd) * base_repr_eps

        # Per-objective representations
        obj_reprs = self.get_obj_reprs(base_repr)  # [B, n_objs, obj_n_embd]
        return base_repr, obj_reprs

    def get_orthogonality_loss(self) -> torch.Tensor:
        losses = []
        for head in self.obj_heads:
            W = head.layers[-1].weight  # [out_dim, in_dim]
            gram = W @ W.T if W.shape[0] <= W.shape[1] else W.T @ W
            loss = nn.functional.mse_loss(gram, torch.eye(gram.shape[0], device=gram.device), reduction="mean")
            losses.append(loss)
        return sum(losses) / len(losses)


class PerObjectiveValueHead(nn.Module):
    def __init__(self, repr_dim: int, n_objs: int = 1, hidden_dims: List[int] = []):
        super().__init__()
        self.value_heads = nn.ModuleList([MLP(repr_dim, 1, hidden_dims, activation="relu") for _ in range(n_objs)])

    def forward(self, obj_reprs: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(obj_reprs[:, i, :]) for i, head in enumerate(self.value_heads)], dim=-1)  # [B, n_objs]


class PolicyEncoderMLP(PolicyEncoder):
    """Policy encoder with average pooling MLP encoder for base policy representation. For benchmark purpose."""

    def _init_base_encoder(self):
        self.base_encoder = MLP(
            input_dim=self.state_dim + self.action_dim,
            output_dim=self.config.n_embd,
            hidden_dims=self.config.embd_hidden_dims,
            activation=self.config.embd_activation,
        )
        print(f"Num params in base encoder: {sum(p.numel() for p in self.base_encoder.parameters())}")

    def _get_base_repr(self, context: torch.Tensor) -> torch.Tensor:
        """Encode context using MLP + Average Pooling, [B, T, token_dim] -> [B, T, n_embd] -> [B, n_embd]"""
        return self.base_encoder(context).mean(dim=1)


class PolicyDecoder(nn.Module):
    """Actor network that takes in state and policy representation to predict actions."""

    def __init__(self, state_dim: int, action_dim: int, policy_repr_dim: int, config: PolicyDecoderConfig):
        super().__init__()
        self.config = config
        self.noise_std_type = config.noise_std_type
        self.init_noise_std = config.init_noise_std
        self.min_log_noise_std = config.min_log_noise_std
        self.max_log_noise_std = config.max_log_noise_std
        self.deterministic = config.deterministic

        self.state_encoder = MLP(state_dim, config.state_repr_dim, config.state_hidden_dims, config.activation)
        self.policy_decoder = MLP(
            input_dim=config.state_repr_dim + policy_repr_dim,
            output_dim=action_dim,
            hidden_dims=config.policy_hidden_dims,
            activation=config.activation,
        )
        self.log_std_decoder = MLP(
            input_dim=policy_repr_dim,
            output_dim=action_dim,
            hidden_dims=config.policy_hidden_dims,
            activation=config.activation,
        )

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations: torch.Tensor, policy_repr: torch.Tensor):
        """
        Args:
            observations: [B, state_dim] - observations
            policy_repr: [B, policy_repr_dim] - policy representation
        """
        # compute mean
        mean = self.forward(observations, policy_repr)  # [B, action_dim]

        # compute log_std
        log_std = self.log_std_decoder(policy_repr)  # [B, action_dim]
        log_std = torch.clamp(log_std, self.min_log_noise_std, self.max_log_noise_std)
        std = torch.exp(log_std)  # [B, action_dim]
        self.distribution = Normal(mean, std)  # [B, action_dim]

    def act(self, observations: torch.Tensor, policy_repr: torch.Tensor) -> torch.Tensor:
        self.update_distribution(observations, policy_repr)
        if self.deterministic:
            return self.distribution.mean
        else:
            return self.distribution.rsample()  # reparametrize trick

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    @torch.no_grad()
    def act_inference(self, observations: torch.Tensor, policy_repr: torch.Tensor) -> torch.Tensor:
        actions_mean = self.forward(observations, policy_repr)
        return actions_mean

    def forward(self, observations: torch.Tensor, policy_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: [B, state_dim] - observations
            policy_repr: [B, policy_repr_dim] - policy representation
        """
        encoded_states = self.state_encoder(observations)  # [B, state_repr_dim]
        if policy_repr.shape[0] != observations.shape[0] and policy_repr.shape[0] == 1:
            policy_repr = policy_repr.expand(observations.shape[0], -1)  # [B, policy_repr_dim]
        return self.policy_decoder(torch.cat([encoded_states, policy_repr], dim=-1))
