from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import MLP, ResNet18

class JEPAWorldModel(nn.Module):
    """
    Joint-Embedding Predictive Architecture (JEPA) World Model for vector observations.
    """
    def __init__(
        self, 
        obs_dim: int, 
        act_dim: int, 
        latent_dim: int = 256,
        hidden_dim: int = 512,
        ema_decay: float = 0.99,
        encoder_type: str = "resnet18",
        predictor_type: str = "mlp",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay

        # Encoder selection
        if encoder_type == "resnet18":
            self.encoder = ResNet18(input_dim=obs_dim, output_dim=latent_dim)
        else: # default to MLP
            self.encoder = MLP(input_dim=obs_dim, output_dim=latent_dim, hidden_dims=[hidden_dim, hidden_dim])

        # Target Encoder (same type as encoder)
        if encoder_type == "resnet18":
            self.target_encoder = ResNet18(input_dim=obs_dim, output_dim=latent_dim)
        else:
            self.target_encoder = MLP(input_dim=obs_dim, output_dim=latent_dim, hidden_dims=[hidden_dim, hidden_dim])
            
        # Initialize target encoder with encoder weights
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Predictor selection
        if predictor_type == "resnet18":
            self.predictor = ResNet18(input_dim=latent_dim + act_dim, output_dim=latent_dim)
        else: # default to MLP
            self.predictor = MLP(input_dim=latent_dim + act_dim, output_dim=latent_dim, hidden_dims=[hidden_dim, hidden_dim])


    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """
        Predict next latent state.
        obs: [B, obs_dim]
        act: [B, act_dim]
        returns: \hat{z}_{t+1} [B, latent_dim]
        """
        z = self.encoder(obs)
        z_next_pred = self.predictor(torch.cat([z, act], dim=-1))
        return z_next_pred

    @torch.no_grad()
    def get_target(self, next_obs: torch.Tensor) -> torch.Tensor:
        """
        Compute target latent state using target encoder.
        next_obs: [B, obs_dim]
        returns: z_{t+1} [B, latent_dim]
        """
        return self.target_encoder(next_obs)

    @torch.no_grad()
    def update_target(self):
        """
        Update target encoder parameters using EMA.
        """
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * self.ema_decay + param_q.data * (1.0 - self.ema_decay)

    def compute_loss(self, obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        """
        Compute JEPA loss (MSE in latent space).
        """
        z_next_pred = self.forward(obs, act)
        with torch.no_grad():
            z_next_target = self.get_target(next_obs)
        
        # Standard MSE loss in latent space
        loss = F.mse_loss(z_next_pred, z_next_target)
        
        return loss
