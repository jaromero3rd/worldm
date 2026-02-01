"""Common nn modules, Linear Layers, MLP and BERT-style Transformer"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from spr.utils import resolve_nn_activation


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        output_l2_norm: bool = False,
    ):
        super().__init__()
        self.output_l2_norm = output_l2_norm
        layers = []
        if len(hidden_dims) == 0:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            activation = resolve_nn_activation(activation)
            layers.append(nn.Linear(input_dim, hidden_dims[0]))
            layers.append(activation)
            for i in range(len(hidden_dims) - 1):
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                layers.append(activation)
            layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        if self.output_l2_norm:
            x = F.normalize(x, p=2, dim=-1)
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = nn.Dropout(config.dropout)
        self.attn_dropout_p = config.dropout

    def forward(self, x, attention_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: [B, T_seq, n_embd]
            attention_mask: [B, T_seq], where True indicates valid tokens / None
        Returns:
            y: [B, T_seq, n_embd]
        """
        B, T, C = x.size()  # sequence length, embedding dimensionality (n_embd)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            mask = mask & mask.transpose(-1, -2)  # [B, 1, T, T] - both query and key must be valid
        else:
            mask = None

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=False, dropout_p=self.attn_dropout_p
        )  # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        y = self.c_proj(y)
        y = self.dropout(y)  # Apply dropout after projection, before residual connection
        return y


class FullyConnectedBlock(nn.Module):
    """
    Fully Connected Block for Transformer
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.dropout(x)  # Apply dropout after activation
        x = self.c_proj(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.self_attn = SelfAttentionBlock(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FullyConnectedBlock(config)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.self_attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """BERT-style Transformer, multiple self-attention layers"""

    def __init__(self, config, token_dim: int):
        super().__init__()
        self.config = config

        # Embedding
        self.embedding = MLP(
            input_dim=token_dim,
            output_dim=config.n_embd,
            hidden_dims=config.embd_hidden_dims,
            activation=config.embd_activation,
        )

        # Transformer encoder
        self.encoder = nn.ModuleDict(
            dict(
                encoder_layers=nn.ModuleList([EncoderLayer(config) for _ in range(config.n_enc_layer)]),
                dropout=nn.Dropout(config.dropout),
                ln_f=nn.LayerNorm(config.n_embd),
                readout=nn.Linear(config.n_embd, config.n_embd),
            )
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer encoder layers

        Args:
            x: [B, T_seq, token_dim] - input tokens
            attention_mask: [B, T_seq] - attention mask where True indicates valid tokens

        Returns:
            [B, T_seq, n_embd] - encoded sequence
        """
        x = self.embedding(x)  # embed tokens
        x = self.encoder.dropout(x)  # input dropout
        for encoder_layer in self.encoder.encoder_layers:
            x = encoder_layer(x, attention_mask)  # encoder layers
        x = self.encoder.ln_f(x)  # final layer norm
        x = self.encoder.readout(x)  # readout layer
        return x
