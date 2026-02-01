from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PolicyEncoderConfig:
    encoder_type: str = "transformer"  # "transformer" or "mlp"
    n_embd: int = 32  # embedding dimension of transformer, ignored for MLP encoder
    n_head: int = 4  # number of attention heads, ignored for MLP encoder
    n_enc_layer: int = 2  # number of encoder layers, ignored for MLP encoder
    dropout: float = 0.1  # dropout rate for regularization, ignored for MLP encoder
    embd_hidden_dims: List[int] = field(default_factory=lambda: [256])  # token embedding hidden dimensions
    embd_activation: str = "relu"  # activation function for token embedding
    obj_n_embd: int = 4  # embedding dimension of per-objective representation
    obj_hidden_dims: List[int] = field(
        default_factory=lambda: []
    )  # hidden dimensions for per-objective representations
    obj_activation: str = "relu"  # activation function for per-objective representations


@dataclass
class PolicyDecoderConfig:
    state_repr_dim: int = 256
    state_hidden_dims: List[int] = field(default_factory=lambda: [256])  # process state
    policy_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])  # policy repr + state repr -> action
    noise_std_type: str = "log"
    init_noise_std: float = 1.0
    min_log_noise_std: float = -20.0
    max_log_noise_std: float = 2.0
    activation: str = "relu"
    deterministic: bool = False


@dataclass
class VAEConfig:
    context_length: int = 32
    obs_norm: bool = True  # normalize observations
    value_norm: bool = True  # normalize values
    encoder: PolicyEncoderConfig = field(default_factory=PolicyEncoderConfig)
    decoder: PolicyDecoderConfig = field(default_factory=PolicyDecoderConfig)
    value_head_hidden_dims: List[int] = field(default_factory=lambda: [])  # default linear value head


@dataclass
class TrainerConfig:
    # Loss coefs
    coef_value: float = 1.0  # coefficient for the value loss (for value head only)
    coef_contrastive: float = 1.0  # coefficient for the contrastive loss (for VAE)
    coef_decoder: float = 1.0  # coefficient for the decoder loss (for VAE)
    coef_ortho: float = 5.0  # coefficient for the orthogonality loss (for projection head g())
    coef_kl_start: float = 0.0  # start of the KL annealing (for VAE)
    coef_kl_end: float = 0.05  # end of the KL annealing (for VAE)

    # Contrastive Loss
    rnc_temperature: float = 0.5  # temperature for the contrastive loss
    rnc_label_diff: str = "l1"  # distance type for the label difference
    rnc_feature_sim: str = "l2"  # similarity type for the feature similarity

    # Optimizer
    learning_rate: float = 1e-3  # learning rate for the optimizer

    # Training Loop
    vae_epochs: int = 100  # number of epochs to train the VAE
    vae_batch_size: int = 64  # batch size (number of distinct trajectories)
    vae_num_context: int = 2  # number of context per distinct trajectory per batch
    vae_num_query: int = 256  # number of query states per distinct trajectory per batch
    value_epochs: int = 100  # number of epochs to train the value head
    value_batch_size: int = 256  # batch size (number of distinct trajectories)
    value_num_context: int = 16  # number of context per distinct trajectory per batch

    # Flags
    value_only: bool = False  # only train the value head
    sample_trajectory: bool = False  # if True, use temporarily related state-actoin pairs as context for the VAE
    contrastive_loss_type: str = "rnc"


@dataclass
class DatasetConfig:
    skip_before: int = 30  # if use self collected data, number of initial policy checkpoints to skip
    skip_interval: int = (
        5  # if use self collected data, skip policy checkpoints to avoid too similar policies; 1 = not skipping
    )
    max_checkpoints: Optional[int] = None  # maximum number of checkpoints keep if using self collected data


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)


def get_config(task_id: str, default_configs: Config = None) -> Config:
    """Get configuration for a specific environment. Returns default config if env not found."""
    if default_configs is not None and task_id in default_configs:
        return default_configs[task_id]
    else:
        return Config(dataset=DatasetConfig(), trainer=TrainerConfig(), vae=VAEConfig())
