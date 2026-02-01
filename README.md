# Learning Policy Representation for Steerable Behavior Synthesis

This repository provides the official implementation of **"Learning Policy Representation for Steerable Behavior Synthesis"** ([arXiv:2601.22350](https://www.arxiv.org/abs/2601.22350)). We propose a method for learning a policy representation space characterized by a value-function-aligned geometry, enabling the synthesis of complex behaviors through latent space steering.

## Installation
We recommend using Conda and install the repo:
```bash
conda create -n [change_to_env_name] python=3.10.15
git clone git@github.com:beimingli0626/steerable-policy-reps.git
cd steerable-policy-reps
pip install -e .
```

To reproduce experiments in the paper, you also need to install our modified [MO-Gymnasium](https://github.com/beimingli0626/MO-Gymnasium).
```bash
git clone git@github.com:beimingli0626/MO-Gymnasium.git
cd MO-Gymnasium
pip install -e .
```


## Collect Offline Dataset with Diverse Behaviors
Learning policy representations requires datasets of trajectories with diverse behaviors. We provide utilities to train diverse policies by randomly sampling objective weights in multi-objective environments, such as [Multi-Objective Gymnasium](https://github.com/beimingli0626/MO-Gymnasium) Mujoco environments. Data are saved to the datasets/ directory by default.

Run the following command to train policies for 10 different sets of objective weights.

```bash
python3 scripts/dataset/train_online.py --algo PPO --env mo-halfcheetah-v5 --num_weight_vectors 10
```
> **📌 Note**: If logging training process with `wandb`, make a copy of `spr/utils/wandb_config.example.py` and rename to `spr/utils/wandb_config.py`, then set the `wandb` API key in the file.

To collect rollout trajectories for stored checkpoints and construct datasets, run
```bash
python3 scripts/dataset/collect_data.py --env mo-halfcheetah-v5
```

## Learning Policy Representations
We propose learning policy representations using a Variational Autoencoder (VAE) combined with RnC contrastive loss. After training, the representations for policies in the training dataset are saved to the log directory. Default hyperparameters are located in `spr/default_configs.py`.

To train the representation:
```bash
python3 scripts/train.py --task_id mo-halfcheetah-v5
```

To test the learned meta actor (the VAE decoder), use

```bash
python3 scripts/test_decoder.py --task_id mo-halfcheetah-v5
```


## Behavior Synthesis via Latent Space Steering
We implement a constrained optimization routine to steer trained policy representations toward specific multi-objective performance targets. By navigating the VAE latent space, it finds policies that satisfy specific reward constraints while hitting a target primary objective.

Run the optimization by specifying a task, the index of a training policy used as the starting point in latent space, a target for the primary objective, and thresholds for secondary objectives (constraints).

```bash
python constrained_policy_steer.py --task_id mo-halfcheetah-v5 --init-idx 300 --target 6000 --constraints -3000
```

## Repository Structure
```
steerable-policy-reps/
├── spr/                        # Core implementation
│   ├── modules/                # Neural network architectures
│   │   ├── policy_vae.py       # VAE for learning policy representations
│   │   ├── contrastive.py      # Contrastive learning (RnC) modules
│   │   ├── modules.py          # Base network architectures
│   │   └── normalizer.py       # Observation/reward normalization
│   ├── onpolicy/               # On-policy training framework (PPO)
│   │   ├── algorithms/         # PPO implementation
│   │   ├── runners/            # PPO execution logic
│   │   └── storage/            # Rollout storage buffers
│   ├── runner/                 # SPR-specific runners for training/evaluation
│   ├── dataset/                # Multi-objective dataset loaders (MoMuJoCo)
│   ├── envs/                   # Environment wrappers for Gymnasium/MO-Gymnasium
│   ├── utils/                  # Helper functions (visualization, wandb, config)
│   └── default_configs.py      # Task-specific default hyperparameters
├── scripts/                    # Top-level execution scripts
│   ├── train.py                # Main training entry point for SPR
│   ├── test_decoder.py         # Evaluation script for the VAE decoder
│   ├── constrained_policy_steer.py  # Constrained Behavior Synthesis
│   ├── dataset/                # Data collection and generation utilities
│   │   ├── train_online.py     # Training diverse policies for datasets
│   │   └── collect_data.py     # Rollout collection for stored checkpoints
├── datasets/                   # Local storage for offline trajectories
│   ├── mo-halfcheetah-v5/
│   │   ├── checkpoints/        # Training checkpoints obtained by dataset/train_online.py
│   │   └── datasets/           # Rollout trajectories collected by dataset/collect_data.py
```

## Pre-commit Hooks
This project supports pre-commit hooks to enforce code style. To install the hooks, run the following command:
```bash
pip install pre-commit
pre-commit install
```
