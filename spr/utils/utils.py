from __future__ import annotations

import os
import pathlib
import pickle
import random
import sys
import time
from typing import Optional, Tuple

import git
import numpy as np
import torch
import yaml

# ----------------------------------------------------------- Helper Functions -----------------------------------------------------------
def load_hyperparameters(algo: str, env_id: str, config_path: str = None) -> dict:
    """Load hyperparameters for a specific algorithm and environment."""
    if config_path is None:  # use default config path
        config_path = os.path.join("spr", "onpolicy", "hyperparams", f"{algo.lower()}.yaml")
    with open(config_path, "r") as f:
        hyperparams = yaml.safe_load(f)

    # Get environment specific parameters
    if env_id in hyperparams:
        params = hyperparams[env_id]
    else:
        print(f"Environment {env_id} not found in {config_path}. Using default parameters.")
        params = hyperparams["default"]
    return params


def load_representations(log_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load policy representations from a directory. Each trajectory is a separate policy.

    Args:
        log_dir: Path to the directory containing 'policy_representations.pkl'

    Returns:
        mus: Mean of policy representations [N, n_embd]
        log_stds: Log std of policy representations [N, n_embd]
        returns: Returns of the policies [N, n_objs]
        model_idxs: Model indices [N]
    """
    path = os.path.join(log_dir, "policy_representations.pkl")
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded {len(data)} representations from {path}")
    except:
        raise Exception(f"Could not load representations from {path}")

    # data is list of (mu, log_std, returns, model_idx)
    # Unpack
    mus = np.array([x[0] for x in data])
    log_stds = np.array([x[1] for x in data])
    returns = np.array([x[2] for x in data])
    model_idxs = np.array([x[3] for x in data])

    # Sort by model_idx
    sort_idx = np.argsort(model_idxs)
    return mus[sort_idx], log_stds[sort_idx], returns[sort_idx], model_idxs[sort_idx]


# ----------------------------------------------------------- Misc -----------------------------------------------------------
def store_code_state(logdir, repositories) -> list:
    git_log_dir = os.path.join(logdir, "git")
    os.makedirs(git_log_dir, exist_ok=True)
    file_paths = []
    for repository_file_path in repositories:
        try:
            repo = git.Repo(repository_file_path, search_parent_directories=True)
        except Exception:
            print(f"Could not find git repository in {repository_file_path}. Skipping.")
            # skip if not a git repository
            continue
        # get the name of the repository
        repo_name = pathlib.Path(repo.working_dir).name
        t = repo.head.commit.tree
        diff_file_name = os.path.join(git_log_dir, f"{repo_name}.diff")
        # check if the diff file already exists
        if os.path.isfile(diff_file_name):
            continue
        # write the diff file
        print(f"Storing git diff for '{repo_name}' in: {diff_file_name}")
        with open(diff_file_name, "x", encoding="utf-8") as f:
            content = f"--- git status ---\n{repo.git.status()} \n\n\n--- git diff ---\n{repo.git.diff(t)}"
            f.write(content)
        # add the file path to the list of files to be uploaded
        file_paths.append(diff_file_name)
    return file_paths


# ----------------------------------------------------------- Neural Network Utils -----------------------------------------------------------
def resolve_nn_activation(act_name: str) -> torch.nn.Module:
    if act_name == "elu":
        return torch.nn.ELU()
    elif act_name == "selu":
        return torch.nn.SELU()
    elif act_name == "relu":
        return torch.nn.ReLU()
    elif act_name == "crelu":
        return torch.nn.CELU()
    elif act_name == "lrelu":
        return torch.nn.LeakyReLU()
    elif act_name == "tanh":
        return torch.nn.Tanh()
    elif act_name == "sigmoid":
        return torch.nn.Sigmoid()
    elif act_name == "identity":
        return torch.nn.Identity()
    else:
        raise ValueError(f"Invalid activation function '{act_name}'.")


def set_seed(seed: Optional[int] = None, deterministic: bool = False) -> int:
    """
    Set the seed for the random number generators

    .. note::

        In distributed runs, the worker/process seed will be incremented (counting from the defined value) according to its rank

    .. warning::

        Due to NumPy's legacy seeding constraint the seed must be between 0 and 2**32 - 1.
        Otherwise a NumPy exception (``ValueError: Seed must be between 0 and 2**32 - 1``) will be raised

    Modified packages:

    - random
    - numpy
    - torch (if available)

    Example::

        # fixed seed
        >>> from skrl.utils import set_seed
        >>> set_seed(42)
        [skrl:INFO] Seed: 42
        42

        # random seed
        >>> from skrl.utils import set_seed
        >>> set_seed()
        [skrl:INFO] Seed: 1776118066
        1776118066

        # enable deterministic. The following environment variables should be established:
        # - CUDA 10.1: CUDA_LAUNCH_BLOCKING=1
        # - CUDA 10.2 or later: CUBLAS_WORKSPACE_CONFIG=:16:8 or CUBLAS_WORKSPACE_CONFIG=:4096:8
        >>> from skrl.utils import set_seed
        >>> set_seed(42, deterministic=True)
        [skrl:INFO] Seed: 42
        [skrl:WARNING] PyTorch/cuDNN deterministic algorithms are enabled. This may affect performance
        42

    :param seed: The seed to set. Is None, a random seed will be generated (default: ``None``)
    :type seed: int, optional
    :param deterministic: Whether PyTorch is configured to use deterministic algorithms (default: ``False``).
                          The following environment variables should be established for CUDA 10.1 (``CUDA_LAUNCH_BLOCKING=1``)
                          and for CUDA 10.2 or later (``CUBLAS_WORKSPACE_CONFIG=:16:8`` or ``CUBLAS_WORKSPACE_CONFIG=:4096:8``).
                          See PyTorch `Reproducibility <https://pytorch.org/docs/stable/notes/randomness.html>`_ for details
    :type deterministic: bool, optional

    :return: Seed
    :rtype: int
    """
    # generate a random seed
    if seed is None:
        try:
            seed = int.from_bytes(os.urandom(4), byteorder=sys.byteorder)
        except NotImplementedError:
            seed = int(time.time() * 1000)
        seed %= 2**31  # NumPy's legacy seeding seed must be between 0 and 2**32 - 1
    seed = int(seed)

    # numpy
    random.seed(seed)
    np.random.seed(seed)

    # torch
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

            # On CUDA 10.1, set environment variable CUDA_LAUNCH_BLOCKING=1
            # On CUDA 10.2 or later, set environment variable CUBLAS_WORKSPACE_CONFIG=:16:8 or CUBLAS_WORKSPACE_CONFIG=:4096:8
    except ImportError:
        pass
    except Exception as e:
        pass
    return seed
