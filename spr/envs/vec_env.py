from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch


# minimal interface of the environment
class VecEnv(ABC):
    """Abstract class for vectorized environment."""

    num_envs: int
    num_obs: int
    num_actions: int
    max_episode_length: int
    obs_buf: torch.Tensor
    rew_buf: torch.Tensor
    reset_buf: torch.Tensor
    episode_length_buf: torch.Tensor  # current episode duration
    extras: dict
    device: torch.device

    def __init__(
        self,
        num_obs,
        device="cuda:0",
        num_envs=1,
        max_episode_length=-1,
        **kwargs,
    ):
        """
        Args:
            num_obs (int): Number of observations per environment.
            device (str): Device to use for the tensors.
            num_envs (int): Number of environments to run in parallel.
            max_episode_length (int): Maximum length of an episode. If -1, the episode length is not limited.
        """
        self.num_obs = num_obs
        self.num_envs = num_envs
        self.max_episode_length = max_episode_length
        self.device = device

    @abstractmethod
    def get_observations(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Return observations and extra information."""
        pass

    @abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Apply input action on the environment.

        Args:
            actions (torch.Tensor): Input actions to apply. Shape: (num_envs, num_actions)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
                A tuple containing the observations, rewards, dones and extra information (metrics).
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Reset all environment instances.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: Tuple containing the observations and extra information (metrics).
        """
        raise NotImplementedError
