"""
Gymnasium environment wrapper.
Modified from: https://github.com/leggedrobotics/rsl_rl/blob/algorithms/rsl_rl/env/gym_env.py
"""

from datetime import datetime
from typing import Any, Dict, Tuple

import gymnasium as gym
import torch

from spr.envs.vec_env import VecEnv


class GymEnv(VecEnv):
    """A vectorized environment wrapper for Gymnasium environments.

    This class wraps a single Gymnasium environment into a vectorized environment. It is assumed that the environment
    is a single agent environment. The environment is wrapped in a `gym.vector.SyncVectorEnv` environment, which
    allows for parallel execution of multiple environments.
    """

    def __init__(self, name, draw=False, draw_cb=None, draw_directory="videos/", seed=None, gym_kwargs={}, **kwargs):
        """
        Args:
            name: The name of the Gymnasium environment.
            draw: Whether to record videos of the environment.
            draw_cb: A callback function that is called after each episode. The callback function is passed the episode
                number and the path to the video file. The callback function should return `True` if the video should
                be recorded and `False` otherwise.
            draw_directory: The directory in which to store the videos.
            seed: The seed to use for the environment. Used when initial reset
            gym_kwargs: Keyword arguments that are passed to the Gymnasium environment.
            **kwargs: Keyword arguments that are passed to the `VecEnv` constructor.
        """
        self.env_cfg = kwargs
        self._gym_kwargs = gym_kwargs
        self._seed = seed

        env = gym.make(name, **self._gym_kwargs)

        assert isinstance(env.observation_space, gym.spaces.Box)
        assert len(env.observation_space.shape) == 1
        assert isinstance(env.action_space, gym.spaces.Box)
        assert len(env.action_space.shape) == 1

        super().__init__(env.observation_space.shape[0], **kwargs)

        self.name = name
        self.draw_directory = draw_directory

        self.num_actions = env.action_space.shape[0]
        self.max_action = env.action_space.high[0]
        self.action_space = env.action_space
        self.num_obs = env.observation_space.shape[0]
        self.num_rewards = 1

        self._gym_venv = gym.vector.SyncVectorEnv(
            [lambda: gym.make(self.name, **self._gym_kwargs) for _ in range(self.num_envs)],
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,  # SAME_STEP for rsl_rl implementation
        )
        self._gym_venv.action_space.seed(seed)  # for reproducibility, if startup steps are not zero

        # visualization
        self._draw = False
        self._draw_cb = draw_cb if draw_cb is not None else lambda *args: True
        self.draw = draw

        # initial reset of the environment
        self.reset()

    def get_observations(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self.obs_buf, self.extras

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Apply input action on the environment. Gym envs are on CPU, need to move to device."""
        obs, rew, term, trunc, infos = self._gym_venv.step(actions.float().cpu().numpy())  # make sure actions are float

        self.obs_buf = torch.from_numpy(obs).float().to(self.device)
        self.rew_buf = (
            torch.from_numpy(rew).float().to(self.device).view(self.num_envs, self.num_rewards)
        )  # follow [n, n_objs] format
        self.reset_buf = torch.from_numpy(term | trunc).float().to(self.device)
        self.extras = {"time_outs": torch.from_numpy(trunc).float().to(self.device), "infos": infos}
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        self.obs_buf = torch.from_numpy(self._gym_venv.reset(seed=self._seed)[0]).float().to(self.device)
        self.rew_buf = torch.zeros((self.num_envs, self.num_rewards), device=self.device).float()
        self.reset_buf = torch.zeros((self.num_envs,), device=self.device).float()
        self.extras = {"time_outs": torch.zeros((self.num_envs,), device=self.device).float(), "infos": {}}
        return self.obs_buf, self.extras

    def sample_actions(self) -> torch.Tensor:
        return torch.from_numpy(self._gym_venv.action_space.sample()).float().to(self.device)

    # ---------------------------- utils ----------------------------
    def close(self) -> None:
        self._gym_venv.close()

    def to(self, device: str) -> None:
        """This is not used"""
        self.device = device
        self.obs_buf = self.obs_buf.to(device)
        self.rew_buf = self.rew_buf.to(device)
        self.reset_buf = self.reset_buf.to(device)

    @property
    def draw(self) -> bool:
        return self._draw

    @draw.setter
    def draw(self, value: bool) -> None:
        """If render, initialize the first environment in render mode and wrap it with RecordVideo."""
        if value != self._draw:
            if value:
                env = gym.make(self.name, render_mode="rgb_array", **self._gym_kwargs)
                env = gym.wrappers.RecordVideo(
                    env,
                    f"{self.draw_directory}/videos/",
                    episode_trigger=lambda ep: self._draw_cb(ep) if ep > 0 else False,
                )
            else:
                env = gym.make(self.name, render_mode=None, **self._gym_kwargs)

            self._gym_venv.envs[0] = env
            self._draw = value
