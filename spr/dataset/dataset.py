from typing import Dict, List

import numpy as np
import torch


class TrajectoryDataset:
    """
    Base Dataset class to load and manage trajectory data.

    Format of the trajectories:
    {
        "observations": [T, obs_dim],
        "actions": [T, act_dim],
        "returns": [T, n_objs], # labels / attributes to align with latent representation
        "model_id": int,
    }
    """

    def __init__(self, context_length: int, device: str):
        self.context_length = context_length
        self.device = device
        self.trajectories: List[Dict[str, torch.Tensor]] = []
        self.state_dim = None
        self.action_dim = None
        self.n_objs = None

    def load(self, *args, **kwargs):
        raise NotImplementedError("Load method must be implemented by subclass.")

    def __len__(self):
        return len(self.trajectories)

    # ----------------------------------------  Batch Generator  --------------------------------------
    def batch_generator(
        self, batch_size: int, num_context: int, num_query: int, sample_trajectory: bool = False, shuffle: bool = True
    ):
        indices = np.arange(len(self))
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, len(self), batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            yield self._get_batch(batch_indices, num_context, num_query, sample_trajectory)

    def _get_batch(self, indices: np.ndarray, num_context: int, num_query: int, sample_trajectory: bool = False):
        """
        Samples a batch of data from specified trajectory indices.
        """
        ctx_s_list, ctx_a_list, ret_list, traj_idx_list, model_idx_list = [], [], [], [], []
        q_s_list, q_a_list, q_context_s_list, q_context_a_list = [], [], [], []
        for idx in indices:
            traj = self.trajectories[idx]

            # Context sets
            cs, ca = self._sample_context(idx, num_context, sample_trajectory)
            ctx_s_list.append(cs)  # [[num_context, T, obs_dim],...]
            ctx_a_list.append(ca)  # [[num_context, T, act_dim],...]
            ret_list.append(traj["returns"].expand(num_context, -1))  # [[num_context, n_objs],...]
            traj_idx_list.append(torch.full((num_context,), idx, device=self.device))  # [[num_context],...]
            model_idx_list.append(
                torch.full((num_context,), traj["model_id"], device=self.device)
            )  # [[num_context],...]

            # Query states and actions
            if num_query > 0:
                q_idx = torch.randint(0, traj["observations"].shape[0], (num_query,), device=self.device)
                query_cs, query_ca = self._sample_context(idx, num_query, sample_trajectory)
                q_s_list.append(traj["observations"][q_idx])  # [[num_query, obs_dim],...]
                q_a_list.append(traj["actions"][q_idx])  # [[num_query, act_dim],...]
                q_context_s_list.append(query_cs)  # [[num_query, T, obs_dim],...]
                q_context_a_list.append(query_ca)  # [[num_query, T, act_dim],...]
        return {
            "context_states": torch.cat(ctx_s_list, dim=0),  # [tot_num_context, T, obs_dim]
            "context_actions": torch.cat(ctx_a_list, dim=0),  # [tot_num_context, T, act_dim]
            "returns": torch.cat(ret_list, dim=0),  # [tot_num_context, n_objs]
            "traj_idx": torch.cat(traj_idx_list, dim=0),  # [tot_num_context]
            "model_idx": torch.cat(model_idx_list, dim=0),  # [tot_num_context]
            "query_states": torch.cat(q_s_list, dim=0) if num_query > 0 else None,
            "query_actions": torch.cat(q_a_list, dim=0) if num_query > 0 else None,
            "query_context_states": torch.cat(q_context_s_list, dim=0) if num_query > 0 else None,
            "query_context_actions": torch.cat(q_context_a_list, dim=0) if num_query > 0 else None,
        }

    def _sample_context(self, traj_idx: int, num_context: int, sample_trajectory: bool = False):
        """
        Sample context (states, actions) from a specific trajectory.
        Returns:
            ctx_states: [num_context, T, obs_dim]
            ctx_actions: [num_context, T, act_dim]
        """
        traj = self.trajectories[traj_idx]
        n_samples = traj["observations"].shape[0]
        if sample_trajectory:  # Sample a continuous segment of the trajectory
            max_start = n_samples - self.context_length
            starts = torch.randint(0, max_start + 1, (num_context,), device=self.device)  # [num_context]
            indices = starts.unsqueeze(1) + torch.arange(self.context_length, device=self.device)
        else:  # Uniformly sample transitions from the trajectory
            indices = torch.randint(0, n_samples, (num_context, self.context_length), device=self.device)
        return traj["observations"][indices], traj["actions"][indices]

    # ----------------------------------------  Utility Functions  --------------------------------------
    def get_all_observations(self) -> torch.Tensor:
        """Returns all observations concatenated (for obs normalization)."""
        all_obs = []
        for traj in self.trajectories:
            all_obs.append(traj["observations"])
        return torch.cat(all_obs, dim=0)  # [total_transitions, obs_dim]

    def get_all_returns(self) -> torch.Tensor:
        """Returns all returns concatenated (for value normalization)."""
        all_returns = []
        for traj in self.trajectories:
            all_returns.append(traj["returns"])
        return torch.stack(all_returns)  # [total_trajectories, n_objs]
