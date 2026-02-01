import os
import pickle
from typing import List, Optional

import torch
import yaml

from .dataset import TrajectoryDataset


class MoMujocoTrajectoryDataset(TrajectoryDataset):
    """
    Dataset class to load and manage self-collected multi-objective mujoco trajectory data.
    """

    def load(self, task_id: str, **kwargs):
        skip_before = kwargs.get("skip_before", 0)
        skip_interval = kwargs.get("skip_interval", 5)
        max_checkpoints = kwargs.get("max_checkpoints", None)

        # load datasets from the datasets directory
        log_dir = os.path.join("datasets", task_id)
        datasets_dir = os.path.join(log_dir, "datasets")
        if not os.path.exists(datasets_dir):
            raise FileNotFoundError(f"Datasets directory not found: {datasets_dir}")
        files = os.listdir(datasets_dir)
        files.sort(key=lambda x: int(x[14:-4]))
        interval = yaml.safe_load(open(os.path.join(log_dir, "config.yaml"), "r"))["train_cfg"][
            "num_learning_iterations"
        ]
        files = self._filter_datasets(files, interval, skip_before, skip_interval, max_checkpoints)
        print(f"Loading {len(files)} datasets from {datasets_dir}")

        for f in files:
            # load trajectories
            with open(os.path.join(datasets_dir, f), "rb") as df:
                data = pickle.load(df)
            trajs = data.get("trajectories", [])
            if not trajs:
                print(f"Warning: No trajectories found in {f}")
                continue

            for traj in trajs:
                obs = torch.from_numpy(traj["observations"]).float().to(self.device)
                if len(obs) < self.context_length:  # skip short trajectories
                    continue

                # if task_id == "mo-halfcheetah-v5":
                #     obs = torch.cat([obs[:, :8], obs[:, 9:]], dim=1)

                # Convert to tensors and add to dataset
                self.trajectories.append(
                    {
                        "observations": obs,  # [T, obs_dim]
                        "actions": torch.from_numpy(traj["actions"]).float().to(self.device),  # [T, act_dim]
                        "returns": torch.from_numpy(traj["traj_returns"]).float().to(self.device),  # [n_objs]
                        "model_id": int(f[14:-4]),
                    }
                )

        # update dimensions
        self.state_dim = self.trajectories[0]["observations"].shape[1]
        self.action_dim = self.trajectories[0]["actions"].shape[1]
        self.n_objs = self.trajectories[0]["returns"].shape[0]

        print(f"Data loaded. Found {len(self.trajectories)} trajectories across {len(files)} policies.")

    @staticmethod
    def _filter_datasets(
        datasets: List[str],
        train_interval: int,
        skip_before: int = 0,
        skip_interval: int = 5,
        max_checkpoints: Optional[int] = None,
    ) -> List[str]:
        """
        Filter list of datasets based on their indices. Skip_before to ignore bad initial policies.
        Skip_interval for skipping too similar policies.
        """
        kept_datasets = []
        n = 0
        while True:
            start = n * train_interval + skip_before
            end = (n + 1) * train_interval
            if start > (max_checkpoints if max_checkpoints is not None else len(datasets)):
                break
            kept_datasets.extend(datasets[start:end:skip_interval])
            n += 1
        return kept_datasets
