import argparse
import os
from dataclasses import asdict

from spr.default_configs import ENV_CONFIGS
from spr.runner import PolicyVAERunner
from spr.utils import get_config, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_id", type=str, default="mo-halfcheetah-v5")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)

    # Optional overrides
    parser.add_argument("-v", "--value_only", action="store_true", default=None) # only train the value head
    parser.add_argument("-traj", "--sample_trajectory", action="store_true", default=None) # consecutive samples as context, benchmark only

    args = parser.parse_args()
    set_seed(args.seed)

    # Manually determine dataset type based on task ID
    dataset_type = "momujoco" if args.task_id.startswith("mo-") else None
    log_dir = os.path.join("datasets", args.task_id)
    if dataset_type == "momujoco" and not os.path.exists(log_dir):  # minari dataset is not stored in a log directory
        return print(f"Log dir not found: {log_dir}")

    # Load default config for environment
    config = get_config(args.task_id, ENV_CONFIGS)

    # Apply overrides
    if args.value_only is not None:
        config.trainer.value_only = args.value_only
    if args.sample_trajectory is not None:
        config.trainer.sample_trajectory = args.sample_trajectory

    runner = PolicyVAERunner(args.task_id, config, args.device, log_dir=log_dir)
    runner.load_dataset(dataset_type, args.task_id, asdict(config.dataset))
    runner.train(value_only=config.trainer.value_only)


if __name__ == "__main__":
    main()
