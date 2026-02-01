"""Train policy with diverse behavior by multi-objective weight search."""

import argparse
import os
import shutil

import torch
import yaml

from spr.envs import MOGymEnv
from spr.onpolicy.algorithms import *
from spr.onpolicy.runners import Runner
from spr.utils import WANDB_API_KEY, WANDB_ENTITY, load_hyperparameters, set_seed

os.environ["WANDB_USERNAME"] = WANDB_ENTITY
os.environ["WANDB_API_KEY"] = WANDB_API_KEY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algo", type=str, default="PPO")
    parser.add_argument("-e", "--env", type=str, default="mo-halfcheetah-v5")
    parser.add_argument("-l", "--log_dir", type=str, default="datasets")
    parser.add_argument("-r", "--render", action="store_true", default=False)
    parser.add_argument("-n", "--num_weight_vectors", type=int, default=10, help="Number of weight vectors to try")
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config file")
    args = parser.parse_args()

    # set device
    device = torch.device("cuda:0") if args.device in ["cuda:0", "cuda", "gpu"] else torch.device("cpu")

    # create experiment directory
    log_dir = os.path.join(args.log_dir, args.env)
    if os.path.exists(log_dir):
        print(f"Log directory already exists: {log_dir}")
        # shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Load algorithm-specific hyperparameters
    config = load_hyperparameters(args.algo, args.env, args.config)

    # Override seed if SEED_OVERRIDE environment variable is set
    if "SEED_OVERRIDE" in os.environ:
        override_seed = int(os.environ["SEED_OVERRIDE"])
        config["seed"] = override_seed
        print(f"Overriding seed with {override_seed}")

    # set seed for reproducibility
    set_seed(config.get("seed", None))

    # save a copy of the config to the log directory
    config_save_path = os.path.join(log_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(config, f)

    # create multi-objective training and evaluation environments
    env = MOGymEnv(
        name=args.env,
        draw=args.render,
        draw_cb=lambda ep: ep % 10 == 0,
        draw_directory=log_dir,
        device=device,
        seed=config.get("seed", None),
        **config["env_cfg"],
    )
    eval_env = MOGymEnv(
        name=args.env,
        draw=False,
        device=device,
        num_envs=config["train_cfg"].get("eval_num_envs", 8),
    )

    # learning algorithm, prioritize algorithm from argparse over algorithm from config
    if args.algo is None:
        agent = eval(config["alg_cfg"].pop("class", "PPO"))(env, alg_cfg=config["alg_cfg"], device=device)
    else:
        agent = eval(args.algo)(env, alg_cfg=config["alg_cfg"], device=device)

    # learn with multi-objective weight search
    runner = Runner(env, agent, train_cfg=config["train_cfg"], log_dir=log_dir, eval_env=eval_env, device=device)
    results = runner.learn_mo(num_weight_vectors=args.num_weight_vectors)

    # print the results for analysis
    print(f"\n=== MULTI-OBJECTIVE TRAINING COMPLETED ===")
    print(f"Explored {len(results)} weight vectors")
    print(f"Results saved to: {log_dir}")
    print(f"========================\n")


if __name__ == "__main__":
    main()
