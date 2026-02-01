"""
Collect rollout dataset from trained agent. The dataset following D4RL format with some extra information.
"""

import argparse
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
import yaml
from tqdm import tqdm

from spr.envs import MOGymEnv
from spr.onpolicy.algorithms import *
from spr.utils import set_seed


def load_actor(agent, checkpoint_path: str, device: torch.device):
    print(f"Loading checkpoint from: {checkpoint_path}")
    loaded_dict = torch.load(checkpoint_path, map_location=device)
    agent.actor.load_state_dict(loaded_dict["actor_state_dict"])
    agent.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
    return agent


def collect_dataset(
    env, agent, min_num_steps: int, device: torch.device, max_episode_length: int = 1000, deterministic: bool = True
):
    def policy(obs):
        normed_obs = agent.obs_normalizer(obs)
        if deterministic:
            raw_actions = agent.actor.act_inference(normed_obs)
        else:
            raw_actions = agent.actor.act(normed_obs)
        return agent.postprocess_actions(raw_actions)

    # Pre-allocate buffers for all environments
    num_envs = env.num_envs
    obs_buffer = np.zeros((num_envs, max_episode_length, env.num_obs), dtype=np.float32)
    actions_buffer = np.zeros((num_envs, max_episode_length, env.num_actions), dtype=np.float32)
    rewards_buffer = np.zeros((num_envs, max_episode_length, env.num_rewards), dtype=np.float32)
    next_obs_buffer = np.zeros((num_envs, max_episode_length, env.num_obs), dtype=np.float32)
    terminals_buffer = np.zeros((num_envs, max_episode_length), dtype=bool)
    timeouts_buffer = np.zeros((num_envs, max_episode_length), dtype=bool)

    # Buffer
    buffer_pos = np.zeros(num_envs, dtype=int)
    completed_trajectories = []
    total_steps_collected = 0
    pbar = tqdm(total=min_num_steps, desc="Collecting steps")

    # Rollout
    obs, _ = env.reset()
    while total_steps_collected < min_num_steps:
        with torch.no_grad():
            actions = policy(obs)
            next_obs, rewards, dones, infos = env.step(actions)

            # Store in buffers
            env_idx = np.arange(num_envs)
            obs_buffer[env_idx, buffer_pos] = obs[env_idx].cpu().numpy()
            actions_buffer[env_idx, buffer_pos] = actions[env_idx].cpu().numpy()
            rewards_buffer[env_idx, buffer_pos] = rewards[env_idx].cpu().numpy()
            next_obs_buffer[env_idx, buffer_pos] = next_obs[env_idx].cpu().numpy()

            # Handle timeouts
            np_dones = dones.cpu().numpy()
            np_timeouts = np.zeros_like(np_dones)
            if "time_outs" in infos:
                np_timeouts = infos["time_outs"].cpu().numpy()
            is_timeout = np_timeouts[env_idx] > 0
            is_done = np_dones[env_idx] > 0
            is_terminal = is_done & (~is_timeout)
            terminals_buffer[env_idx, buffer_pos] = is_terminal
            timeouts_buffer[env_idx, buffer_pos] = is_timeout

            # Final observation
            done_indices = env_idx[is_done]
            final_obs = infos.get("infos", {}).get("final_obs", None)  # refer to gymnasium sync_vector_env.py
            if len(done_indices) > 0 and final_obs is not None:
                final_obs_list = [final_obs[i] for i in done_indices]  # it was a list of np arrays
                next_obs_buffer[done_indices, buffer_pos[done_indices]] = np.stack(final_obs_list)

            # Update buffer position
            buffer_pos[env_idx] += 1

            # Process completed episodes
            done_indices = np.where(np_dones)[0]
            for env_idx in done_indices:
                if total_steps_collected >= min_num_steps:
                    break

                length = buffer_pos[env_idx]
                ep_data = {
                    "observations": obs_buffer[env_idx, :length].copy(),
                    "actions": actions_buffer[env_idx, :length].copy(),
                    "next_observations": next_obs_buffer[env_idx, :length].copy(),
                    "rewards": rewards_buffer[env_idx, :length].copy(),
                    "terminals": terminals_buffer[env_idx, :length].copy(),
                    "timeouts": timeouts_buffer[env_idx, :length].copy(),
                    "traj_returns": rewards_buffer[env_idx, :length].sum(axis=0),
                }

                completed_trajectories.append(ep_data)
                total_steps_collected += length
                pbar.update(length)

                # Reset buffer for this env
                buffer_pos[env_idx] = 0

            # Update observation
            obs = next_obs
    pbar.close()

    # Merge all completed episodes into D4RL format (flat arrays)
    final_dataset = defaultdict(list)
    keys = [k for k in completed_trajectories[0].keys() if k != "traj_returns"]
    for key in keys:
        final_dataset[key] = np.concatenate([t[key] for t in completed_trajectories], axis=0)
    final_dataset["trajectories"] = completed_trajectories
    return dict(final_dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, default="mo-halfcheetah-v5")
    parser.add_argument("-n", "--num_steps", type=int, default=8000)
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument("-l", "--log_dir", type=str, default="datasets")
    parser.add_argument("-det", "--deterministic", action="store_true", default=False)
    args = parser.parse_args()

    device = (
        torch.device("cuda:0")
        if args.device in ["cuda:0", "cuda", "gpu"] and torch.cuda.is_available()
        else torch.device("cpu")
    )
    log_dir = os.path.join(args.log_dir, args.env)
    if not os.path.exists(log_dir):
        print(f"Log directory not found: {log_dir}")
        return

    # load config
    with open(os.path.join(log_dir, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    set_seed(config.get("seed", None))
    config["env_cfg"]["num_envs"] = 8

    # init env and agent
    env = MOGymEnv(name=args.env, draw=False, device=device, seed=config.get("seed"), **config["env_cfg"])
    agent = eval(config["alg_cfg"].pop("class", "PPO"))(env, alg_cfg=config["alg_cfg"], device=device)

    # sort checkpoints
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir) if f.startswith("model_") and f.endswith(".pt") and f[6:-3].isdigit()
    ]
    checkpoint_files.sort(key=lambda x: int(x[6:-3]))

    os.makedirs(os.path.join(log_dir, "datasets"), exist_ok=True)
    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        load_actor(agent, checkpoint_path, device)
        agent.eval_mode()
        dataset = collect_dataset(env, agent, args.num_steps, device, deterministic=args.deterministic)

        # Print stats
        traj_returns = np.array([t["traj_returns"] for t in dataset["trajectories"]])
        num_episodes = len(dataset["trajectories"])
        print(f"Trajectory returns: {traj_returns.mean(axis=0)} std: {traj_returns.std(axis=0)}")
        print(f"Collected {len(dataset['observations'])} transitions over {num_episodes} episodes.\n")
        output_file = os.path.join(log_dir, "datasets", f"rollout_{checkpoint_file.split('.')[0]}.pkl")
        with open(output_file, "wb") as f:
            pickle.dump(dataset, f)

    env.close()


if __name__ == "__main__":
    main()
