import argparse
import os
import pickle
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from spr.envs import MOGymEnv
from spr.runner import PolicyVAERunner
from spr.utils import get_config, load_representations, set_seed


def plot_results(results: List[Dict], output_dir: str, env_name: str):
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.figure(figsize=(12, 8), dpi=100)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    markers = ["o", "s", "^", "D", "v", "<", ">"]

    pe_data = {}
    oa_data = {}
    for r in results:
        m = r["model"]
        pe_data.setdefault(m, []).append(r["imitation_return"])
        oa_data.setdefault(m, []).append(r["original_return"])
    models = sorted(list(pe_data.keys()))

    for i in range(results[0]["imitation_return"].shape[0]):
        # Calculate mean for each model for objective i
        pe_means = []
        oa_means = []
        for m in models:
            pe_rews = [x[i] for x in pe_data[m]]
            pe_means.append(np.mean(pe_rews))
            oa_rews = [x[i] for x in oa_data[m]]
            oa_means.append(np.mean(oa_rews))
        plt.plot(
            models,
            pe_means,
            color=colors[i % len(colors)],
            linestyle="-",
            marker=markers[i % len(markers)],
            linewidth=2.5,
            markersize=8,
            alpha=0.9,
            label=f"Decoder Obj {i+1}",
        )
        plt.plot(
            models,
            oa_means,
            color=colors[i % len(colors)],
            linestyle="--",
            marker=markers[i % len(markers)],
            linewidth=2,
            markersize=8,
            alpha=0.4,
            label=f"Original Obj {i+1}",
        )

    plt.xlabel("Model Number", fontsize=14)
    plt.ylabel("Episodic Return", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    avg_diff = np.mean([r["imitation_return"] - r["original_return"] for r in results], axis=0)
    diff_str = ", ".join([f"{d:.4f}" for d in avg_diff])
    plt.title(
        f"Policy Decoder vs Original Performance ({env_name})\nAvg Return Difference (Decoder - Original): [{diff_str}]",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.legend(fontsize=12, frameon=True, shadow=True, fancybox=True)
    plt.grid(True, linestyle="--", alpha=0.3)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "decoder_performance.png"), dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_dir}/decoder_performance.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_id", type=str, default="mo-halfcheetah-v5")
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument("-n", "--num_repr", type=int, default=10000, help="Number of random representations to test")
    parser.add_argument("-ep", "--episodes", type=int, default=3, help="Number of episodes per sampled representation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mean", action="store_true", default=False, help="Use mean representation instead of sampling"
    )
    parser.add_argument("--plot", action="store_true", default=False)
    args = parser.parse_args()
    set_seed(args.seed)

    # Load system
    log_dir = os.path.join("datasets", args.task_id, "policy_vae")

    runner = PolicyVAERunner(args.task_id, get_config(args.task_id), args.device)
    runner.load(log_dir)
    runner.vae.eval()

    # Load representations
    try:
        mus, log_stds, returns, model_idxs = load_representations(log_dir)
    except:
        raise Exception(f"Could not load representations from {log_dir}")

    # Setup Environment
    env = MOGymEnv(name=args.task_id, draw=False, device=args.device, num_envs=3)

    # Randomly select representations and sort by model index for cleaner visualization
    n_total = len(mus)
    sorted_indices = sorted(np.random.choice(n_total, min(args.num_repr, n_total), replace=False))
    print(f"Testing {len(sorted_indices)} / {n_total} policy representations.")
    results = []
    for idx in sorted_indices:
        mu = mus[idx]
        log_std = log_stds[idx]
        original_return = returns[idx]
        model_idx = model_idxs[idx]

        # Prepare for inference
        mu_t = torch.from_numpy(mu).to(args.device)
        log_std_t = torch.from_numpy(log_std).to(args.device)
        if mu_t.dim() == 1:
            mu_t = mu_t.unsqueeze(0)  # [1, n_embd]
            log_std_t = log_std_t.unsqueeze(0)  # [1, n_embd]
        std = torch.exp(log_std_t)
        z = mu_t + (not args.mean) * std * torch.randn_like(mu_t)

        # evaluate
        mean_ret, std_ret = runner.evaluate(env, z, args.episodes)
        results.append(
            {
                "model": model_idx,
                "trajectory": idx,
                "original_return": original_return,
                "imitation_return": mean_ret,
                "imitation_return_std": std_ret,
            }
        )
        print(f"  Model {model_idx} (trajectory idx {idx}):")
        print(f"    Original Return: {original_return}")
        print(f"    Imitated Return (Avg {args.episodes} eps): {mean_ret} (std: {std_ret})")
        print(f"    Difference: {mean_ret - original_return}")
    env.close()

    # Save results
    save_path = os.path.join(log_dir, "decoder_results.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {save_path}")

    print(
        f"Average Difference (Decoder - Original): {np.mean([r['imitation_return'] - r['original_return'] for r in results], axis=0)}"
    )
    if args.plot:
        plot_results(results, log_dir, args.task_id)


if __name__ == "__main__":
    main()
