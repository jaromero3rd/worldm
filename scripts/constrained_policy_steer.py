import argparse
import os
import pickle

import numpy as np
import torch

from spr.runner import PolicyVAERunner
from spr.utils import get_config, load_representations, set_seed


class Constraint:
    def __init__(self, idx: int, threshold: float):
        self.idx, self.threshold = idx, threshold

    def violation(self, val: float) -> float:
        return max(self.threshold - val, 0.0)


def tangent_projected_step(z, train_data, grad, k=50):
    # 1. Find local neighbors
    dists = torch.cdist(z, train_data)
    _, idx = torch.topk(dists, k, largest=False)
    neighbors = train_data[idx[0]]

    # 2. Local PCA via SVD
    centered = neighbors - neighbors.mean(0)
    _, S, V = torch.linalg.svd(centered, full_matrices=False)  # V: [D, D] or [k, D]

    # 3. Select basis explaining >95% variance
    var_exp = torch.cumsum(S**2, dim=0) / torch.sum(S**2)
    n_comp = torch.where(var_exp >= 0.95)[0][0].item() + 1
    basis = V[:n_comp]  # [n_comp, D]

    # 4. Project gradient
    proj_grad = (grad @ basis.T) @ basis
    return proj_grad


def optimize_latent(
    runner,
    target,
    constraints,
    training_data,
    init_z=None,
    restarts=1,
    lr=0.03,
    lambda_lr=0.01,
    target_tol=100.0,
    max_steps=2000,
    device="cuda:0",
):
    vae = runner.vae
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    target_norm = vae.value_normalizer(torch.tensor([[target] + [0.0] * (runner.n_objs - 1)], device=device))[0, 0]

    best_overall_z, best_overall_viol, best_overall_history = None, float("inf"), []

    for r in range(restarts):
        z = init_z.clone() if r == 0 else init_z.clone() + 0.1 * torch.randn_like(init_z)
        z.requires_grad = True

        lambdas = torch.zeros(len(constraints), device=device)
        current_history = []
        best_z, best_viol = z.detach().clone(), float("inf")

        for step in range(max_steps):
            current_history.append(z.detach().cpu())
            val_norm = vae.get_value(z)
            val_denorm = vae.value_normalizer.inverse(val_norm)[0]

            target_loss = torch.nn.functional.mse_loss(val_norm[0, 0], target_norm)
            c_loss = 0.0

            # Target violation (clipped by tolerance)
            target_val = val_denorm[0].item()
            target_viol = abs(target_val - target)
            total_viol = max(0, target_viol - target_tol)

            for i, c in enumerate(constraints):
                v = c.violation(val_denorm[c.idx].item())
                total_viol += v
                if v > 0:
                    c_loss += (
                        lambdas[i] * (c.threshold - val_denorm[c.idx]) / (vae.value_normalizer._std[0, c.idx] + 1e-6)
                    )

            loss = target_loss + c_loss
            if z.grad is not None:
                z.grad.zero_()
            loss.backward()

            with torch.no_grad():
                z.grad.data = tangent_projected_step(z, training_data, z.grad.data)
                z.data -= lr * z.grad.data
                for i, c in enumerate(constraints):
                    lambdas[i] += (
                        lambda_lr * c.violation(val_denorm[c.idx].item()) / (vae.value_normalizer._std[0, c.idx] + 1e-6)
                    )

            if total_viol < best_viol:
                best_viol, best_z = total_viol, z.detach().clone()

            if total_viol == 0:
                print(f"[Restart {r}] Feasible solution found at step {step+1}!")
                return best_z, current_history, True

            if step % 50 == 0:
                print(
                    f"[Restart {r}] step {step+1}/{max_steps} | Predicted={val_denorm.detach().cpu().numpy()} | "
                    f"total_v={total_viol:.3f} target_v={target_viol:.3f} | z_norm={z.norm().item():.3f}"
                )

        if best_viol < best_overall_viol:
            best_overall_viol, best_overall_z, best_overall_history = best_viol, best_z, current_history

    return best_overall_z, best_overall_history, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_id", default="mo-halfcheetah-v5")
    parser.add_argument("-tar", "--target", type=float, required=True)
    parser.add_argument("-con", "--constraints", nargs="+", type=float, default=[])
    parser.add_argument("-i", "--init-idx", type=int, default=300)
    parser.add_argument("-r", "--restarts", type=int, default=3)
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.01)
    parser.add_argument("-s", "--max_steps", type=int, default=2000)
    parser.add_argument("--target-tol", type=float, default=100.0)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    log_dir = os.path.join("datasets", args.task_id, "policy_vae")
    runner = PolicyVAERunner(args.task_id, get_config(args.task_id), "cuda:0")
    runner.load(log_dir)

    mus, log_stds, _, _ = load_representations(log_dir)
    training_data = torch.tensor(mus, device="cuda:0", dtype=torch.float32)

    mu = training_data[args.init_idx].unsqueeze(0)
    log_std = torch.tensor(log_stds[args.init_idx], device="cuda:0", dtype=torch.float32)
    init_z = mu + torch.exp(log_std) * torch.randn_like(mu)

    constraints = [Constraint(i + 1, v) for i, v in enumerate(args.constraints)]
    z, history, found_feasible = optimize_latent(
        runner,
        args.target,
        constraints,
        training_data,
        init_z=init_z,
        restarts=args.restarts,
        lr=args.learning_rate,
        target_tol=args.target_tol,
        max_steps=args.max_steps,
    )

    # Evaluate
    from spr.envs import MOGymEnv

    with torch.no_grad():
        pred = runner.vae.value_normalizer.inverse(runner.vae.get_value(z)).cpu().numpy()[0]

    if not found_feasible:
        print("\nWarning: No fully feasible solution found. Evaluating best candidate.")

    env = MOGymEnv(name=args.task_id, draw=False, device="cuda:0")
    mean_ret, std_ret = runner.evaluate(env, z, args.episodes)
    env.close()

    print(f"\nOptimization Finished.")
    print(f"Predicted Outcome: {pred}")
    print(f"Evaluation Results (Mean): {mean_ret}")
    print(f"Evaluation Results (Std):  {std_ret}")
    print(f"Target Error (Actual): {abs(mean_ret[0] - args.target):.4f}")

    # Save results
    out_dir = os.path.join(log_dir, "steering_results")
    os.makedirs(out_dir, exist_ok=True)
    init_str = f"init{args.init_idx}" if args.init_idx is not None else "rand"
    c_str = "_".join([f"{c:.0f}" for c in args.constraints]) if args.constraints else "none"
    res_path = os.path.join(out_dir, f"{init_str}_t{args.target:.0f}_c{c_str}_seed{args.seed}.pkl")

    with open(res_path, "wb") as f:
        pickle.dump(
            {
                "task_id": args.task_id,
                "z_final": z.cpu().numpy(),
                "z_history": torch.stack([h.flatten() for h in history]).numpy(),
                "target": args.target,
                "constraints": args.constraints,
                "init_idx": args.init_idx,
                "found_feasible": found_feasible,
                "predicted_outcome": pred,
                "eval_mean_return": mean_ret,
                "eval_std_return": std_ret,
            },
            f,
        )
    print(f"Saved results to {res_path}")


if __name__ == "__main__":
    main()
