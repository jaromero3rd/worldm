import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random

from spr.modules import JEPAWorldModel
from spr.dataset.momujoco import MoMujocoTrajectoryDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_id", type=str, default="mo-halfcheetah-v5")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4) # Upped learning rate
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--vis_interval", type=int, default=100)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--encoder_type", type=str, default="resnet18", choices=["resnet18", "mlp"])
    parser.add_argument("--predictor_type", type=str, default="mlp", choices=["resnet18", "mlp"])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 0. Prepare Save Directories
    model_tag = f"enc_{args.encoder_type}_pred_{args.predictor_type}"
    save_dir = os.path.join("datasets", args.task_id, "jepa_wm")
    vis_dir = os.path.join(save_dir, f"visuals_{model_tag}")
    os.makedirs(vis_dir, exist_ok=True)

    # 1. Initialize Datasets
    full_dataset = MoMujocoTrajectoryDataset(context_length=2, device=device)
    full_dataset.load(args.task_id, skip_before=0, skip_interval=10)
    
    # Manual Split
    all_trajs = full_dataset.trajectories
    random.shuffle(all_trajs)
    split_idx = int(len(all_trajs) * (1 - args.val_split))
    
    train_dataset = MoMujocoTrajectoryDataset(context_length=2, device=device)
    train_dataset.trajectories = all_trajs[:split_idx]
    train_dataset.state_dim = full_dataset.state_dim
    train_dataset.action_dim = full_dataset.action_dim
    train_dataset.n_objs = full_dataset.n_objs
    
    val_dataset = MoMujocoTrajectoryDataset(context_length=2, device=device)
    val_dataset.trajectories = all_trajs[split_idx:]
    val_dataset.state_dim = full_dataset.state_dim
    val_dataset.action_dim = full_dataset.action_dim
    val_dataset.n_objs = full_dataset.n_objs

    print(f"Train Trajectories: {len(train_dataset.trajectories)}")
    print(f"Val Trajectories: {len(val_dataset.trajectories)}")

    # 2. Initialize Model
    model = JEPAWorldModel(
        obs_dim=train_dataset.state_dim, 
        act_dim=train_dataset.action_dim, 
        latent_dim=args.latent_dim,
        ema_decay=args.ema_decay,
        encoder_type=args.encoder_type,
        predictor_type=args.predictor_type
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # 3. Training Loop
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss = 0
        gen = train_dataset.batch_generator(batch_size=args.batch_size, num_context=1, num_query=0, sample_trajectory=True)
        num_train_batches = len(train_dataset) // args.batch_size
        
        current_lr = optimizer.param_groups[0]['lr']
        pbar = tqdm(gen, desc=f"Epoch {epoch+1}/{args.epochs}", total=num_train_batches)
        for batch in pbar:
            obs_seq = batch["context_states"] # [B, 2, obs_dim]
            act_seq = batch["context_actions"] # [B, 2, act_dim]
            obs, act, next_obs = obs_seq[:, 0], act_seq[:, 0], obs_seq[:, 1]

            loss = model.compute_loss(obs, act, next_obs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_target()

            epoch_train_loss += loss.item()
            pbar.set_postfix({"train_loss": f"{loss.item():.6f}", "lr": f"{current_lr:.2e}"})
        
        avg_train_loss = epoch_train_loss / (num_train_batches + 1)
        train_losses.append(avg_train_loss)

        # Step scheduler
        scheduler.step()

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            val_gen = val_dataset.batch_generator(batch_size=args.batch_size, num_context=1, num_query=0, sample_trajectory=True)
            num_val_batches = len(val_dataset) // args.batch_size
            for batch in val_gen:
                obs_seq = batch["context_states"]
                act_seq = batch["context_actions"]
                obs, act, next_obs = obs_seq[:, 0], act_seq[:, 0], obs_seq[:, 1]
                loss = model.compute_loss(obs, act, next_obs)
                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / (num_val_batches + 1)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # Visualization
        if (epoch + 1) % args.vis_interval == 0:
            visualize_latent_prediction(model, val_dataset, epoch + 1, vis_dir, device)
            visualize_loss_curve(train_losses, val_losses, epoch + 1, vis_dir)

    # 4. Save Model
    save_path = os.path.join(save_dir, "jepa_wm_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def visualize_loss_curve(train_losses, val_losses, epoch, vis_dir):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title(f"JEPA World Model Loss - Epoch {epoch}")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(vis_dir, "loss_curve.png"))
    plt.close()

def visualize_latent_prediction(model, dataset, epoch, vis_dir, device):
    import matplotlib.pyplot as plt
    model.eval()
    with torch.no_grad():
        gen = dataset.batch_generator(batch_size=128, num_context=1, num_query=0, sample_trajectory=True)
        batch = next(gen)
        obs_seq, act_seq = batch["context_states"], batch["context_actions"]
        obs, act, next_obs = obs_seq[:, 0], act_seq[:, 0], obs_seq[:, 1]
        
        z_next_pred = model.forward(obs, act)
        z_next_target = model.get_target(next_obs)
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(z_next_target[:, 0].cpu(), z_next_target[:, 1].cpu(), alpha=0.5, label='Target')
        plt.scatter(z_next_pred[:, 0].cpu(), z_next_pred[:, 1].cpu(), alpha=0.5, label='Pred')
        plt.title(f"Latent Space (Dim 0 & 1) - Epoch {epoch}")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        errors = torch.norm(z_next_pred - z_next_target, dim=-1).cpu().numpy()
        plt.hist(errors, bins=30)
        plt.title(f"Prediction Error Distribution - Epoch {epoch}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"latent_vis_epoch_{epoch}.png"))
        plt.close()

if __name__ == "__main__":
    main()
