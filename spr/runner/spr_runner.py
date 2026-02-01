import os
import pickle
from dataclasses import asdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm.auto import tqdm

from spr.dataset import MoMujocoTrajectoryDataset, TrajectoryDataset
from spr.modules import PerObjectiveValueHead, PolicyVAE, RnCLoss
from spr.utils import Config


class PolicyVAERunner:
    """
    Runner for training, evaluating, and managing Policy VAE.
    """

    def __init__(self, task_id: str, config: Config, device: str = "cuda:0", log_dir: Optional[str] = None):
        self.task_id = task_id
        self.config = config
        self.device = device

        # Setup directories
        if log_dir is None:
            self.log_dir = os.path.join("datasets", self.task_id)
        else:
            self.log_dir = log_dir
        self.save_dir = os.path.join(self.log_dir, "policy_vae")
        os.makedirs(self.save_dir, exist_ok=True)

        # Hyperparameters from config
        self.lr = config.trainer.learning_rate
        self.loss_coefs = {
            "contrastive": config.trainer.coef_contrastive,
            "value": config.trainer.coef_value,
            "decoder": config.trainer.coef_decoder,
            "ortho": config.trainer.coef_ortho,
        }
        self.kl_sched = {
            "start": config.trainer.coef_kl_start,
            "end": config.trainer.coef_kl_end,
        }
        self.sample_trajectory = config.trainer.sample_trajectory
        self.context_length = config.vae.context_length

        # Normalization
        self.obs_norm = config.vae.obs_norm
        self.value_norm = config.vae.value_norm

        # VAE Training Hyperparameters
        self.vae_epochs = config.trainer.vae_epochs
        self.vae_batch_size = config.trainer.vae_batch_size
        self.vae_num_context = config.trainer.vae_num_context
        self.vae_num_query = config.trainer.vae_num_query

        # Value Head Training Hyperparameters
        self.value_epochs = config.trainer.value_epochs
        self.value_batch_size = config.trainer.value_batch_size
        self.value_num_context = config.trainer.value_num_context

        # Contrastive Loss
        self.contrastive_loss_type = config.trainer.contrastive_loss_type
        if self.contrastive_loss_type == "rnc":
            self.contrastive_loss = RnCLoss(
                temperature=config.trainer.rnc_temperature,
                label_diff=config.trainer.rnc_label_diff,
                feature_sim=config.trainer.rnc_feature_sim,
            ).to(self.device)
        else:
            raise ValueError(f"Invalid contrastive loss type: {self.contrastive_loss_type}")

        # Placeholders for data, model, and dimensions
        self.dataset: Optional[TrajectoryDataset] = None
        self.vae: Optional[PolicyVAE] = None
        self.state_dim = None
        self.action_dim = None
        self.n_objs = None

    def init_model(self):
        """
        Must be called after state_dim, action_dim, and n_objs are set.
        """
        self.vae = PolicyVAE(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            n_objs=self.n_objs,
            config=self.config.vae,
        ).to(self.device)
        self.vae_opt = optim.AdamW(self.vae.parameters(), lr=self.lr)
        self.value_opt = optim.AdamW(self.vae.value_head.parameters(), lr=self.lr)

    def reinitialize_value_head(self):
        self.vae.value_head = PerObjectiveValueHead(
            self.config.vae.encoder.obj_n_embd, self.n_objs, self.config.vae.value_head_hidden_dims
        ).to(self.device)
        self.value_opt = optim.AdamW(self.vae.value_head.parameters(), lr=self.lr)

    def load_dataset(
        self,
        dataset_type: str,
        task_id: str,
        dataset_config: dict,  # treat dataset config as a dictionary for flexibility
    ):
        """Load the dataset, and update the normalizers."""
        if dataset_type == "momujoco":
            self.dataset = MoMujocoTrajectoryDataset(
                context_length=self.context_length,
                device=self.device,
            )
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}")
        self.dataset.load(task_id, **dataset_config)

        self.state_dim = self.dataset.state_dim
        self.action_dim = self.dataset.action_dim
        self.n_objs = self.dataset.n_objs

    def train(self, value_only: bool = False):
        if self.dataset is None:
            raise ValueError("Dataset not loaded!")

        if value_only:
            self.load(self.save_dir)
            for p in self.vae.encoder.parameters():
                p.requires_grad = False
            self.vae.encoder.eval()
            self.reinitialize_value_head()
        else:
            self.init_model()
            self.vae.update_normalization(self.dataset)
            print(f"{'='*30}\nTraining VAE\n{'='*30}")
            for ep in range(self.vae_epochs):
                stats = self.train_epoch(ep)
                print(
                    f"Ep {ep+1:3d}: Total={stats['total']:.3f} "
                    f"Recon={stats['dec']:.3f} RnC={stats['rnc']:.3f} Ortho={stats['ortho']:.3f} "
                    f"KL={stats['kl']:.3f} (coef={stats['kl_coef']:.4f}, sig={stats['std']:.2f}, norm={stats['norm']:.2f}) "
                    f"Value={stats['val']:.3f} "
                )
        self.train_value_head()
        self.save()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.vae.train()
        total_loss_sum, rnc_loss_sum, dec_loss_sum, ortho_loss_sum = 0.0, 0.0, 0.0, 0.0
        kl_loss_sum, avg_std_sum, avg_norm_sum, val_loss_sum = 0.0, 0.0, 0.0, 0.0
        num_batches = max(1, len(self.dataset) // self.vae_batch_size)
        batch_iter = self.dataset.batch_generator(
            self.vae_batch_size,
            num_context=self.vae_num_context,
            num_query=self.vae_num_query,
            sample_trajectory=self.sample_trajectory,
            shuffle=True,
        )

        for batch in tqdm(
            batch_iter,
            total=num_batches,
            desc=f"VAE Epoch {epoch+1}/{self.vae_epochs}",
            ncols=0,
        ):
            # --------- 1. Contrastive Loss ---------
            _, obj_reprs = self.vae.encode(
                {"context_states": batch["context_states"], "context_actions": batch["context_actions"]},
                False,
            )
            loss_rnc = 0.0
            for i in range(self.n_objs):
                loss_rnc += self.contrastive_loss(obj_reprs[:, i], batch["returns"][:, i]) / self.n_objs

            # --------- 2. Decoder & KL Loss ---------
            mu, log_std = self.vae.encode(
                {"context_states": batch["query_context_states"], "context_actions": batch["query_context_actions"]},
                True,
            )

            # KL
            loss_kl, avg_std, avg_norm = self._kl(mu, log_std)

            # Behavior cloning
            z, _ = self.vae.sample(mu, log_std)  # [B, n_embd]
            act_dist = self.vae.decode(batch["query_states"], z)  # a Normal distribution
            loss_dec = -act_dist.log_prob(batch["query_actions"]).sum(dim=-1).mean()  # [num_total_query] -> scalar

            # --------- 4. Total Loss ---------
            kl_coef = self._kl_coef(epoch)
            ortho_loss = self.vae.get_orthogonality_loss()
            total = (
                self.loss_coefs["contrastive"] * loss_rnc
                + self.loss_coefs["decoder"] * loss_dec
                + kl_coef * loss_kl
                + self.loss_coefs["ortho"] * ortho_loss
            )

            # Step
            self.vae_opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
            self.vae_opt.step()

            # Update stats
            total_loss_sum += total.item()
            dec_loss_sum += loss_dec.item()
            kl_loss_sum += loss_kl.item()
            rnc_loss_sum += loss_rnc.item()
            ortho_loss_sum += ortho_loss.item()
            avg_std_sum += avg_std
            avg_norm_sum += avg_norm

        # Calculate the max and min singular value for each obj_head
        max_svs = []
        min_svs = []
        for i, head in enumerate(self.vae.encoder.obj_heads):
            svs1 = torch.linalg.svdvals(head.layers[-1].weight).cpu().detach().numpy()
            max_svs.append((svs1.max()))
            min_svs.append((svs1.min()))
        for i, (mx, mn) in enumerate(zip(max_svs, min_svs)):
            print(f"ObjHead {i}: MaxSVs: {mx}, MinSVs: {mn}")

        return {
            "total": total_loss_sum / num_batches,
            "rnc": rnc_loss_sum / num_batches,
            "dec": dec_loss_sum / num_batches,
            "ortho": ortho_loss_sum / num_batches,
            "kl": kl_loss_sum / num_batches,
            "kl_coef": kl_coef,
            "std": avg_std_sum / num_batches,
            "norm": avg_norm_sum / num_batches,
            "val": val_loss_sum / num_batches,
        }

    def train_value_head(self):
        print(f"{'='*30}\nFitting Value Head\n{'='*30}")
        self.vae.value_head.train()
        self.vae.encoder.eval()
        for ep in range(self.value_epochs):
            ep_loss = 0.0
            for batch in self.dataset.batch_generator(
                self.value_batch_size, self.value_num_context, 0, self.sample_trajectory, False
            ):
                with torch.no_grad():
                    _, obj_reprs = self.vae.encode(batch, return_distribution=False)
                pred = self.vae.value_head(obj_reprs)  # [B, n_objs]
                loss = nn.functional.mse_loss(pred, self.vae.value_normalizer(batch["returns"]), reduction="mean")

                self.value_opt.zero_grad()
                loss.backward()
                self.value_opt.step()
                ep_loss += loss.item()

            if (ep + 1) % 10 == 0:
                print(
                    f"VH Epoch {ep+1}/{self.value_epochs}: Loss {ep_loss/(len(self.dataset)/self.value_batch_size):.4f}"
                )

    @torch.no_grad()
    def evaluate(self, env, z: torch.Tensor, episodes: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the policy defined by latent z in the given environment.

        Args:
            z: Latent policy representation [1, n_embd].
        """
        self.vae.eval()
        obs, _ = env.reset()
        returns = []
        current_returns = torch.zeros(env.num_envs, env.num_rewards, device=self.device)
        while len(returns) < episodes:
            action = self.vae.act_inference(obs, z)
            obs, rew, term, _ = env.step(action)
            current_returns += rew

            done_idx = torch.where(term)[0]  # [n]
            if len(done_idx) > 0:
                vector_returns = current_returns[done_idx]  # [n, n_objs]
                returns.extend(vector_returns.cpu().tolist())
                current_returns[done_idx] = 0
        returns = np.asarray(returns[:episodes])
        return returns.mean(axis=0), returns.std(axis=0)

    # ----------------------------- Helper Functions -----------------------------
    def _kl_coef(self, epoch: int) -> float:
        return self.kl_sched["start"] + (self.kl_sched["end"] - self.kl_sched["start"]) * epoch / self.vae_epochs

    def _kl(self, mean: torch.Tensor, log_std: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Compute KL(N(mean, std^2) || N(0, 1))."""
        std = torch.exp(log_std)
        kl = 0.5 * (mean.pow(2) + std.pow(2) - 1 - 2 * log_std).sum(dim=-1).mean()
        return kl, std.mean().item(), torch.norm(mean, p=2, dim=-1).mean().item()

    def save(self):
        """Save the VAE model and precomputed representations."""
        # 1. Compute representations
        self.vae.eval()
        repr_cache = []
        with torch.no_grad():
            batch_iter = self.dataset.batch_generator(
                batch_size=1024,  # avoid OOM
                num_context=1,
                num_query=0,
                sample_trajectory=self.sample_trajectory,
                shuffle=False,
            )
            for batch in batch_iter:
                inp = {"context_states": batch["context_states"], "context_actions": batch["context_actions"]}
                mu, log_std = self.vae.encode(inp, True)
                repr_cache.extend(
                    zip(
                        mu.cpu().numpy(),
                        log_std.cpu().numpy(),
                        batch["returns"].cpu().numpy(),
                        batch["model_idx"].cpu().numpy(),
                    )
                )
        with open(os.path.join(self.save_dir, "policy_representations.pkl"), "wb") as f:
            pickle.dump(repr_cache, f)

        # 2. Save System
        sys_data = {
            "model": self.vae.state_dict(),
            "config": self.config,  # Save entire config as dataclass
            "state_dim": self.state_dim,  # Save dims separately for convenience/legacy
            "action_dim": self.action_dim,
            "n_objs": self.n_objs,
        }
        torch.save(sys_data, os.path.join(self.save_dir, "policy_vae_system.pt"))

        # Save config as yaml
        config_dict = asdict(self.config)
        with open(os.path.join(self.save_dir, "policy_vae_config.yaml"), "w") as f:
            yaml.dump(config_dict, f)

        print(f"Saved models, representations to {self.save_dir}")

    def load(self, path: str):
        """Load a trained system from path."""
        print(f"Loading system from {path}")
        sys_path = os.path.join(path, "policy_vae_system.pt")
        if not os.path.exists(sys_path):
            raise FileNotFoundError(f"Policy VAE system file not found: {sys_path}")
        ckpt = torch.load(sys_path, map_location=self.device, weights_only=False)

        self.config = ckpt["config"]
        self.state_dim = ckpt["state_dim"]
        self.action_dim = ckpt["action_dim"]
        self.n_objs = ckpt["n_objs"]
        self.init_model()
        self.vae.load_state_dict(ckpt["model"])
