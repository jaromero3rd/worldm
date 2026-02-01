# Modified from RSL_RL implementation: https://github.com/leggedrobotics/rsl_rl/tree/main/rsl_rl
#
# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
from collections import deque

import numpy as np
import torch

from spr.utils import store_code_state


class Runner:
    """Runner for training and evaluation in Multi-Objective Environments."""

    def __init__(self, env, alg, train_cfg: dict, log_dir: str | None = None, eval_env=None, device="cpu"):
        self.env = env
        self.eval_env = eval_env
        self.alg = alg
        self.cfg = train_cfg
        self.device = device

        # store training configuration
        self.num_learning_iterations = self.cfg.get("num_learning_iterations", 1000)
        self.num_steps_per_env = self.alg.num_steps_per_env
        self.save_interval = self.cfg.get("save_interval", "auto")
        if self.save_interval == "auto":
            self.save_interval = self.num_learning_iterations // 10
        self.upload_model = self.cfg.get("upload_model", True)

        # evaluation
        self.eval_freq = self.cfg.get("eval_freq", 5)
        self.eval_num_envs = self.cfg.get("eval_num_envs", 8)

        # Log
        self.log_dir = log_dir
        if self.log_dir is not None:
            self._init_logger()
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.best_reward = None

        self.vector_rewbuffer = deque(maxlen=100)  # for morl reward vector
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.n_objs = getattr(self.env, "num_rewards", 1)
        self.cur_reward_sum = torch.zeros(self.env.num_envs, self.n_objs, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

    def learn_mo(self, num_weight_vectors=10):
        """Randomly sample weight vectors and train the policy for each weight vector."""
        # Generate weight vectors for multi-objective search
        weight_vectors = self._generate_weight_vectors(num_weight_vectors)

        all_results = []
        for weight_idx, weight_vector in enumerate(weight_vectors):
            # Update objective weights for the RL agent
            self.alg.update_objective_weights(weight_vector)

            # Train with current weight vector
            best_reward = self.learn()
            self.current_learning_iteration += 1  # this is because of the for loop we are using in runner.learn()
            all_results.append({"weight_vector": weight_vector, "best_reward": best_reward})
        return all_results

    def learn(self):  # noqa: C901
        """Training loop for one specific weight vector in the weight space search."""
        self.train_mode()
        obs, _ = self.env.get_observations()
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + self.num_learning_iterations
        for it in range(start_iter, tot_iter):
            # ---------------- Rollout ----------------
            start = time.time()
            with torch.no_grad():
                for _ in range(self.num_steps_per_env):
                    # Sample actions from policy
                    raw_actions = self.alg.act(obs)
                    actions = self.alg.postprocess_actions(raw_actions)

                    # Step environment
                    next_obs, rewards, dones, infos = self.env.step(actions)

                    # Process env step and store in buffer
                    self.alg.store_transition(obs, raw_actions, next_obs, rewards, dones, infos)
                    obs = next_obs

                    if self.log_dir is not None:
                        # Update statistics
                        self.cur_reward_sum += rewards
                        self.cur_episode_length += 1

                        # Store data for completed episodes, clear buffer
                        new_ids = (dones > 0).nonzero(as_tuple=False).squeeze(1)  # [n]
                        vector_rewards = self.cur_reward_sum[new_ids]  # [n, n_objs]
                        self.rewbuffer.extend(vector_rewards.matmul(self.alg.w_objs).squeeze(1).cpu().tolist())
                        self.vector_rewbuffer.extend(vector_rewards.cpu().tolist())
                        self.lenbuffer.extend(self.cur_episode_length[new_ids].cpu().tolist())
                        self.cur_reward_sum[new_ids] = 0
                        self.cur_episode_length[new_ids] = 0
            collection_time = time.time() - start

            # ---------------- Update Policy ----------------
            start = time.time()
            mean_value_loss, mean_policy_loss, info = self.alg.update(obs)
            learn_time = time.time() - start
            self.current_learning_iteration = it

            # ---------------- Evaluation ----------------
            eval_reward, eval_episode_length, eval_reward_std, eval_vector_reward = None, None, None, None
            if self.eval_env is not None and it % self.eval_freq == 0:
                start = time.time()
                eval_reward, eval_episode_length, eval_reward_std, mean_eval_vector_reward = self.eval()
                eval_time = time.time() - start
                if self.best_reward is None or eval_reward > self.best_reward:
                    self.best_reward = eval_reward

            # ---------------- Logging ----------------
            if self.log_dir is not None:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.checkpoints_dir, f"model_{it}.pt"))
        return self.best_reward

    def eval(self):
        """Evaluate the current policy on the evaluation environment."""
        # switch to evaluation mode and get inference policy
        policy = self.get_inference_policy(self.device)
        eval_rewards, eval_episode_lengths, eval_vector_rewards = [], [], []
        episodes_completed = 0
        cur_reward_sum = torch.zeros(self.eval_env.num_envs, self.n_objs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.eval_env.num_envs, dtype=torch.float, device=self.device)

        obs, _ = self.eval_env.reset()
        while episodes_completed < self.eval_num_envs:
            with torch.no_grad():
                actions = policy(obs)
                obs, rewards, dones, _ = self.eval_env.step(actions)
                cur_reward_sum += rewards
                cur_episode_length += 1
                if torch.any(dones > 0):
                    done_idx = (dones > 0).nonzero(as_tuple=False).squeeze(1)  # [n]
                    vector_rewards = cur_reward_sum[done_idx]  # [n, n_objs]
                    eval_rewards.extend(vector_rewards.matmul(self.alg.w_objs).squeeze(1).cpu().tolist())
                    eval_vector_rewards.extend(vector_rewards.cpu().tolist())
                    eval_episode_lengths.extend(cur_episode_length[done_idx].cpu().tolist())
                    episodes_completed += len(done_idx)
                    cur_reward_sum[done_idx] = 0
                    cur_episode_length[done_idx] = 0

        # Calculate statistics
        mean_eval_reward = sum(eval_rewards) / len(eval_rewards) if eval_rewards else 0
        mean_eval_episode_length = sum(eval_episode_lengths) / len(eval_episode_lengths) if eval_episode_lengths else 0
        std_eval_reward = statistics.stdev(eval_rewards) if eval_rewards else 0

        # Calculate per-objective statistics
        mean_eval_vector_reward = None
        if self.n_objs > 1 and len(eval_vector_rewards) > 0:
            mean_eval_vector_reward = np.array(eval_vector_rewards).mean(axis=0)  # [n_objs]

        # Switch back to training mode
        self.train_mode()
        return mean_eval_reward, mean_eval_episode_length, std_eval_reward, mean_eval_vector_reward

    def save(self, path: str, infos=None):
        saved_dict = {
            "actor_state_dict": self.alg.actor.state_dict(),
            "critic_state_dict": self.alg.critic.state_dict(),
            "actor_optimizer_state_dict": self.alg.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.alg.critic_optimizer.state_dict(),
            "obs_norm_state_dict": self.alg.obs_normalizer.state_dict(),
            "reward_norm_state_dict": self.alg.reward_normalizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["wandb"] and self.upload_model:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, load_actor: bool = True, load_critic: bool = True):
        loaded_dict = torch.load(path, weights_only=False, map_location=self.device)
        if load_actor:
            self.alg.actor.load_state_dict(loaded_dict["actor_state_dict"])
        if load_critic:
            self.alg.critic.load_state_dict(loaded_dict["critic_state_dict"])
        if load_optimizer:
            self.alg.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
            self.alg.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
        self.alg.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor.to(device)
            self.alg.obs_normalizer.to(device)
        return lambda x: self.alg.postprocess_actions(self.alg.actor.act_inference(self.alg.obs_normalizer(x)))

    def train_mode(self):
        self.alg.train_mode()

    def eval_mode(self):
        self.alg.eval_mode()

    # --------------------------------- Utility ---------------------------------
    def _generate_weight_vectors(self, num_vectors):
        """Simplest way to generate diverse weight vectors for multi-objective search."""
        weight_vectors = []
        # Generate uniform weight vectors on the simplex
        for i in range(num_vectors):
            if self.n_objs == 2:
                # For 2 objectives, create weights from [1, 0] to [0, 1]
                w2 = i / (num_vectors - 1) if num_vectors > 1 else 0.5
                w1 = 1 - w2
                weight_vectors.append([w1, w2])
            else:
                # For more objectives, use Dirichlet distribution
                alpha = np.ones(self.n_objs)  # Uniform Dirichlet
                weights = np.random.dirichlet(alpha)
                weight_vectors.append(weights.tolist())
        return weight_vectors

    def _init_logger(self):
        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir, exist_ok=True)

        # initialize writer
        self.logger_type = self.cfg.get("logger", "wandb")
        self.logger_type = self.logger_type.lower()

        if self.logger_type == "wandb":
            import spr
            from spr.utils.wandb_utils import WandbSummaryWriter

            self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg, env_id=self.env.name)
            self.writer.log_config(self.env.env_cfg, self.cfg, self.alg.alg_cfg, self.alg.policy_cfg)
            self.git_status_repos = [spr.__file__]
            if self.cfg.get("store_code_state", False):
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)
        elif self.logger_type == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        else:
            raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        mean_std = self.alg.actor.action_std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        # -- Global step
        self.writer.add_scalar("Global/step", self.tot_timesteps, locs["it"])

        # -- Losses
        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/policy", locs["mean_policy_loss"], locs["it"])
        if "mean_entropy_loss" in locs["info"]:
            self.writer.add_scalar("Loss/entropy", locs["info"]["mean_entropy_loss"], locs["it"])
        if "mean_value_loss_per_obj" in locs["info"]:
            for obj_idx in range(self.n_objs):
                self.writer.add_scalar(
                    f"Loss/value_loss_obj_{obj_idx + 1}", locs["info"]["mean_value_loss_per_obj"][obj_idx], locs["it"]
                )
        self.writer.add_scalar("Loss/actor_learning_rate", self.alg.actor_lr, locs["it"])
        self.writer.add_scalar("Loss/critic_learning_rate", self.alg.critic_lr, locs["it"])

        # -- Policy
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Evaluation
        if locs["eval_reward"] is not None:
            self.writer.add_scalar("Eval/mean_reward", locs["eval_reward"], locs["it"])
            self.writer.add_scalar("Eval/mean_episode_length", locs["eval_episode_length"], locs["it"])
            self.writer.add_scalar("Eval/reward_std", locs["eval_reward_std"], locs["it"])
            self.writer.add_scalar("Eval/eval_time", locs["eval_time"], locs["it"])

            # Log individual evaluation reward components for MORL
            if self.n_objs > 1 and locs["eval_vector_reward"] is not None:
                for obj_idx in range(self.n_objs):
                    self.writer.add_scalar(
                        f"Eval/mean_reward_obj_{obj_idx + 1}", locs["eval_vector_reward"][obj_idx], locs["it"]
                    )

        # -- Training
        if len(self.rewbuffer) > 0:
            # everything else
            self.writer.add_scalar("Train/mean_reward", statistics.mean(self.rewbuffer), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(self.lenbuffer), locs["it"])
        # Log individual reward components for MORL
        if "mean_explained_variance_per_obj" in locs["info"]:
            for obj_idx in range(self.n_objs):
                self.writer.add_scalar(
                    f"Train/explained_variance_obj_{obj_idx + 1}",
                    locs["info"]["mean_explained_variance_per_obj"][obj_idx],
                    locs["it"],
                )
        if self.n_objs > 1 and len(self.vector_rewbuffer) > 0:
            mean_rewards_per_obj = np.array(list(self.vector_rewbuffer)).mean(axis=0)  # [n_objs]
            for obj_idx in range(self.n_objs):
                self.writer.add_scalar(
                    f"Train/mean_reward_obj_{obj_idx + 1}", mean_rewards_per_obj[obj_idx].item(), locs["it"]
                )
            for obj_idx in range(self.n_objs):
                self.writer.add_scalar(f"Train/weight_obj_{obj_idx + 1}", self.alg.w_objs[obj_idx].item(), locs["it"])
        if self.alg.reward_normalization:
            self.writer.add_scalar("Train/reward_mean", self.alg.reward_normalizer.emp_norm.mean, locs["it"])
            self.writer.add_scalar("Train/reward_std", self.alg.reward_normalizer.emp_norm.std, locs["it"])

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(self.rewbuffer) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_policy_loss']:.4f}\n"""
            )
            log_string += f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            log_string += f"""{'Mean total reward:':>{pad}} {statistics.mean(self.rewbuffer):.2f}\n"""
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(self.lenbuffer):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_policy_loss']:.4f}\n"""
            )
            log_string += f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""

        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (
                               locs['start_iter'] + self.num_learning_iterations - locs['it']):.1f}s\n"""
        )
        print(log_string)
