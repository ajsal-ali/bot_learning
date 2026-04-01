#!/usr/bin/env python3
"""
Main training script for GO-BDX RL walking.

Usage:
    # Train from scratch (stage 1)
    python -m rl_forwalking.train --model_path go_bdx.xml

    # Resume from checkpoint at stage 2
    python -m rl_forwalking.train --model_path go_bdx.xml \
        --checkpoint checkpoints/curriculum/standing_complete.zip --stage 2

    # Train with fewer parallel envs (low VRAM)
    python -m rl_forwalking.train --model_path go_bdx.xml --num_envs 4
"""

import argparse
import os
import sys
import time
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from . import config as C
from .env import GoBdxWalkingEnv
from .curriculum import CurriculumScheduler
from .logger import TrainingLogger


def make_env(model_path: str, stage: int, rank: int, seed: int = 0):
    """Factory function for creating environments."""
    def _init():
        env = GoBdxWalkingEnv(
            model_path=model_path,
            curriculum_stage=stage,
            target_velocity=0.0,
            randomize=True,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


class CurriculumCallback(BaseCallback):
    """
    SB3 callback that:
      - Tracks episode outcomes for curriculum
      - Logs training metrics
      - Saves periodic and curriculum checkpoints
      - Updates environment stage on curriculum advance
    """

    def __init__(
        self,
        curriculum: CurriculumScheduler,
        logger_obj: TrainingLogger,
        model_path: str,
        checkpoint_dir: str = "./checkpoints",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.curriculum = curriculum
        self.train_logger = logger_obj
        self.model_path = model_path
        self.checkpoint_dir = checkpoint_dir
        self.best_reward = -np.inf
        self.episode_count = 0

        os.makedirs(os.path.join(checkpoint_dir, "periodic"), exist_ok=True)
        os.makedirs(os.path.join(checkpoint_dir, "curriculum"), exist_ok=True)
        os.makedirs(os.path.join(checkpoint_dir, "best"), exist_ok=True)

        # Wire up curriculum checkpoint saving
        self.curriculum.save_checkpoint_fn = self._save_curriculum_checkpoint

    def _save_curriculum_checkpoint(self, name: str):
        path = os.path.join(self.checkpoint_dir, "curriculum", name)
        self.model.save(path)
        # Save VecNormalize stats if available
        if isinstance(self.training_env, VecNormalize):
            self.training_env.save(path + "_vecnorm.pkl")
        print(f"[Checkpoint] Saved curriculum checkpoint: {path}")

    def _save_periodic_checkpoint(self):
        step = self.num_timesteps
        path = os.path.join(self.checkpoint_dir, "periodic", f"step_{step}")
        self.model.save(path)
        if isinstance(self.training_env, VecNormalize):
            self.training_env.save(path + "_vecnorm.pkl")

    def _save_best_checkpoint(self):
        path = os.path.join(self.checkpoint_dir, "best", "best_model")
        self.model.save(path)
        if isinstance(self.training_env, VecNormalize):
            self.training_env.save(path + "_vecnorm.pkl")

    def _on_step(self) -> bool:
        # Check for completed episodes in all parallel envs
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                infos = self.locals.get("infos", [])
                if i < len(infos):
                    info = infos[i]
                    self.episode_count += 1

                    # Record in curriculum
                    result = self.curriculum.record_episode(info)

                    # Log metrics
                    self.train_logger.log(
                        step=self.num_timesteps,
                        reward=info.get("episode_reward", 0),
                        episode_length=info.get("step", 0),
                        success=result["success"],
                        stage=result["stage"],
                        height=info.get("height", 0),
                        velocity=info.get("velocity", 0),
                    )

                    # Track best reward
                    ep_reward = info.get("episode_reward", 0)
                    if ep_reward > self.best_reward:
                        self.best_reward = ep_reward
                        self._save_best_checkpoint()

                    # Stage advanced - update all envs
                    if result["advanced"]:
                        self._update_env_stages()

                    # Print status periodically
                    if self.episode_count % 50 == 0:
                        print(f"[Train] Step {self.num_timesteps:>8d} | "
                              f"Ep {self.episode_count:>5d} | "
                              f"{self.curriculum.status_str()} | "
                              f"Best: {self.best_reward:.2f}")

        # Periodic checkpoint
        if self.num_timesteps % C.CHECKPOINT_INTERVAL == 0 and self.num_timesteps > 0:
            self._save_periodic_checkpoint()
            self.curriculum.save_state()
            self.train_logger.save_logs()

        # Update plots
        if self.num_timesteps % C.PLOT_INTERVAL == 0 and self.num_timesteps > 0:
            self.train_logger.update_plots()

        return True

    def _update_env_stages(self):
        """Update curriculum stage in all parallel environments."""
        opts = self.curriculum.get_env_options()
        stage = opts["curriculum_stage"]
        vel = opts["target_velocity"]

        # For SubprocVecEnv, we need to set via env method calls
        env = self.training_env
        # Unwrap VecNormalize if present
        while hasattr(env, "venv"):
            env = env.venv

        if hasattr(env, "env_method"):
            env.env_method("set_curriculum_stage", stage)
            env.env_method("set_target_velocity", vel)

    def _on_training_end(self):
        self.train_logger.close()
        self.curriculum.save_state()
        print(f"[Train] Training complete. {self.episode_count} episodes, "
              f"{self.num_timesteps} steps.")


def train(args):
    model_path = os.path.abspath(args.model_path)
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[Train] Device: {device}")
    if device == "cuda":
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")

    # Curriculum
    curriculum = CurriculumScheduler(
        start_stage=args.stage,
        checkpoint_dir=os.path.join(args.output_dir, "checkpoints", "curriculum"),
        log_dir=os.path.join(args.output_dir, "logs"),
    )

    # Logger
    logger = TrainingLogger(
        log_dir=os.path.join(args.output_dir, "logs"),
        live_plot=args.live_plot,
    )

    # Parallel environments
    print(f"[Train] Creating {args.num_envs} parallel environments...")
    env_fns = [
        make_env(model_path, args.stage, i, seed=args.seed)
        for i in range(args.num_envs)
    ]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # PPO config
    policy_kwargs = {
        "net_arch": C.PPO_CONFIG["policy_kwargs"]["net_arch"],
    }

    # Map activation function string to class
    act_name = C.PPO_CONFIG["policy_kwargs"]["activation_fn"]
    act_cls = getattr(torch.nn, act_name, torch.nn.ELU)
    policy_kwargs["activation_fn"] = act_cls

    if args.checkpoint:
        print(f"[Train] Loading checkpoint: {args.checkpoint}")
        model = PPO.load(args.checkpoint, env=vec_env, device=device)
        # Load VecNormalize if exists
        vecnorm_path = args.checkpoint.replace(".zip", "_vecnorm.pkl")
        if os.path.exists(vecnorm_path):
            vec_env = VecNormalize.load(vecnorm_path, vec_env.venv)
            model.set_env(vec_env)
            print(f"[Train] Loaded VecNormalize from {vecnorm_path}")
        # Resume curriculum
        curriculum.load_state()
    else:
        print("[Train] Creating new PPO model...")
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=C.PPO_CONFIG["learning_rate"],
            n_steps=C.PPO_CONFIG["n_steps"],
            batch_size=C.PPO_CONFIG["batch_size"],
            n_epochs=C.PPO_CONFIG["n_epochs"],
            gamma=C.PPO_CONFIG["gamma"],
            gae_lambda=C.PPO_CONFIG["gae_lambda"],
            clip_range=C.PPO_CONFIG["clip_range"],
            ent_coef=C.PPO_CONFIG["ent_coef"],
            vf_coef=C.PPO_CONFIG["vf_coef"],
            max_grad_norm=C.PPO_CONFIG["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=1,
        )

    # Callback
    callback = CurriculumCallback(
        curriculum=curriculum,
        logger_obj=logger,
        model_path=model_path,
        checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
    )

    # Train
    total_steps = args.total_timesteps or C.TOTAL_TIMESTEPS
    print(f"[Train] Starting training for {total_steps:,} steps...")
    print(f"[Train] Stage: {args.stage} ({C.CURRICULUM_STAGES[args.stage]})")
    print(f"[Train] Output: {args.output_dir}")
    start = time.time()

    model.learn(
        total_timesteps=total_steps,
        callback=callback,
        log_interval=C.LOG_INTERVAL,
        progress_bar=True,
    )

    elapsed = time.time() - start
    print(f"[Train] Done in {elapsed/3600:.1f} hours ({elapsed:.0f}s)")

    # Final saves
    final_path = os.path.join(args.output_dir, "checkpoints", "final_model")
    model.save(final_path)
    vec_env.save(final_path + "_vecnorm.pkl")
    print(f"[Train] Saved final model: {final_path}")

    vec_env.close()


def main():
    parser = argparse.ArgumentParser(description="Train GO-BDX walking with PPO")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to go_bdx.xml MuJoCo model")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Curriculum stage to start at (default: 1)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint .zip to resume from")
    parser.add_argument("--num_envs", type=int, default=C.NUM_ENVS,
                        help=f"Number of parallel environments (default: {C.NUM_ENVS})")
    parser.add_argument("--total_timesteps", type=int, default=None,
                        help=f"Total training steps (default: {C.TOTAL_TIMESTEPS:,})")
    parser.add_argument("--output_dir", type=str, default="./rl_forwalking_output",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if GPU available")
    parser.add_argument("--live_plot", action="store_true",
                        help="Show live matplotlib plots during training")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
