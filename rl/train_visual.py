#!/usr/bin/env python3
"""
GO-BDX Training with Visual Rendering

Same as train.py but with MuJoCo visualization.
Includes: parallel envs, matplotlib plotting, periodic checkpoints.

Usage:
    python train_visual.py --n-envs 4 --timesteps 1000000
"""

import os
import sys
import argparse
import numpy as np
import torch
import time
import csv
import matplotlib
# Use TkAgg for interactive plotting
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, List
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from go_bdx_env import GoBdxEnv
from curriculum import CurriculumScheduler


class TrainingLogger:
    """Matplotlib-based training logger with live plotting."""
    
    def __init__(self, log_dir: str, plot_interval: int = 50):
        self.log_dir = log_dir
        self.plot_interval = plot_interval
        os.makedirs(log_dir, exist_ok=True)
        
        self.rewards: List[float] = []
        self.lengths: List[int] = []
        self.success_rates: List[float] = []
        self.stages: List[int] = []
        self.timestamps: List[float] = []
        self.start_time = time.time()
        
        # Interactive plot setup
        plt.ion()  # Turn on interactive mode
        self.fig = None
        self.axes = None
        
    def log_episode(self, reward: float, length: int, success_rate: float, stage: int):
        self.rewards.append(reward)
        self.lengths.append(length)
        self.success_rates.append(success_rate)
        self.stages.append(stage)
        self.timestamps.append(time.time() - self.start_time)
        
        if len(self.rewards) % self.plot_interval == 0:
            self.update_plot()
    
    def update_plot(self):
        if len(self.rewards) < 2:
            return
        
        # Create figure on first call
        if self.fig is None:
            self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
            self.fig.suptitle('GO-BDX Training Progress (Live)', fontsize=14)
        
        # Clear axes
        for ax in self.axes.flat:
            ax.clear()
        
        episodes = range(len(self.rewards))
        
        # Rewards
        ax1 = self.axes[0, 0]
        ax1.plot(episodes, self.rewards, alpha=0.3, color='blue', label='Raw')
        if len(self.rewards) >= 10:
            window = min(100, len(self.rewards))
            rolling = np.convolve(self.rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(self.rewards)), rolling, color='blue', linewidth=2, label=f'Avg({window})')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Episode Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Success Rate
        ax2 = self.axes[0, 1]
        ax2.plot(episodes, self.success_rates, color='green', linewidth=2)
        ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='80% threshold')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Success Rate & Stage')
        ax2.set_ylim(0, 1.05)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(episodes, self.stages, color='orange', linestyle='--')
        ax2_twin.set_ylabel('Stage', color='orange')
        ax2_twin.set_ylim(0, 5)
        
        # Episode Length
        ax3 = self.axes[1, 0]
        ax3.plot(episodes, self.lengths, alpha=0.3, color='purple')
        if len(self.lengths) >= 10:
            window = min(100, len(self.lengths))
            rolling = np.convolve(self.lengths, np.ones(window)/window, mode='valid')
            ax3.plot(range(window-1, len(self.lengths)), rolling, color='purple', linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Steps')
        ax3.set_title('Episode Length')
        ax3.grid(True, alpha=0.3)
        
        # Stats
        ax4 = self.axes[1, 1]
        ax4.axis('off')
        elapsed = (time.time() - self.start_time) / 60
        avg_rew = np.mean(self.rewards[-100:]) if len(self.rewards) >= 100 else np.mean(self.rewards)
        stage_names = ["", "Standing", "Balance", "Stepping", "Walking"]
        stats = f"""
        Episodes: {len(self.rewards)}
        Time: {elapsed:.1f} min
        
        Stage: {self.stages[-1]} ({stage_names[self.stages[-1]]})
        Success: {self.success_rates[-1]:.1%}
        Avg Reward: {avg_rew:.2f}
        """
        ax4.text(0.1, 0.5, stats, fontsize=14, family='monospace', verticalalignment='center')
        
        plt.tight_layout()
        
        # Save to file AND show live
        self.fig.savefig(os.path.join(self.log_dir, 'training_progress.png'), dpi=150)
        plt.pause(0.01)  # Brief pause to update display
    
    def save_data(self):
        with open(os.path.join(self.log_dir, 'episode_data.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'reward', 'length', 'success_rate', 'stage', 'time'])
            for i, (r, l, s, st, t) in enumerate(zip(
                self.rewards, self.lengths, self.success_rates, self.stages, self.timestamps
            )):
                writer.writerow([i, r, l, s, st, t])
        print(f"📊 Data saved to {self.log_dir}/episode_data.csv")
    
    def close(self):
        if self.fig is not None:
            self.update_plot()
            plt.ioff()  # Turn off interactive mode
            plt.close(self.fig)
        self.save_data()


class VisualCallback(BaseCallback):
    """Callback with rendering, logging, and curriculum tracking."""
    
    def __init__(self, render_env: GoBdxEnv, curriculum: CurriculumScheduler, 
                 training_logger: TrainingLogger, checkpoint_dir: str, 
                 realtime: bool = False, verbose: int = 1):
        super().__init__(verbose)
        self.render_env = render_env
        self.curriculum = curriculum
        self.training_logger = training_logger  # Renamed to avoid conflict with BaseCallback.logger
        self.checkpoint_dir = checkpoint_dir
        self.episode_count = 0
        self.current_ep_rewards = {}
        self.render_obs = None
        self.realtime = realtime  # If True, slow down to real-time (50Hz = 0.02s per step)
        
    def _on_training_start(self):
        self.render_obs, _ = self.render_env.reset()
        
    def _on_step(self) -> bool:
        # Render visualization env
        action, _ = self.model.predict(self.render_obs, deterministic=False)
        self.render_obs, _, term, trunc, _ = self.render_env.step(action)
        self.render_env.render()
        
        # Slow down to real-time if requested (50Hz = 0.02s per step)
        if self.realtime:
            time.sleep(0.02)
        
        if term or trunc:
            self.render_obs, _ = self.render_env.reset()
        
        # Track training episodes
        for i, done in enumerate(self.locals['dones']):
            if i not in self.current_ep_rewards:
                self.current_ep_rewards[i] = 0
            self.current_ep_rewards[i] += self.locals['rewards'][i]
            
            if done:
                info = self.locals['infos'][i]
                ep_reward = self.current_ep_rewards[i]
                ep_length = info.get('step', 0)
                self.current_ep_rewards[i] = 0
                self.episode_count += 1
                
                # Curriculum
                if 'termination_reason' in info:
                    result = self.curriculum.record_episode(info)
                    
                    # Stage advance - save checkpoint
                    if result['advanced']:
                        path = os.path.join(self.checkpoint_dir, 'curriculum',
                                          f'stage_{self.curriculum.stage-1}_complete.zip')
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        self.model.save(path)
                        print(f"\n💾 Stage checkpoint: {path}")
                
                # Log
                self.training_logger.log_episode(
                    reward=ep_reward,
                    length=ep_length,
                    success_rate=self.curriculum.success_rate,
                    stage=self.curriculum.stage
                )
                
                # Print progress
                if self.episode_count % 20 == 0:
                    avg = np.mean(self.training_logger.rewards[-100:]) if len(self.training_logger.rewards) >= 100 else np.mean(self.training_logger.rewards)
                    print(f"Ep {self.episode_count}: R={ep_reward:.1f}, Avg={avg:.1f}, "
                          f"Success={self.curriculum.success_rate:.1%}, Stage={self.curriculum.stage}")
        
        return True
    
    def _on_training_end(self):
        self.render_env.close()
        self.curriculum.save_state()
        self.training_logger.close()


def train_visual(
    timesteps: int = 500_000,
    n_envs: int = 4,
    start_stage: int = 1,
    resume_path: Optional[str] = None,
    save_dir: str = "./checkpoints",
    log_dir: str = "./logs",
    realtime: bool = False,  # NEW: Set to True for real-time visualization
):
    """Train with visualization, plotting, and checkpoints."""
    
    print("=" * 60)
    print("GO-BDX Training with Visualization")
    print("=" * 60)
    print(f"Timesteps: {timesteps:,}")
    print(f"Parallel Envs: {n_envs}")
    print(f"Stage: {start_stage}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Realtime Mode: {'ON (slow, for checking)' if realtime else 'OFF (fast training)'}")
    print("=" * 60)
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Curriculum
    curriculum = CurriculumScheduler(
        start_stage=start_stage,
        checkpoint_dir=os.path.join(save_dir, "curriculum"),
        log_dir=log_dir
    )
    curriculum.load_state()
    print(f"📚 Curriculum: Stage {curriculum.stage} ({curriculum.stage_name}), Success: {curriculum.success_rate:.1%}")
    
    # Training envs (no render)
    def make_env(rank):
        def _init():
            env = GoBdxEnv(render_mode=None, curriculum_stage=curriculum.stage,
                          max_episode_steps=250, randomize=True)
            env.reset(seed=rank * 1000)
            return env
        return _init
    
    print(f"Creating {n_envs} training environments...")
    vec_env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    
    # Render env (separate)
    print("Creating render environment...")
    render_env = GoBdxEnv(render_mode="human", curriculum_stage=curriculum.stage,
                         max_episode_steps=250, randomize=True)
    render_env.reset(seed=9999)
    
    # Logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join(log_dir, f"run_{timestamp}")
    logger = TrainingLogger(run_log_dir, plot_interval=50)
    
    # PPO - same params as train.py
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if resume_path and os.path.exists(resume_path):
        print(f"📂 Resuming from: {resume_path}")
        model = PPO.load(resume_path, env=vec_env, device=device)
    else:
        model = PPO(
            "MlpPolicy", vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=0.2,
            ent_coef=0.005,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
            device=device,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
                activation_fn=torch.nn.ELU,
            ),
        )
    
    # Callbacks
    visual_callback = VisualCallback(render_env, curriculum, logger, save_dir, realtime=realtime)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1000),
        save_path=os.path.join(save_dir, "periodic"),
        name_prefix="go_bdx",
        verbose=1
    )
    
    print("=" * 60)
    print("🚀 Training started! Watch MuJoCo viewer.")
    print(f"📊 Plots saved to: {run_log_dir}/training_progress.png")
    print("Press Ctrl+C to stop and save.")
    print("=" * 60)
    
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=[visual_callback, checkpoint_callback],
            progress_bar=True,
            reset_num_timesteps=resume_path is None
        )
    except KeyboardInterrupt:
        print("\n\n⏸️  Training interrupted!")
    finally:
        # Save final
        final_path = os.path.join(save_dir, "final_model.zip")
        model.save(final_path)
        print(f"\n💾 Final model: {final_path}")
        vec_env.close()
    
    print(f"\nFinal: {visual_callback.episode_count} episodes")
    print(f"Success: {curriculum.success_rate:.1%}, Stage: {curriculum.stage}")


def main():
    parser = argparse.ArgumentParser(description='Train GO-BDX with visualization')
    parser.add_argument('--timesteps', '-t', type=int, default=2_000_000)
    parser.add_argument('--n-envs', '-n', type=int, default=4)
    parser.add_argument('--stage', '-s', type=int, default=1, choices=[1,2,3,4])
    parser.add_argument('--resume', '-r', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--save-dir', type=str, default='./checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--realtime', action='store_true', help='Run at real-time speed (slow, for checking)')
    args = parser.parse_args()
    
    train_visual(
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        start_stage=args.stage,
        resume_path=args.resume,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        realtime=args.realtime,
    )


if __name__ == "__main__":
    main()
