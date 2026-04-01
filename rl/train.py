#!/usr/bin/env python3
"""
GO-BDX PPO Training Script

Train the bipedal robot to walk using Proximal Policy Optimization.
Uses curriculum learning: standing → balance → stepping → walking

Author: RL Training Pipeline
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, Dict, Any, List
from collections import deque

# Stable Baselines 3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import get_linear_fn

# Local imports
from go_bdx_env import GoBdxEnv
from curriculum import CurriculumScheduler


class TrainingLogger:
    """
    Matplotlib-based training logger (no TensorBoard).
    
    Tracks and plots:
        - Episode rewards
        - Episode lengths
        - Success rate
        - Curriculum stage
        - Policy loss
        - Value loss
    """
    
    def __init__(self, log_dir: str, plot_interval: int = 50):
        """
        Initialize training logger.
        
        Args:
            log_dir: Directory to save logs and plots
            plot_interval: How often to update plots (in episodes)
        """
        self.log_dir = log_dir
        self.plot_interval = plot_interval
        
        os.makedirs(log_dir, exist_ok=True)
        
        # Tracking data
        self.rewards: List[float] = []
        self.lengths: List[int] = []
        self.success_rates: List[float] = []
        self.stages: List[int] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.timestamps: List[float] = []
        
        # Rolling averages
        self.reward_window = deque(maxlen=100)
        
        # Start time
        self.start_time = time.time()
        
        # Plot setup
        self.fig = None
        self.axes = None
        
    def log_episode(
        self,
        reward: float,
        length: int,
        success_rate: float,
        stage: int,
        policy_loss: Optional[float] = None,
        value_loss: Optional[float] = None
    ):
        """Log a single episode."""
        self.rewards.append(reward)
        self.lengths.append(length)
        self.success_rates.append(success_rate)
        self.stages.append(stage)
        self.timestamps.append(time.time() - self.start_time)
        
        if policy_loss is not None:
            self.policy_losses.append(policy_loss)
        if value_loss is not None:
            self.value_losses.append(value_loss)
        
        self.reward_window.append(reward)
        
        # Update plot periodically
        if len(self.rewards) % self.plot_interval == 0:
            self.update_plot()
    
    def update_plot(self, save: bool = True):
        """Update the training progress plot."""
        if len(self.rewards) < 2:
            return
        
        if self.fig is None:
            plt.ion()  # Interactive mode
            self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
            self.fig.suptitle('GO-BDX Training Progress', fontsize=14)
        
        # Clear axes
        for ax in self.axes.flat:
            ax.clear()
        
        episodes = range(len(self.rewards))
        
        # Plot 1: Rewards
        ax1 = self.axes[0, 0]
        ax1.plot(episodes, self.rewards, alpha=0.3, color='blue', label='Raw')
        # Rolling average
        if len(self.rewards) >= 10:
            window = min(100, len(self.rewards))
            rolling = np.convolve(self.rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(self.rewards)), rolling, color='blue', linewidth=2, label=f'Avg ({window} ep)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Episode Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Success Rate and Stage
        ax2 = self.axes[0, 1]
        ax2.plot(episodes, self.success_rates, color='green', linewidth=2, label='Success Rate')
        ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Advance Threshold (80%)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate', color='green')
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Stage on secondary axis
        ax2_twin = ax2.twinx()
        ax2_twin.plot(episodes, self.stages, color='orange', linewidth=2, linestyle='--', label='Stage')
        ax2_twin.set_ylabel('Curriculum Stage', color='orange')
        ax2_twin.set_ylim(0, 5)
        ax2_twin.tick_params(axis='y', labelcolor='orange')
        ax2_twin.legend(loc='upper right')
        ax2.set_title('Success Rate & Curriculum Stage')
        
        # Plot 3: Episode Length
        ax3 = self.axes[1, 0]
        ax3.plot(episodes, self.lengths, alpha=0.3, color='purple', label='Raw')
        if len(self.lengths) >= 10:
            window = min(100, len(self.lengths))
            rolling = np.convolve(self.lengths, np.ones(window)/window, mode='valid')
            ax3.plot(range(window-1, len(self.lengths)), rolling, color='purple', linewidth=2, label=f'Avg ({window} ep)')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Length (steps)')
        ax3.set_title('Episode Length')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Training Time and Stats
        ax4 = self.axes[1, 1]
        ax4.axis('off')
        
        # Calculate stats
        elapsed_mins = (time.time() - self.start_time) / 60
        avg_reward = np.mean(self.rewards[-100:]) if len(self.rewards) >= 100 else np.mean(self.rewards)
        avg_length = np.mean(self.lengths[-100:]) if len(self.lengths) >= 100 else np.mean(self.lengths)
        current_stage = self.stages[-1] if self.stages else 1
        current_success = self.success_rates[-1] if self.success_rates else 0
        
        stage_names = ["", "Standing", "Balance", "Stepping", "Walking"]
        
        stats_text = f"""
        Training Statistics
        ═══════════════════════════════
        
        Total Episodes:     {len(self.rewards):,}
        Training Time:      {elapsed_mins:.1f} minutes
        
        Current Stage:      {current_stage} ({stage_names[current_stage]})
        Success Rate:       {current_success:.1%}
        
        Avg Reward (100 ep): {avg_reward:.2f}
        Avg Length (100 ep): {avg_length:.1f} steps
        
        Max Reward:         {max(self.rewards):.2f}
        Min Reward:         {min(self.rewards):.2f}
        """
        
        ax4.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                 verticalalignment='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        if save:
            self.fig.savefig(os.path.join(self.log_dir, 'training_progress.png'), dpi=150)
        
        plt.pause(0.01)  # Brief pause for display
    
    def save_data(self):
        """Save all logged data to CSV files."""
        import csv
        
        # Save episode data
        with open(os.path.join(self.log_dir, 'episode_data.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'reward', 'length', 'success_rate', 'stage', 'time'])
            for i, (r, l, s, st, t) in enumerate(zip(
                self.rewards, self.lengths, self.success_rates, self.stages, self.timestamps
            )):
                writer.writerow([i, r, l, s, st, t])
        
        print(f"📊 Data saved to {self.log_dir}/episode_data.csv")
    
    def close(self):
        """Close the plot."""
        if self.fig is not None:
            self.update_plot(save=True)
            plt.ioff()
            plt.close(self.fig)
        self.save_data()


class CurriculumCallback(BaseCallback):
    """
    Callback that integrates curriculum learning with PPO training.
    
    Features:
        - Updates curriculum stage based on success rate
        - Saves checkpoints at stage transitions
        - Logs to matplotlib-based logger
    """
    
    def __init__(
        self,
        curriculum: CurriculumScheduler,
        logger: TrainingLogger,
        checkpoint_dir: str = "./checkpoints",
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.curriculum = curriculum
        self.training_logger = logger
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Track episode info from vectorized envs
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        """Called at every step."""
        # Check for completed episodes
        for i, done in enumerate(self.locals['dones']):
            if done:
                # Get episode info
                info = self.locals['infos'][i]
                
                # Record in curriculum
                if 'termination_reason' in info:
                    curriculum_result = self.curriculum.record_episode(info)
                    
                    # Update environment's curriculum stage
                    env = self.training_env.envs[i]
                    if hasattr(env, 'env'):  # Wrapped env
                        env = env.env
                    if hasattr(env, 'curriculum_stage'):
                        env.curriculum_stage = self.curriculum.stage
                        env.target_velocity = self.curriculum.target_velocity
                    
                    # Log to training logger
                    episode_reward = info.get('episode', {}).get('r', 0)
                    episode_length = info.get('episode', {}).get('l', 0)
                    
                    self.training_logger.log_episode(
                        reward=episode_reward,
                        length=episode_length,
                        success_rate=self.curriculum.success_rate,
                        stage=self.curriculum.stage
                    )
                    
                    # Check for stage advancement
                    if curriculum_result['advanced']:
                        self._save_stage_checkpoint()
                        
                        # Update all environments
                        for env in self.training_env.envs:
                            if hasattr(env, 'env'):
                                env = env.env
                            if hasattr(env, 'curriculum_stage'):
                                env.curriculum_stage = self.curriculum.stage
                                env.target_velocity = self.curriculum.target_velocity
        
        return True
    
    def _save_stage_checkpoint(self):
        """Save checkpoint when stage advances."""
        stage_name = self.curriculum.stage_name
        path = os.path.join(
            self.checkpoint_dir,
            'curriculum',
            f'stage_{self.curriculum.stage - 1}_{stage_name}_complete.zip'
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"💾 Saved stage completion checkpoint: {path}")
    
    def _on_training_end(self):
        """Called when training ends."""
        self.curriculum.save_state()
        self.training_logger.close()


def make_env(rank: int, curriculum_stage: int = 1, target_velocity: float = 0.0, model_path: str = None):
    """
    Create a single environment instance.
    
    Args:
        rank: Environment index (for seeding)
        curriculum_stage: Starting curriculum stage
        target_velocity: Target walking velocity
        model_path: Path to model XML (for subprocess loading)
        
    Returns:
        Callable that creates an environment instance
    """
    def _init():
        # Import inside function for subprocess compatibility
        import sys
        import os
        
        # Add parent directory to path for imports
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        
        from go_bdx_env import GoBdxEnv
        
        env = GoBdxEnv(
            render_mode=None,  # No rendering during training
            curriculum_stage=curriculum_stage,
            target_velocity=target_velocity,
            randomize=True
        )
        env.reset(seed=rank * 1000)
        return env
    return _init


def train(
    total_timesteps: int = 2_000_000,
    n_envs: int = 8,
    start_stage: int = 1,
    resume_path: Optional[str] = None,
    save_dir: str = "./checkpoints",
    log_dir: str = "./logs",
    device: str = "auto"
):
    """
    Main training function.
    
    Args:
        total_timesteps: Total environment steps for training
        n_envs: Number of parallel environments
        start_stage: Starting curriculum stage (1-4)
        resume_path: Path to checkpoint to resume from
        save_dir: Directory for checkpoints
        log_dir: Directory for logs
        device: Training device ('cuda', 'cpu', or 'auto')
    """
    print("=" * 60)
    print("GO-BDX PPO Training")
    print("=" * 60)
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Parallel Envs:   {n_envs}")
    print(f"Starting Stage:  {start_stage}")
    print(f"Device:          {device}")
    print("=" * 60)
    
    # Check GPU availability
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  Using CPU (training will be slower)")
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize curriculum scheduler
    curriculum = CurriculumScheduler(
        start_stage=start_stage,
        checkpoint_dir=os.path.join(save_dir, "curriculum"),
        log_dir=log_dir
    )
    
    # Try to resume curriculum state
    curriculum.load_state()
    
    print(f"\n📚 Curriculum: Stage {curriculum.stage} ({curriculum.stage_name})")
    print(f"   Success Rate: {curriculum.success_rate:.1%}")
    
    # Create vectorized environment
    print(f"\n🏗️  Creating {n_envs} parallel environments...")
    
    env_fns = [
        make_env(i, curriculum.stage, curriculum.target_velocity) 
        for i in range(n_envs)
    ]
    
    # Use DummyVecEnv for MuJoCo compatibility (SubprocVecEnv has pickling issues)
    # DummyVecEnv runs envs sequentially but avoids multiprocessing problems
    vec_env = DummyVecEnv(env_fns)
    
    print(f"   Observation space: {vec_env.observation_space.shape}")
    print(f"   Action space: {vec_env.action_space.shape}")
    
    # Initialize training logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join(log_dir, f"run_{timestamp}")
    training_logger = TrainingLogger(run_log_dir)
    
    # PPO hyperparameters (optimized for bipedal locomotion)
    ppo_params = {
        # Core PPO
        "learning_rate": get_linear_fn(3e-4, 1e-5, 1.0),  # Linear decay
        "n_steps": 2048,          # Steps per update
        "batch_size": 256,        # Minibatch size
        "n_epochs": 10,           # Epochs per update
        "gamma": 0.99,            # Discount factor
        "gae_lambda": 0.95,       # GAE lambda
        "clip_range": 0.2,        # PPO clip range
        "clip_range_vf": 0.2,     # Value function clip
        "ent_coef": 0.005,        # Entropy bonus
        "vf_coef": 0.5,           # Value function coefficient
        "max_grad_norm": 0.5,     # Gradient clipping
        
        # Network architecture
        "policy_kwargs": dict(
            net_arch=dict(
                pi=[256, 256, 128],  # Policy network
                vf=[256, 256, 128]   # Value network
            ),
            activation_fn=torch.nn.ELU,  # ELU activation
        ),
        
        # Other
        "verbose": 1,
        "device": device,
    }
    
    # Create or load model
    if resume_path and os.path.exists(resume_path):
        print(f"\n📂 Resuming from checkpoint: {resume_path}")
        model = PPO.load(resume_path, env=vec_env, device=device)
    else:
        print("\n🆕 Creating new PPO model...")
        model = PPO("MlpPolicy", vec_env, **ppo_params)
    
    # Print model summary
    print("\n📊 Model Architecture:")
    print(f"   Policy: {ppo_params['policy_kwargs']['net_arch']['pi']}")
    print(f"   Value:  {ppo_params['policy_kwargs']['net_arch']['vf']}")
    print(f"   Activation: ELU")
    
    # Create callbacks
    curriculum_callback = CurriculumCallback(
        curriculum=curriculum,
        logger=training_logger,
        checkpoint_dir=save_dir
    )
    
    # Checkpoint callback (every 100k steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1000),
        save_path=os.path.join(save_dir, "periodic"),
        name_prefix="go_bdx",
        verbose=1
    )
    
    # Train!
    print("\n🚀 Starting training...")
    print("-" * 60)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[curriculum_callback, checkpoint_callback],
            progress_bar=True,
            reset_num_timesteps=resume_path is None
        )
    except KeyboardInterrupt:
        print("\n\n⏸️  Training interrupted by user")
    finally:
        # Save final model
        final_path = os.path.join(save_dir, "final_model.zip")
        model.save(final_path)
        print(f"\n💾 Final model saved: {final_path}")
        
        # Save curriculum state
        curriculum.save_state()
        
        # Close logger
        training_logger.close()
        
        # Close environment
        vec_env.close()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final Stage: {curriculum.stage} ({curriculum.stage_name})")
    print(f"Total Episodes: {curriculum.total_episodes}")
    print(f"Final Success Rate: {curriculum.success_rate:.1%}")
    print(f"Logs saved to: {run_log_dir}")
    print("=" * 60)


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description='Train GO-BDX to walk using PPO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start fresh training
    python train.py
    
    # Resume from checkpoint
    python train.py --resume checkpoints/periodic/go_bdx_500000_steps.zip
    
    # Train with more environments (faster, needs more RAM)
    python train.py --n-envs 12
    
    # Start from balance stage
    python train.py --start-stage 2
    
    # Train for longer
    python train.py --timesteps 5000000
"""
    )
    
    parser.add_argument(
        '--timesteps', '-t',
        type=int,
        default=2_000_000,
        help='Total training timesteps (default: 2M)'
    )
    parser.add_argument(
        '--n-envs', '-n',
        type=int,
        default=8,
        help='Number of parallel environments (default: 8)'
    )
    parser.add_argument(
        '--start-stage', '-s',
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help='Starting curriculum stage (1=standing, 2=balance, 3=stepping, 4=walking)'
    )
    parser.add_argument(
        '--resume', '-r',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./checkpoints',
        help='Directory for checkpoints'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='./logs',
        help='Directory for logs'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Training device (default: auto)'
    )
    
    args = parser.parse_args()
    
    train(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        start_stage=args.start_stage,
        resume_path=args.resume,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()
