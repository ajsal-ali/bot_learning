#!/usr/bin/env python3
"""
GO-BDX Training with Visual Rendering

Train PPO with N parallel environments, rendering one of them.
All N envs train in parallel, env[0] is visualized in MuJoCo viewer.

Usage:
    python train_visual.py
    python train_visual.py --n-envs 4 --timesteps 100000
"""

import os
import sys
import argparse
import numpy as np
import torch
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.callbacks import BaseCallback

from go_bdx_env import GoBdxEnv
from curriculum import CurriculumScheduler


class RenderCallback(BaseCallback):
    """
    Callback that renders a separate env and tracks training progress.
    """
    
    def __init__(self, render_env: GoBdxEnv, curriculum: CurriculumScheduler, verbose: int = 1):
        super().__init__(verbose)
        self.render_env = render_env
        self.curriculum = curriculum
        self.episode_count = 0
        self.episode_rewards = []
        self.current_ep_rewards = {}
        self.render_obs = None
        self.render_ep_reward = 0
        
    def _on_training_start(self):
        self.render_obs, _ = self.render_env.reset()
        
    def _on_step(self) -> bool:
        # Get action for render env using current policy
        action, _ = self.model.predict(self.render_obs, deterministic=False)
        
        # Step render env
        self.render_obs, reward, term, trunc, info = self.render_env.step(action)
        self.render_ep_reward += reward
        self.render_env.render()
        
        # Reset render env if done
        if term or trunc:
            self.render_obs, _ = self.render_env.reset()
            self.render_ep_reward = 0
        
        # Track training env episodes
        for i, done in enumerate(self.locals['dones']):
            if i not in self.current_ep_rewards:
                self.current_ep_rewards[i] = 0
            self.current_ep_rewards[i] += self.locals['rewards'][i]
            
            if done:
                info = self.locals['infos'][i]
                ep_reward = self.current_ep_rewards[i]
                self.episode_rewards.append(ep_reward)
                self.current_ep_rewards[i] = 0
                self.episode_count += 1
                
                # Record in curriculum
                if 'termination_reason' in info:
                    self.curriculum.record_episode(info)
                
                # Print every 10 episodes
                if self.episode_count % 10 == 0:
                    avg_rew = np.mean(self.episode_rewards[-100:])
                    reason = info.get('termination_reason', '?')
                    print(f"Ep {self.episode_count}: R={ep_reward:.1f}, "
                          f"Avg={avg_rew:.1f}, Success={self.curriculum.success_rate:.1%}, "
                          f"Stage={self.curriculum.stage} ({reason})")
        
        return True
    
    def _on_training_end(self):
        self.render_env.close()


def train_visual(
    timesteps: int = 500_000,
    n_envs: int = 4,
    start_stage: int = 1,
    save_dir: str = "./checkpoints",
):
    """
    Train with N parallel environments, visualizing env[0].
    """
    
    print("=" * 60)
    print("GO-BDX Parallel Training with Visualization")
    print("=" * 60)
    print(f"Timesteps: {timesteps:,}")
    print(f"Parallel Envs: {n_envs}")
    print(f"Stage: {start_stage}")
    print("=" * 60)
    
    # Create N training environments (no render)
    def make_env(rank):
        def _init():
            env = GoBdxEnv(
                render_mode=None,  # All training envs without render
                curriculum_stage=start_stage,
                max_episode_steps=250,
                randomize=True,
            )
            env.reset(seed=rank * 1000)
            return env
        return _init
    
    print(f"Creating {n_envs} training environments...")
    env_fns = [make_env(i) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    
    # Create separate render environment for visualization
    print("Creating render environment...")
    render_env = GoBdxEnv(
        render_mode="human",
        curriculum_stage=start_stage,
        max_episode_steps=250,
        randomize=True,
    )
    render_env.reset(seed=9999)
    
    print(f"Obs space: {vec_env.observation_space.shape}")
    print(f"Act space: {vec_env.action_space.shape}")
    
    # Curriculum
    curriculum = CurriculumScheduler(start_stage=start_stage)
    
    # PPO model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=1024,  # Smaller for more frequent updates
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        device=device,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.ELU,
        ),
    )
    
    # Callback for rendering and progress
    callback = RenderCallback(render_env, curriculum)
    
    print("=" * 60)
    print("Training started! Watch the MuJoCo viewer.")
    print("Press Ctrl+C to stop and save.")
    print("=" * 60)
    
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\n⏸️  Training interrupted!")
    finally:
        # Save
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "visual_trained.zip")
        model.save(path)
        print(f"\n💾 Model saved: {path}")
        
        curriculum.save_state()
        vec_env.close()
    
    print(f"\nFinal: {callback.episode_count} episodes")
    print(f"Success rate: {curriculum.success_rate:.1%}")
    print(f"Stage: {curriculum.stage} ({curriculum.stage_name})")


def main():
    parser = argparse.ArgumentParser(description='Train GO-BDX with visualization')
    parser.add_argument('--timesteps', '-t', type=int, default=500_000)
    parser.add_argument('--n-envs', '-n', type=int, default=4, help='Number of parallel envs')
    parser.add_argument('--stage', '-s', type=int, default=1, choices=[1,2,3,4])
    args = parser.parse_args()
    
    train_visual(timesteps=args.timesteps, n_envs=args.n_envs, start_stage=args.stage)


if __name__ == "__main__":
    main()
