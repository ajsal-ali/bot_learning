#!/usr/bin/env python3
"""
Matplotlib-based training logger for GO-BDX RL.
No TensorFlow/TensorBoard dependency.
"""

import os
import json
import matplotlib
matplotlib.use("Agg")  # non-interactive backend by default
import matplotlib.pyplot as plt
from typing import Optional


class TrainingLogger:

    def __init__(self, log_dir: str = "./logs", live_plot: bool = False):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.steps = []
        self.rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.stages = []
        self.heights = []
        self.velocities = []

        self._recent_successes = []

        if live_plot:
            matplotlib.use("TkAgg")
            plt.ion()

        self.fig = None
        self.axes = None

    def log(
        self,
        step: int,
        reward: float,
        episode_length: int,
        success: bool,
        stage: int,
        height: float = 0.0,
        velocity: float = 0.0,
    ):
        self.steps.append(step)
        self.rewards.append(reward)
        self.episode_lengths.append(episode_length)
        self.stages.append(stage)
        self.heights.append(height)
        self.velocities.append(velocity)

        self._recent_successes.append(1 if success else 0)
        if len(self._recent_successes) > 100:
            self._recent_successes.pop(0)
        self.success_rates.append(
            sum(self._recent_successes) / len(self._recent_successes)
        )

    def save_logs(self):
        data = {
            "steps": self.steps,
            "rewards": self.rewards,
            "episode_lengths": self.episode_lengths,
            "success_rates": self.success_rates,
            "stages": self.stages,
        }
        with open(os.path.join(self.log_dir, "training_log.json"), "w") as f:
            json.dump(data, f)

    def update_plots(self, save: bool = True):
        if len(self.steps) < 2:
            return

        if self.fig is None:
            self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 9))

        for ax in self.axes.flat:
            ax.clear()

        # Reward
        ax = self.axes[0, 0]
        ax.plot(self.steps, self.rewards, "b-", alpha=0.2, linewidth=0.5)
        if len(self.rewards) > 50:
            window = min(100, len(self.rewards))
            rolling = [
                sum(self.rewards[max(0, i - window):i]) / min(i, window)
                for i in range(1, len(self.rewards) + 1)
            ]
            ax.plot(self.steps, rolling, "b-", linewidth=2, label="rolling avg")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Episode Reward")
        ax.set_title("Training Reward")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Episode length
        ax = self.axes[0, 1]
        ax.plot(self.steps, self.episode_lengths, "g-", alpha=0.3, linewidth=0.5)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Episode Length")
        ax.set_title("Episode Duration")
        ax.grid(True, alpha=0.3)

        # Success rate
        ax = self.axes[1, 0]
        ax.plot(self.steps, self.success_rates, "r-", linewidth=1.5)
        ax.axhline(y=0.8, color="k", linestyle="--", alpha=0.5, label="80% threshold")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Success Rate (last 100)")
        ax.set_title("Success Rate")
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Curriculum stage
        ax = self.axes[1, 1]
        ax.plot(self.steps, self.stages, "m-", linewidth=2)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Stage")
        ax.set_title("Curriculum Stage")
        ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels(["Standing", "Balance", "Stepping", "Walking"])
        ax.set_ylim([0.5, 4.5])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(
                os.path.join(self.log_dir, "training_curves.png"), dpi=150
            )

        try:
            plt.pause(0.01)
        except Exception:
            pass

    def close(self):
        self.save_logs()
        self.update_plots(save=True)
        plt.close("all")
