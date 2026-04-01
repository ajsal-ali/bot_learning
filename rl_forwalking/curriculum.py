#!/usr/bin/env python3
"""
Curriculum scheduler for GO-BDX RL walking.

Tracks success rates and auto-advances through 4 stages:
    1. Standing -> 2. Balance -> 3. Stepping -> 4. Walking
"""

from collections import deque
from typing import Dict, Any, Optional, Callable
import json
import os

from . import config as C


class CurriculumScheduler:

    def __init__(
        self,
        start_stage: int = 1,
        checkpoint_dir: str = "./checkpoints/curriculum",
        log_dir: str = "./logs",
    ):
        self.stage = start_stage
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.success_history: deque = deque(maxlen=C.SUCCESS_HISTORY_LEN)
        self.total_episodes = 0
        self.stage_episodes = 0

        # Callbacks (set externally)
        self.on_stage_advance: Optional[Callable[[int, int], None]] = None
        self.save_checkpoint_fn: Optional[Callable[[str], None]] = None

        # Walking velocity progression
        self.velocity_idx = 0

    # ----------------------------------------------------------
    # Properties
    # ----------------------------------------------------------
    @property
    def stage_name(self) -> str:
        return C.CURRICULUM_STAGES.get(self.stage, "unknown")

    @property
    def success_rate(self) -> float:
        if not self.success_history:
            return 0.0
        return sum(self.success_history) / len(self.success_history)

    @property
    def target_velocity(self) -> float:
        if self.stage < 4:
            return 0.0
        idx = min(self.velocity_idx, len(C.WALKING_VELOCITIES) - 1)
        return C.WALKING_VELOCITIES[idx]

    # ----------------------------------------------------------
    # Episode evaluation
    # ----------------------------------------------------------
    def is_success(self, info: Dict[str, Any]) -> bool:
        reason = info.get("termination_reason", "")
        if self.stage == 1:
            return reason == "timeout"
        elif self.stage == 2:
            return reason == "timeout" and info.get("survived_pushes", 0) >= 3
        elif self.stage == 3:
            return reason == "timeout"
        else:
            return reason == "timeout" and info.get("avg_velocity_error", 1.0) < 0.15

    def record_episode(self, info: Dict[str, Any]) -> Dict[str, Any]:
        self.total_episodes += 1
        self.stage_episodes += 1

        success = self.is_success(info)
        self.success_history.append(1 if success else 0)

        advanced = False
        if self._should_advance():
            advanced = self._advance()

        return {
            "stage": self.stage,
            "stage_name": self.stage_name,
            "success": success,
            "success_rate": self.success_rate,
            "episodes_in_stage": self.stage_episodes,
            "total_episodes": self.total_episodes,
            "advanced": advanced,
            "target_velocity": self.target_velocity,
        }

    # ----------------------------------------------------------
    # Stage transitions
    # ----------------------------------------------------------
    def _should_advance(self) -> bool:
        if len(self.success_history) < C.SUCCESS_HISTORY_LEN:
            return False
        if self.success_rate < C.SUCCESS_THRESHOLD:
            return False
        if self.stage >= 4:
            if self.velocity_idx < len(C.WALKING_VELOCITIES) - 1:
                self.velocity_idx += 1
                self.success_history.clear()
                print(f"[Curriculum] Increased target velocity to {self.target_velocity:.1f} m/s")
            return False
        return True

    def _advance(self) -> bool:
        old = self.stage

        if self.save_checkpoint_fn:
            self.save_checkpoint_fn(f"{self.stage_name}_complete")

        self.stage += 1
        self.success_history.clear()
        self.stage_episodes = 0

        print(f"[Curriculum] Advanced: Stage {old} ({C.CURRICULUM_STAGES[old]}) "
              f"-> Stage {self.stage} ({self.stage_name})")

        if self.on_stage_advance:
            self.on_stage_advance(old, self.stage)

        return True

    # ----------------------------------------------------------
    # Env options helper
    # ----------------------------------------------------------
    def get_env_options(self) -> Dict[str, Any]:
        return {
            "curriculum_stage": self.stage,
            "target_velocity": self.target_velocity,
        }

    # ----------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------
    def save_state(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(self.log_dir, "curriculum_state.json")
        state = {
            "stage": self.stage,
            "total_episodes": self.total_episodes,
            "stage_episodes": self.stage_episodes,
            "success_history": list(self.success_history),
            "velocity_idx": self.velocity_idx,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(self.log_dir, "curriculum_state.json")
        if not os.path.exists(path):
            print(f"[Curriculum] No saved state at {path}")
            return
        with open(path, "r") as f:
            state = json.load(f)
        self.stage = state["stage"]
        self.total_episodes = state["total_episodes"]
        self.stage_episodes = state["stage_episodes"]
        self.success_history = deque(state["success_history"], maxlen=C.SUCCESS_HISTORY_LEN)
        self.velocity_idx = state.get("velocity_idx", 0)
        print(f"[Curriculum] Loaded: Stage {self.stage} ({self.stage_name}), "
              f"success={self.success_rate:.0%}")

    def status_str(self) -> str:
        return (f"Stage {self.stage}/4 ({self.stage_name}) | "
                f"Success: {self.success_rate:.0%} "
                f"({len(self.success_history)}/{C.SUCCESS_HISTORY_LEN}) | "
                f"Ep: {self.stage_episodes} (total {self.total_episodes})")
