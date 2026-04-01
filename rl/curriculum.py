#!/usr/bin/env python3
"""
Curriculum Scheduler for GO-BDX RL Training

Manages progression through training stages based on success rate.
"""

from collections import deque
from typing import Dict, Any, Optional, Callable
import json
import os


class CurriculumScheduler:
    """
    Manages curriculum learning for GO-BDX walking training.
    
    Stages:
        1. Standing  - Stay upright without falling
        2. Balance   - Recover from random pushes
        3. Stepping  - Alternate foot contacts in place
        4. Walking   - Forward locomotion at target velocity
    
    Transitions:
        - Track last 100 episodes
        - Advance when success rate >= 80%
        - Save checkpoint at each stage completion
    """
    
    STAGE_NAMES = ["", "standing", "balance", "stepping", "walking"]
    SUCCESS_THRESHOLD = 0.80  # 80% success rate to advance
    HISTORY_LENGTH = 100
    
    def __init__(
        self,
        start_stage: int = 1,
        checkpoint_dir: str = "./checkpoints/curriculum",
        log_dir: str = "./logs"
    ):
        """
        Initialize curriculum scheduler.
        
        Args:
            start_stage: Starting curriculum stage (1-4)
            checkpoint_dir: Directory for curriculum checkpoints
            log_dir: Directory for logs
        """
        self.stage = start_stage
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Success history for current stage
        self.success_history = deque(maxlen=self.HISTORY_LENGTH)
        
        # Overall statistics
        self.total_episodes = 0
        self.stage_episodes = 0
        self.stage_start_episode = 0
        
        # Callbacks
        self.on_stage_advance: Optional[Callable[[int, int], None]] = None
        self.save_checkpoint_fn: Optional[Callable[[str], None]] = None
        
        # Target velocity schedule for walking stage
        self.walking_velocities = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.current_velocity_idx = 0
        
    @property
    def stage_name(self) -> str:
        """Get current stage name."""
        return self.STAGE_NAMES[self.stage]
    
    @property
    def success_rate(self) -> float:
        """Get current success rate."""
        if len(self.success_history) == 0:
            return 0.0
        return sum(self.success_history) / len(self.success_history)
    
    @property
    def target_velocity(self) -> float:
        """Get current target velocity for walking stage."""
        if self.stage < 4:
            return 0.0
        return self.walking_velocities[min(self.current_velocity_idx, len(self.walking_velocities) - 1)]
    
    def is_episode_success(self, info: Dict[str, Any]) -> bool:
        """
        Determine if episode was successful based on current stage.
        
        Args:
            info: Episode info dict from environment
            
        Returns:
            True if episode was successful
        """
        reason = info.get("termination_reason", "")
        
        if self.stage == 1:  # Standing
            # Success if episode timed out (didn't fall)
            return reason == "timeout"
        
        elif self.stage == 2:  # Balance
            # Success if survived at least 3 pushes without falling
            survived = info.get("survived_pushes", 0)
            return reason == "timeout" and survived >= 3
        
        elif self.stage == 3:  # Stepping
            # Success if maintained good contact pattern
            # (This would need contact_accuracy tracking in env)
            return reason == "timeout"
        
        else:  # Walking
            # Success if maintained target velocity within tolerance
            avg_vel_error = info.get("avg_velocity_error", 1.0)
            return reason == "timeout" and avg_vel_error < 0.15
    
    def record_episode(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record episode result and check for stage advancement.
        
        Args:
            info: Episode info dict from environment
            
        Returns:
            Dict with curriculum status
        """
        self.total_episodes += 1
        self.stage_episodes += 1
        
        # Determine success
        success = self.is_episode_success(info)
        self.success_history.append(1 if success else 0)
        
        # Check for stage advancement
        advanced = False
        if self._should_advance():
            advanced = self._advance_stage()
        
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
    
    def _should_advance(self) -> bool:
        """Check if should advance to next stage."""
        # Need at least HISTORY_LENGTH episodes
        if len(self.success_history) < self.HISTORY_LENGTH:
            return False
        
        # Check success rate
        if self.success_rate < self.SUCCESS_THRESHOLD:
            return False
        
        # Don't advance past stage 4
        if self.stage >= 4:
            # But can increase velocity target
            if self.current_velocity_idx < len(self.walking_velocities) - 1:
                self.current_velocity_idx += 1
                self.success_history.clear()
                print(f"📈 Increased target velocity to {self.target_velocity:.1f} m/s")
            return False
        
        return True
    
    def _advance_stage(self) -> bool:
        """Advance to next curriculum stage."""
        old_stage = self.stage
        
        # Save checkpoint for completed stage
        if self.save_checkpoint_fn:
            checkpoint_name = f"{self.stage_name}_complete"
            self.save_checkpoint_fn(checkpoint_name)
            print(f"💾 Saved checkpoint: {checkpoint_name}")
        
        # Advance stage
        self.stage += 1
        self.success_history.clear()
        self.stage_start_episode = self.total_episodes
        self.stage_episodes = 0
        
        print(f"🎉 Advanced to Stage {self.stage}: {self.stage_name}")
        print(f"   (Success rate was {self.success_rate:.1%} over {self.HISTORY_LENGTH} episodes)")
        
        # Call callback if set
        if self.on_stage_advance:
            self.on_stage_advance(old_stage, self.stage)
        
        return True
    
    def get_env_options(self) -> Dict[str, Any]:
        """Get environment options for current stage."""
        return {
            "curriculum_stage": self.stage,
            "target_velocity": self.target_velocity,
        }
    
    def save_state(self, path: Optional[str] = None):
        """Save curriculum state to file."""
        if path is None:
            path = os.path.join(self.log_dir, "curriculum_state.json")
        
        state = {
            "stage": self.stage,
            "total_episodes": self.total_episodes,
            "stage_episodes": self.stage_episodes,
            "success_history": list(self.success_history),
            "current_velocity_idx": self.current_velocity_idx,
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, path: Optional[str] = None):
        """Load curriculum state from file."""
        if path is None:
            path = os.path.join(self.log_dir, "curriculum_state.json")
        
        if not os.path.exists(path):
            print(f"No curriculum state found at {path}")
            return
        
        with open(path, 'r') as f:
            state = json.load(f)
        
        self.stage = state["stage"]
        self.total_episodes = state["total_episodes"]
        self.stage_episodes = state["stage_episodes"]
        self.success_history = deque(state["success_history"], maxlen=self.HISTORY_LENGTH)
        self.current_velocity_idx = state.get("current_velocity_idx", 0)
        
        print(f"Loaded curriculum state: Stage {self.stage} ({self.stage_name})")
        print(f"  Episodes: {self.total_episodes}, Success rate: {self.success_rate:.1%}")
    
    def status_string(self) -> str:
        """Get formatted status string."""
        return (
            f"Stage {self.stage}/{4} ({self.stage_name}) | "
            f"Success: {self.success_rate:.1%} ({len(self.success_history)}/{self.HISTORY_LENGTH}) | "
            f"Episodes: {self.stage_episodes} (total: {self.total_episodes})"
        )


if __name__ == "__main__":
    # Test curriculum scheduler
    print("Testing CurriculumScheduler...")
    
    scheduler = CurriculumScheduler(start_stage=1)
    
    # Simulate some episodes
    for i in range(150):
        # Simulate success rate increasing over time
        success = i > 50 and (i % 5 != 0)  # ~80% success after episode 50
        
        info = {
            "termination_reason": "timeout" if success else "fallen",
            "survived_pushes": 3 if success else 0,
        }
        
        result = scheduler.record_episode(info)
        
        if i % 20 == 0 or result["advanced"]:
            print(f"Episode {i}: {scheduler.status_string()}")
    
    print(f"\nFinal: {scheduler.status_string()}")
