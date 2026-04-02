#!/usr/bin/env python3
"""
GO-BDX Gymnasium Environment for Reinforcement Learning

This environment wraps the MuJoCo simulation of the GO-BDX bipedal robot
for training walking behaviors using PPO.

Author: Generated for RL walking training
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
from typing import Optional, Tuple, Dict, Any
import os


class GoBdxEnv(gym.Env):
    """
    GO-BDX Bipedal Robot Environment
    
    Observation Space (54 dims):
        - Body height (1)
        - Body orientation quaternion (4)
        - Body linear velocity (3)
        - Body angular velocity (3)
        - Leg joint positions (10)
        - Leg joint velocities (10)
        - Previous action t-1 (10)
        - Previous action t-2 (10)
        - Target velocity (1)
        - Gait phase sin/cos (2)
    
    Action Space (10 dims):
        - 10 leg joint position targets, normalized to [-1, 1]
        - Scaled by ACTION_SCALES before applying
    
    Reward:
        - Depends on curriculum stage (standing, balance, stepping, walking)
    
    Observation Noise (for sim2real robustness):
        - Joint positions: ±0.03 rad
        - Joint velocities: ±2.5 rad/s
        - Linear velocity: ±0.1 m/s
        - Angular velocity: ±0.1 rad/s
        - Height: ±0.01 m
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    # Mapping from RL action index to MuJoCo actuator index
    LEG_ACTUATOR_IDS = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
    
    # Action scaling - hip roll is limited to prevent falling sideways
    ACTION_SCALES = np.array([
        0.3,   # left_hip_roll  - LIMITED
        0.8,   # left_hip_pitch - main walking joint
        0.5,   # left_hip_yaw
        1.0,   # left_shin (knee)
        0.5,   # left_foot (ankle)
        0.3,   # right_hip_roll - LIMITED
        0.8,   # right_hip_pitch - main walking joint
        0.5,   # right_hip_yaw
        1.0,   # right_shin (knee)
        0.5,   # right_foot (ankle)
    ], dtype=np.float32)
    
    # Joint qpos addresses for leg joints
    JOINT_QPOS_ADDRS = [7, 8, 9, 10, 11, 17, 18, 19, 20, 21]
    
    # Joint qvel addresses for leg joints  
    JOINT_QVEL_ADDRS = [6, 7, 8, 9, 10, 16, 17, 18, 19, 20]
    
    # Physical constants (measured from simulation)
    TARGET_HEIGHT = 0.26        # Target CoM height when standing
    FALL_HEIGHT = 0.12          # Height below which robot is considered fallen
    MAX_TILT = 0.78             # ~45 degrees max roll/pitch before termination
    
    # Observation noise scales (for domain randomization / sim2real)
    NOISE_SCALES = {
        'joint_pos': 0.03,      # rad
        'joint_vel': 2.5,       # rad/s
        'lin_vel': 0.1,         # m/s
        'ang_vel': 0.1,         # rad/s
        'height': 0.01,         # m
        'quat': 0.05,           # quaternion components
    }
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 250,
        curriculum_stage: int = 1,
        target_velocity: float = 0.0,
        randomize: bool = True,
        obs_noise: bool = True,
    ):
        """
        Initialize the GO-BDX environment.
        
        Args:
            render_mode: "human" for visualization, None for training
            max_episode_steps: Maximum steps per episode (250 = 5 seconds at 50Hz)
            curriculum_stage: 1=standing, 2=balance, 3=stepping, 4=walking
            target_velocity: Target forward velocity for walking stage
            randomize: Whether to add random perturbations on reset
            obs_noise: Whether to add observation noise (for sim2real robustness)
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.curriculum_stage = curriculum_stage
        self.target_velocity = target_velocity
        self.randomize = randomize
        self.obs_noise = obs_noise
        
        # Load MuJoCo model
        model_path = os.path.join(os.path.dirname(__file__), "..", "go_bdx.xml")
        model_path = os.path.abspath(model_path)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}. Run convert_urdf.py first.")
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Control frequency: 50 Hz (step every 20ms = 10 physics steps at 500Hz)
        self.control_steps = 10  # Physics steps per control step
        self.dt = self.model.opt.timestep * self.control_steps  # 0.02 seconds
        
        # Get body IDs for foot contact detection
        self.left_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_link")
        self.right_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_foot_link")
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "floating_base")
        
        # Define observation and action spaces
        # Observation: 54 dimensions
        # 1 (height) + 4 (quat) + 3 (lin_vel) + 3 (ang_vel) + 10 (joint_pos) + 
        # 10 (joint_vel) + 10 (prev_action t-1) + 10 (prev_action t-2) + 1 (target_vel) + 2 (gait phase) = 54
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(54,), dtype=np.float32
        )
        
        # Action: 10 leg joints, normalized [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )
        
        # State tracking
        self.step_count = 0
        self.prev_action = np.zeros(10, dtype=np.float32)
        self.prev_prev_action = np.zeros(10, dtype=np.float32)
        self.gait_phase = 0.0
        self.episode_reward = 0.0
        
        # For balance stage - push tracking
        self.push_count = 0
        self.steps_since_push = 0
        self.was_pushed = False
        
        # Viewer for rendering
        self.viewer = None
        
    def _get_obs(self) -> np.ndarray:
        """Build observation vector from current state with optional noise."""
        obs = []
        
        # Body height (1) - with noise
        height = self.data.qpos[2]
        if self.obs_noise:
            height += np.random.uniform(-self.NOISE_SCALES['height'], self.NOISE_SCALES['height'])
        obs.append(height)
        
        # Body orientation quaternion (4) - wxyz, with noise
        quat = self.data.qpos[3:7].copy()
        if self.obs_noise:
            quat += np.random.uniform(-self.NOISE_SCALES['quat'], self.NOISE_SCALES['quat'], size=4)
            # Renormalize quaternion
            quat = quat / np.linalg.norm(quat)
        obs.extend(quat)
        
        # Body linear velocity (3) - with noise
        lin_vel = self.data.qvel[0:3].copy()
        if self.obs_noise:
            lin_vel += np.random.uniform(-self.NOISE_SCALES['lin_vel'], self.NOISE_SCALES['lin_vel'], size=3)
        obs.extend(lin_vel)
        
        # Body angular velocity (3) - with noise
        ang_vel = self.data.qvel[3:6].copy()
        if self.obs_noise:
            ang_vel += np.random.uniform(-self.NOISE_SCALES['ang_vel'], self.NOISE_SCALES['ang_vel'], size=3)
        obs.extend(ang_vel)
        
        # Leg joint positions (10) - with noise
        joint_pos = np.array([self.data.qpos[addr] for addr in self.JOINT_QPOS_ADDRS])
        if self.obs_noise:
            joint_pos += np.random.uniform(-self.NOISE_SCALES['joint_pos'], self.NOISE_SCALES['joint_pos'], size=10)
        obs.extend(joint_pos)
        
        # Leg joint velocities (10) - with noise
        joint_vel = np.array([self.data.qvel[addr] for addr in self.JOINT_QVEL_ADDRS])
        if self.obs_noise:
            joint_vel += np.random.uniform(-self.NOISE_SCALES['joint_vel'], self.NOISE_SCALES['joint_vel'], size=10)
        obs.extend(joint_vel)
        
        # Previous action t-1 (10)
        obs.extend(self.prev_action)
        
        # Previous action t-2 (10)
        obs.extend(self.prev_prev_action)
        
        # Target velocity (1)
        obs.append(self.target_velocity)
        
        # Gait phase as sin/cos (2) - for periodic gait
        obs.append(np.sin(self.gait_phase))
        obs.append(np.cos(self.gait_phase))
        
        return np.array(obs, dtype=np.float32)
    
    def _apply_action(self, action: np.ndarray):
        """Apply action to the robot with rate limiting."""
        # Action rate limiting - prevent violent jerks
        MAX_ACTION_CHANGE = 0.1  # Max 10% change per step
        action_clamped = np.clip(
            action, 
            self.prev_action - MAX_ACTION_CHANGE, 
            self.prev_action + MAX_ACTION_CHANGE
        )
        
        # Scale action
        scaled_action = action_clamped * self.ACTION_SCALES
        
        # Zero all controls first (head/neck stay at 0)
        self.data.ctrl[:] = 0
        
        # Apply to leg actuators
        for i, act_id in enumerate(self.LEG_ACTUATOR_IDS):
            self.data.ctrl[act_id] = scaled_action[i]
    
    def _get_foot_contact(self, foot: str) -> bool:
        """Check if foot is in contact with ground."""
        foot_id = self.left_foot_id if foot == "left" else self.right_foot_id
        foot_z = self.data.xpos[foot_id][2]
        # Contact if foot is close to ground (within 3cm)
        return foot_z < 0.03
    
    def _get_foot_height(self, foot: str) -> float:
        """Get foot height above ground."""
        foot_id = self.left_foot_id if foot == "left" else self.right_foot_id
        return self.data.xpos[foot_id][2]
    
    def _quat_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion (wxyz) to euler angles (roll, pitch, yaw)."""
        w, x, y, z = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    def _compute_reward(self) -> float:
        """Compute reward based on curriculum stage."""
        if self.curriculum_stage == 1:
            return self._standing_reward()
        elif self.curriculum_stage == 2:
            return self._balance_reward()
        elif self.curriculum_stage == 3:
            return self._stepping_reward()
        else:
            return self._walking_reward()
    
    def _standing_reward(self) -> float:
        """
        Stage 1: Standing Reward
        Goal: Stay upright without falling
        """
        # Height reward: Gaussian centered at target height
        height = self.data.qpos[2]
        height_reward = np.exp(-50 * (height - self.TARGET_HEIGHT) ** 2)
        
        # Upright reward: penalize roll and pitch
        quat = self.data.qpos[3:7]
        roll, pitch, yaw = self._quat_to_euler(quat)
        upright_reward = np.exp(-5 * (roll ** 2 + pitch ** 2))
        
        # Stillness reward: penalize body velocity
        lin_vel = np.linalg.norm(self.data.qvel[0:3])
        ang_vel = np.linalg.norm(self.data.qvel[3:6])
        stillness_reward = np.exp(-2 * (lin_vel + ang_vel))
        
        # Action penalty: minimize control effort
        action_penalty = 0.01 * np.sum(self.prev_action ** 2)
        
        # Combine with weights
        reward = (
            0.4 * height_reward +
            0.4 * upright_reward +
            0.2 * stillness_reward -
            action_penalty
        )
        return reward
    
    def _balance_reward(self) -> float:
        """
        Stage 2: Balance Reward
        Goal: Recover from pushes while standing
        """
        base_reward = self._standing_reward()
        
        # Recovery bonus if we were pushed but still standing
        if self.was_pushed and self.data.qpos[2] > 0.20:
            recovery_bonus = 0.3
            self.was_pushed = False
        else:
            recovery_bonus = 0.0
        
        return base_reward + recovery_bonus
    
    def _stepping_reward(self) -> float:
        """
        Stage 3: Stepping in Place Reward
        Goal: Alternate foot contacts with proper timing
        """
        # Get foot contacts
        left_contact = self._get_foot_contact("left")
        right_contact = self._get_foot_contact("right")
        
        # Desired pattern based on gait phase:
        # phase 0-π: left swing (up), right stance (down)
        # phase π-2π: left stance (down), right swing (up)
        phase_mod = self.gait_phase % (2 * np.pi)
        
        if phase_mod < np.pi:
            # Left should be swinging (no contact), right should be stance (contact)
            desired_left = False
            desired_right = True
        else:
            # Left stance, right swing
            desired_left = True
            desired_right = False
        
        # Contact accuracy reward
        left_correct = float(left_contact == desired_left)
        right_correct = float(right_contact == desired_right)
        contact_reward = 0.5 * left_correct + 0.5 * right_correct
        
        # Foot clearance reward for swing foot
        if not desired_left:  # Left is swinging
            clearance = self._get_foot_height("left")
            clearance_reward = np.clip(clearance / 0.05, 0, 1)  # Target 5cm lift
        else:  # Right is swinging
            clearance = self._get_foot_height("right")
            clearance_reward = np.clip(clearance / 0.05, 0, 1)
        
        # Height maintenance
        height = self.data.qpos[2]
        height_reward = np.exp(-50 * (height - self.TARGET_HEIGHT) ** 2)
        
        # Upright bonus
        quat = self.data.qpos[3:7]
        roll, pitch, yaw = self._quat_to_euler(quat)
        upright_reward = np.exp(-5 * (roll ** 2 + pitch ** 2))
        
        # Action smoothness penalty
        if hasattr(self, '_prev_prev_action'):
            jerk = self.prev_action - self._prev_prev_action
            smooth_penalty = 0.01 * np.sum(jerk ** 2)
        else:
            smooth_penalty = 0
        
        reward = (
            0.25 * height_reward +
            0.25 * upright_reward +
            0.30 * contact_reward +
            0.20 * clearance_reward -
            smooth_penalty
        )
        return reward
    
    def _walking_reward(self) -> float:
        """
        Stage 4: Walking Reward
        Goal: Move forward at target velocity
        """
        # Forward velocity tracking
        actual_vel = self.data.qvel[0]  # X velocity
        vel_error = np.abs(actual_vel - self.target_velocity)
        vel_reward = np.exp(-4 * vel_error)
        
        # Height maintenance
        height = self.data.qpos[2]
        height_reward = np.exp(-50 * (height - self.TARGET_HEIGHT) ** 2)
        
        # Upright reward
        quat = self.data.qpos[3:7]
        roll, pitch, yaw = self._quat_to_euler(quat)
        upright_reward = np.exp(-5 * (roll ** 2 + pitch ** 2))
        
        # Foot contact reward (same as stepping)
        left_contact = self._get_foot_contact("left")
        right_contact = self._get_foot_contact("right")
        # At least one foot should be on ground
        contact_reward = 1.0 if (left_contact or right_contact) else 0.0
        # Bonus for alternating
        if left_contact != right_contact:
            contact_reward += 0.5
        
        # Energy efficiency penalty
        energy_penalty = 0.001 * np.sum(self.prev_action ** 2)
        
        # Action smoothness penalty
        if hasattr(self, '_prev_prev_action'):
            jerk = self.prev_action - self._prev_prev_action
            smooth_penalty = 0.005 * np.sum(jerk ** 2)
        else:
            smooth_penalty = 0
        
        # Lateral velocity penalty (don't drift sideways)
        lateral_vel = np.abs(self.data.qvel[1])
        lateral_penalty = 0.1 * lateral_vel
        
        reward = (
            0.35 * vel_reward +
            0.20 * height_reward +
            0.15 * upright_reward +
            0.15 * contact_reward -
            energy_penalty -
            smooth_penalty -
            lateral_penalty
        )
        return reward
    
    def _check_termination(self) -> Tuple[bool, bool, str]:
        """
        Check if episode should terminate.
        
        Returns:
            terminated: True if failed (fell, tilted)
            truncated: True if timeout (success for standing)
            reason: String describing termination reason
        """
        terminated = False
        truncated = False
        reason = ""
        
        # Fall detection
        height = self.data.qpos[2]
        if height < self.FALL_HEIGHT:
            terminated = True
            reason = "fallen"
            return terminated, truncated, reason
        
        # Excessive tilt detection
        quat = self.data.qpos[3:7]
        roll, pitch, yaw = self._quat_to_euler(quat)
        if abs(roll) > self.MAX_TILT or abs(pitch) > self.MAX_TILT:
            terminated = True
            reason = "tilted"
            return terminated, truncated, reason
        
        # Timeout (success for standing/balance stages)
        if self.step_count >= self.max_episode_steps:
            truncated = True
            reason = "timeout"
            return terminated, truncated, reason
        
        return terminated, truncated, reason
    
    def _maybe_apply_push(self):
        """Apply random push for balance training (stage 2)."""
        if self.curriculum_stage != 2:
            return
        
        self.steps_since_push += 1
        
        # Apply push every 50-100 steps
        if self.steps_since_push > np.random.randint(50, 100):
            # Random push force
            push_force = np.random.uniform(-30, 30, size=3)
            push_force[2] = 0  # No vertical push
            
            # Apply to base body
            self.data.xfrc_applied[self.base_id, :3] = push_force
            
            self.was_pushed = True
            self.push_count += 1
            self.steps_since_push = 0
        else:
            # Clear force
            self.data.xfrc_applied[self.base_id, :] = 0
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take one environment step.
        
        Args:
            action: 10-dim action array, normalized [-1, 1]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Store previous actions for action history
        self.prev_prev_action = self.prev_action.copy()
        self.prev_action = action.copy()
        
        # Apply action
        self._apply_action(action)
        
        # Maybe apply push (balance stage)
        self._maybe_apply_push()
        
        # Step physics multiple times (control at 50Hz, physics at 500Hz)
        for _ in range(self.control_steps):
            mujoco.mj_step(self.model, self.data)
        
        # Update gait phase (1 second period)
        self.gait_phase += 2 * np.pi * self.dt
        
        # Compute reward
        reward = self._compute_reward()
        self.episode_reward += reward
        
        # Check termination
        terminated, truncated, reason = self._check_termination()
        
        # Increment step counter
        self.step_count += 1
        
        # Build info dict
        info = {
            "step": self.step_count,
            "height": self.data.qpos[2],
            "velocity": self.data.qvel[0],
            "termination_reason": reason,
            "curriculum_stage": self.curriculum_stage,
            "episode_reward": self.episode_reward,
        }
        
        # Stage-specific info
        if self.curriculum_stage == 2:
            info["push_count"] = self.push_count
            info["survived_pushes"] = self.push_count if not terminated else self.push_count - 1
        
        if self.curriculum_stage >= 3:
            left_contact = self._get_foot_contact("left")
            right_contact = self._get_foot_contact("right")
            info["left_contact"] = left_contact
            info["right_contact"] = right_contact
        
        # Get observation
        obs = self._get_obs()
        
        return obs, reward, terminated, truncated, info
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Optional dict with 'curriculum_stage' or 'target_velocity'
        """
        super().reset(seed=seed)
        
        # Handle options
        if options:
            if 'curriculum_stage' in options:
                self.curriculum_stage = options['curriculum_stage']
            if 'target_velocity' in options:
                self.target_velocity = options['target_velocity']
        
        # Reset MuJoCo
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial pose
        self.data.qpos[2] = self.TARGET_HEIGHT  # Height
        self.data.qpos[3:7] = [1, 0, 0, 0]  # Upright quaternion
        
        # Slightly bent knees for stability
        initial_pose = {
            7: 0.0,    # left_hip_roll
            8: 0.2,    # left_hip_pitch
            9: 0.0,    # left_hip_yaw
            10: -0.4,  # left_shin
            11: 0.2,   # left_foot
            17: 0.0,   # right_hip_roll
            18: 0.2,   # right_hip_pitch
            19: 0.0,   # right_hip_yaw
            20: -0.4,  # right_shin
            21: 0.2,   # right_foot
        }
        
        for addr, angle in initial_pose.items():
            self.data.qpos[addr] = angle
        
        # Add randomization
        if self.randomize:
            self.data.qpos[2] += np.random.uniform(-0.02, 0.02)
            # Small random joint perturbations
            for addr in self.JOINT_QPOS_ADDRS:
                self.data.qpos[addr] += np.random.uniform(-0.05, 0.05)
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        # Reset state
        self.step_count = 0
        self.prev_action = np.zeros(10, dtype=np.float32)
        self.prev_prev_action = np.zeros(10, dtype=np.float32)
        self.gait_phase = 0.0
        self.episode_reward = 0.0
        self.push_count = 0
        self.steps_since_push = 0
        self.was_pushed = False
        
        # Set initial control to hold position
        self._apply_action(np.zeros(10))
        
        obs = self._get_obs()
        info = {"curriculum_stage": self.curriculum_stage}
        
        return obs, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.cam.distance = 1.5
                self.viewer.cam.elevation = -20
                self.viewer.cam.azimuth = 135
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            # TODO: Implement offscreen rendering
            pass
    
    def close(self):
        """Close the environment."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    def set_curriculum_stage(self, stage: int):
        """Set the curriculum stage."""
        self.curriculum_stage = stage
    
    def set_target_velocity(self, velocity: float):
        """Set target velocity for walking stage."""
        self.target_velocity = velocity


# Register the environment with Gymnasium
gym.register(
    id="GoBdx-v0",
    entry_point="rl.go_bdx_env:GoBdxEnv",
    max_episode_steps=250,
)


if __name__ == "__main__":
    # Test the environment
    print("Testing GoBdxEnv...")
    
    env = GoBdxEnv(render_mode="human", curriculum_stage=1)
    obs, info = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Action shape: {env.action_space.shape}")
    
    # Run a few steps with random actions
    total_reward = 0
    for i in range(500):
        action = env.action_space.sample() * 0.1  # Small random actions
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        
        if terminated or truncated:
            print(f"Episode ended at step {i}: {info['termination_reason']}")
            break
    
    print(f"Total reward: {total_reward:.2f}")
    env.close()
