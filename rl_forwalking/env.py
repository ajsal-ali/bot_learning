#!/usr/bin/env python3
"""
GO-BDX Gymnasium Environment for RL walking training.

Observation (43 dims):
    body height, orientation quat, linear vel, angular vel,
    joint positions, joint velocities, prev action, target vel, gait phase sin/cos

Action (10 dims):
    leg joint position targets normalized to [-1, 1]
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from typing import Optional, Tuple, Dict, Any

from . import config as C
from . import rewards as R


class GoBdxWalkingEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": C.CONTROL_FREQ}

    def __init__(
        self,
        model_path: str,
        render_mode: Optional[str] = None,
        curriculum_stage: int = 1,
        target_velocity: float = 0.0,
        max_episode_steps: int = C.MAX_EPISODE_STEPS,
        randomize: bool = True,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.curriculum_stage = curriculum_stage
        self.target_velocity = target_velocity
        self.max_episode_steps = max_episode_steps
        self.randomize = randomize

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.dt = self.model.opt.timestep * C.PHYSICS_STEPS_PER_CTRL

        # Body IDs
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, C.BODY_BASE)
        self.left_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, C.BODY_LEFT_FOOT)
        self.right_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, C.BODY_RIGHT_FOOT)

        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(C.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(C.ACT_DIM,), dtype=np.float32
        )

        # State
        self.step_count = 0
        self.prev_action = np.zeros(C.ACT_DIM, dtype=np.float32)
        self.prev_prev_action = np.zeros(C.ACT_DIM, dtype=np.float32)
        self.gait_phase = 0.0
        self.episode_reward = 0.0

        # Balance stage
        self.push_count = 0
        self.steps_since_push = 0
        self.was_pushed = False

        # Viewer
        self.viewer = None

    # ----------------------------------------------------------
    # Observation
    # ----------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(C.OBS_DIM, dtype=np.float32)
        idx = 0

        # Body height (1)
        obs[idx] = self.data.qpos[2]; idx += 1

        # Body orientation quaternion wxyz (4)
        obs[idx:idx+4] = self.data.qpos[3:7]; idx += 4

        # Body linear velocity (3)
        obs[idx:idx+3] = self.data.qvel[0:3]; idx += 3

        # Body angular velocity (3)
        obs[idx:idx+3] = self.data.qvel[3:6]; idx += 3

        # Leg joint positions (10)
        for i, addr in enumerate(C.JOINT_QPOS_ADDRS):
            obs[idx+i] = self.data.qpos[addr]
        idx += 10

        # Leg joint velocities (10)
        for i, addr in enumerate(C.JOINT_QVEL_ADDRS):
            obs[idx+i] = self.data.qvel[addr]
        idx += 10

        # Previous action (10)
        obs[idx:idx+10] = self.prev_action; idx += 10

        # Target velocity (1)
        obs[idx] = self.target_velocity; idx += 1

        # Gait phase sin/cos (2)
        obs[idx] = np.sin(self.gait_phase); idx += 1
        obs[idx] = np.cos(self.gait_phase); idx += 1

        return obs

    # ----------------------------------------------------------
    # Action application
    # ----------------------------------------------------------
    def _apply_action(self, action: np.ndarray):
        scaled = action * C.ACTION_SCALES
        self.data.ctrl[:] = 0.0
        for i, act_id in enumerate(C.LEG_ACTUATOR_IDS):
            self.data.ctrl[act_id] = scaled[i]

    # ----------------------------------------------------------
    # Foot helpers
    # ----------------------------------------------------------
    def _foot_contact(self, side: str) -> bool:
        fid = self.left_foot_id if side == "left" else self.right_foot_id
        return self.data.xpos[fid][2] < C.FOOT_CONTACT_THRESHOLD

    def _foot_height(self, side: str) -> float:
        fid = self.left_foot_id if side == "left" else self.right_foot_id
        return float(self.data.xpos[fid][2])

    # ----------------------------------------------------------
    # Quaternion to euler
    # ----------------------------------------------------------
    @staticmethod
    def _quat_to_rpy(quat: np.ndarray):
        w, x, y, z = quat
        sinr = 2.0 * (w * x + y * z)
        cosr = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr, cosr)

        sinp = 2.0 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny, cosy)

        return roll, pitch, yaw

    # ----------------------------------------------------------
    # Rewards (delegated to rewards.py)
    # ----------------------------------------------------------
    def _compute_reward(self) -> float:
        height = float(self.data.qpos[2])
        quat = self.data.qpos[3:7].copy()
        lin_vel = self.data.qvel[0:3].copy()
        ang_vel = self.data.qvel[3:6].copy()

        if self.curriculum_stage == 1:
            return R.standing_reward(height, quat, lin_vel, ang_vel, self.prev_action)

        elif self.curriculum_stage == 2:
            r = R.balance_reward(
                height, quat, lin_vel, ang_vel, self.prev_action, self.was_pushed
            )
            if self.was_pushed and height > 0.20:
                self.was_pushed = False
            return r

        elif self.curriculum_stage == 3:
            return R.stepping_reward(
                height, quat,
                self._foot_contact("left"), self._foot_contact("right"),
                self._foot_height("left"), self._foot_height("right"),
                self.gait_phase, self.prev_action, self.prev_prev_action,
            )

        else:
            return R.walking_reward(
                height, quat,
                float(self.data.qvel[0]), float(self.data.qvel[1]),
                self._foot_contact("left"), self._foot_contact("right"),
                self.target_velocity,
                self.prev_action, self.prev_prev_action,
            )

    # ----------------------------------------------------------
    # Termination
    # ----------------------------------------------------------
    def _check_termination(self):
        height = self.data.qpos[2]
        if height < C.FALL_HEIGHT:
            return True, False, "fallen"

        roll, pitch, _ = self._quat_to_rpy(self.data.qpos[3:7])
        if abs(roll) > C.MAX_TILT or abs(pitch) > C.MAX_TILT:
            return True, False, "tilted"

        if self.step_count >= self.max_episode_steps:
            return False, True, "timeout"

        return False, False, ""

    # ----------------------------------------------------------
    # Balance push
    # ----------------------------------------------------------
    def _maybe_push(self):
        if self.curriculum_stage != 2:
            return
        self.steps_since_push += 1
        if self.steps_since_push > np.random.randint(C.PUSH_INTERVAL_MIN, C.PUSH_INTERVAL_MAX):
            force = np.random.uniform(-C.PUSH_FORCE_RANGE, C.PUSH_FORCE_RANGE, size=3)
            force[2] = 0.0
            self.data.xfrc_applied[self.base_id, :3] = force
            self.was_pushed = True
            self.push_count += 1
            self.steps_since_push = 0
        else:
            self.data.xfrc_applied[self.base_id, :] = 0.0

    # ----------------------------------------------------------
    # Core Gym API
    # ----------------------------------------------------------
    def step(self, action: np.ndarray):
        self.prev_prev_action = self.prev_action.copy()
        self.prev_action = action.copy()

        self._apply_action(action)
        self._maybe_push()

        for _ in range(C.PHYSICS_STEPS_PER_CTRL):
            mujoco.mj_step(self.model, self.data)

        self.gait_phase += 2.0 * np.pi * self.dt / C.GAIT_PERIOD

        reward = self._compute_reward()
        self.episode_reward += reward
        self.step_count += 1

        terminated, truncated, reason = self._check_termination()

        info = {
            "step": self.step_count,
            "height": float(self.data.qpos[2]),
            "velocity": float(self.data.qvel[0]),
            "termination_reason": reason,
            "curriculum_stage": self.curriculum_stage,
            "episode_reward": self.episode_reward,
        }
        if self.curriculum_stage == 2:
            info["push_count"] = self.push_count
            info["survived_pushes"] = self.push_count if not terminated else max(0, self.push_count - 1)
        if self.curriculum_stage >= 3:
            info["left_contact"] = self._foot_contact("left")
            info["right_contact"] = self._foot_contact("right")

        return self._get_obs(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options:
            if "curriculum_stage" in options:
                self.curriculum_stage = options["curriculum_stage"]
            if "target_velocity" in options:
                self.target_velocity = options["target_velocity"]

        mujoco.mj_resetData(self.model, self.data)

        # Set initial pose
        self.data.qpos[2] = C.TARGET_HEIGHT
        self.data.qpos[3:7] = [1, 0, 0, 0]
        for addr, angle in C.INITIAL_JOINT_POSE.items():
            self.data.qpos[addr] = angle

        if self.randomize:
            self.data.qpos[2] += self.np_random.uniform(-0.02, 0.02)
            for addr in C.JOINT_QPOS_ADDRS:
                self.data.qpos[addr] += self.np_random.uniform(-0.05, 0.05)

        mujoco.mj_forward(self.model, self.data)

        # Reset state
        self.step_count = 0
        self.prev_action[:] = 0.0
        self.prev_prev_action[:] = 0.0
        self.gait_phase = 0.0
        self.episode_reward = 0.0
        self.push_count = 0
        self.steps_since_push = 0
        self.was_pushed = False

        self._apply_action(np.zeros(C.ACT_DIM))

        return self._get_obs(), {"curriculum_stage": self.curriculum_stage}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.cam.distance = 1.5
                self.viewer.cam.elevation = -20
                self.viewer.cam.azimuth = 135
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def set_curriculum_stage(self, stage: int):
        self.curriculum_stage = stage

    def set_target_velocity(self, velocity: float):
        self.target_velocity = velocity
