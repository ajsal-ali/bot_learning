#!/usr/bin/env python3
"""
Reward functions for GO-BDX RL walking training.

Each stage has a dedicated reward function that receives the environment
and returns a scalar reward. Rewards are designed so that:
  - All components are in [0, 1] range (Gaussian or clipped)
  - Penalties are small and bounded
  - Weights sum to ~1.0 before penalties
  - Higher stages build on skills from lower stages

Reward Design Philosophy:
=========================
1. GAUSSIAN SHAPING for continuous targets (height, velocity):
      r = exp(-k * error^2)
   Gives smooth gradient everywhere, peaks at target, never negative.
   k controls sensitivity: larger k = sharper peak = tighter tolerance.

2. CONTACT MATCHING for gait timing:
   Compare actual foot contacts against desired pattern from a gait clock.
   Binary per-foot, averaged. Teaches alternating stance/swing.

3. PENALTY TERMS are subtractive and small:
   - Action magnitude (energy efficiency)
   - Action jerk (smoothness)
   - Lateral drift (straight walking)
   These prevent degenerate solutions (e.g. vibrating in place).

4. CURRICULUM PROGRESSION:
   Stage 1 (Standing):  height + upright + stillness
   Stage 2 (Balance):   standing reward + recovery bonus after pushes
   Stage 3 (Stepping):  height + upright + contact pattern + foot clearance
   Stage 4 (Walking):   velocity tracking + height + upright + contacts - efficiency

   Each stage inherits stability skills from prior stages via shared
   reward components (height_reward, upright_reward).
"""

import numpy as np
from . import config as C


def quat_to_rpy(quat: np.ndarray):
    """Convert quaternion (w,x,y,z) to (roll, pitch, yaw) in radians."""
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


# ==============================================================
# Shared reward components
# ==============================================================

def height_reward(height: float, target: float = C.TARGET_HEIGHT) -> float:
    """
    Gaussian reward for maintaining target CoM height.

    r = exp(-50 * (h - h_target)^2)

    Examples:
        h = 0.26 (target) -> 1.0
        h = 0.24          -> 0.82
        h = 0.20          -> 0.08
        h = 0.12 (fallen) -> ~0.0
    """
    return float(np.exp(-50.0 * (height - target) ** 2))


def upright_reward(quat: np.ndarray) -> float:
    """
    Gaussian reward for staying upright (small roll and pitch).

    r = exp(-5 * (roll^2 + pitch^2))

    Examples:
        tilt = 0 rad    -> 1.0
        tilt = 0.2 rad  -> 0.82  (11 degrees)
        tilt = 0.5 rad  -> 0.29  (29 degrees)
        tilt = 0.78 rad -> 0.05  (45 degrees, near termination)
    """
    roll, pitch, _ = quat_to_rpy(quat)
    return float(np.exp(-5.0 * (roll ** 2 + pitch ** 2)))


def stillness_reward(lin_vel: np.ndarray, ang_vel: np.ndarray) -> float:
    """
    Exponential penalty for body movement. Used in standing/balance.

    r = exp(-2 * (|lin_vel| + |ang_vel|))

    Encourages the robot to hold still rather than oscillate.
    """
    return float(np.exp(-2.0 * (np.linalg.norm(lin_vel) + np.linalg.norm(ang_vel))))


def action_magnitude_penalty(action: np.ndarray, scale: float = 0.01) -> float:
    """
    L2 penalty on action magnitude. Encourages energy efficiency.

    penalty = scale * sum(action^2)

    With 10 actions in [-1,1], max penalty = scale * 10 = 0.1
    """
    return scale * float(np.sum(action ** 2))


def action_smoothness_penalty(
    action: np.ndarray, prev_action: np.ndarray, scale: float = 0.01
) -> float:
    """
    L2 penalty on action change (jerk). Encourages smooth movements.

    penalty = scale * sum((action - prev_action)^2)

    Prevents high-frequency oscillation that MuJoCo can exploit.
    """
    return scale * float(np.sum((action - prev_action) ** 2))


def lateral_velocity_penalty(lateral_vel: float, scale: float = 0.1) -> float:
    """
    Penalize sideways drift during walking.

    penalty = scale * |v_y|
    """
    return scale * abs(lateral_vel)


# ==============================================================
# Contact pattern reward
# ==============================================================

def gait_contact_reward(
    left_contact: bool,
    right_contact: bool,
    gait_phase: float,
) -> float:
    """
    Reward for matching desired foot contact pattern.

    Gait clock divides cycle into two halves:
        phase [0, pi):   left SWING (up), right STANCE (down)
        phase [pi, 2pi): left STANCE (down), right SWING (up)

    Returns 0.0 to 1.0 (average of left + right correctness).
    """
    phase = gait_phase % (2.0 * np.pi)
    if phase < np.pi:
        want_left, want_right = False, True
    else:
        want_left, want_right = True, False

    return 0.5 * float(left_contact == want_left) + 0.5 * float(right_contact == want_right)


def foot_clearance_reward(
    left_contact_desired: bool,
    left_foot_z: float,
    right_foot_z: float,
    gait_phase: float,
    target_clearance: float = 0.05,
) -> float:
    """
    Reward for lifting the swing foot to target clearance height.

    Only measures the foot that should be swinging.
    Clipped to [0, 1]: reaches 1.0 when swing foot >= target_clearance.
    """
    phase = gait_phase % (2.0 * np.pi)
    if phase < np.pi:
        # Left is swinging
        return float(np.clip(left_foot_z / target_clearance, 0.0, 1.0))
    else:
        # Right is swinging
        return float(np.clip(right_foot_z / target_clearance, 0.0, 1.0))


def velocity_tracking_reward(actual_vel: float, target_vel: float) -> float:
    """
    Gaussian reward for tracking a target forward velocity.

    r = exp(-4 * |v - v_target|)

    Examples (target=0.3):
        v = 0.30 -> 1.0
        v = 0.25 -> 0.82
        v = 0.10 -> 0.45
        v = 0.00 -> 0.30
        v = -0.1 -> 0.20
    """
    return float(np.exp(-4.0 * abs(actual_vel - target_vel)))


def walking_contact_reward(left_contact: bool, right_contact: bool) -> float:
    """
    Contact reward for walking: at least one foot on ground,
    bonus for alternating (one up, one down).

    Returns:
        0.0: both feet in air (bad - jumping/falling)
        1.0: both feet on ground (acceptable - double support)
        1.5: one foot on ground, one in air (ideal - single support)
    """
    base = 1.0 if (left_contact or right_contact) else 0.0
    if left_contact != right_contact:
        base += 0.5
    return base


# ==============================================================
# Stage reward functions
# ==============================================================

def standing_reward(
    height: float,
    quat: np.ndarray,
    lin_vel: np.ndarray,
    ang_vel: np.ndarray,
    action: np.ndarray,
) -> float:
    """
    Stage 1: Standing Reward

    Goal: Stay upright at target height, minimize movement.

    Components (weights sum to 1.0):
        0.4 * height_reward     - maintain ~0.26m CoM height
        0.4 * upright_reward    - keep body level (no roll/pitch)
        0.2 * stillness_reward  - don't wobble or drift
        - action_penalty        - minimize energy use

    Typical range: [-0.1, 1.0]
    A perfectly standing robot scores ~1.0 per step.
    """
    h_r = height_reward(height)
    u_r = upright_reward(quat)
    s_r = stillness_reward(lin_vel, ang_vel)
    a_pen = action_magnitude_penalty(action)

    return 0.4 * h_r + 0.4 * u_r + 0.2 * s_r - a_pen


def balance_reward(
    height: float,
    quat: np.ndarray,
    lin_vel: np.ndarray,
    ang_vel: np.ndarray,
    action: np.ndarray,
    was_pushed: bool,
) -> float:
    """
    Stage 2: Balance Reward

    Goal: Same as standing + bonus for recovering after pushes.

    Components:
        standing_reward (all of stage 1)
        + 0.3 recovery bonus if pushed but still above 0.20m

    The recovery bonus teaches the robot that being disturbed is
    normal and it should actively correct rather than just be stiff.
    """
    base = standing_reward(height, quat, lin_vel, ang_vel, action)
    recovery = 0.3 if (was_pushed and height > 0.20) else 0.0
    return base + recovery


def stepping_reward(
    height: float,
    quat: np.ndarray,
    left_contact: bool,
    right_contact: bool,
    left_foot_z: float,
    right_foot_z: float,
    gait_phase: float,
    action: np.ndarray,
    prev_action: np.ndarray,
) -> float:
    """
    Stage 3: Stepping in Place Reward

    Goal: Alternate foot contacts in rhythm with gait clock.

    Components (weights sum to 1.0):
        0.25 * height_reward     - don't sink or fall
        0.25 * upright_reward    - stay level while stepping
        0.30 * contact_reward    - match desired contact pattern
        0.20 * clearance_reward  - lift swing foot off ground
        - smoothness_penalty     - no jerky leg movements

    The gait clock (sin/cos in observation) tells the policy
    which foot should be up. This is the key stepping signal.
    """
    h_r = height_reward(height)
    u_r = upright_reward(quat)
    c_r = gait_contact_reward(left_contact, right_contact, gait_phase)
    cl_r = foot_clearance_reward(True, left_foot_z, right_foot_z, gait_phase)
    s_pen = action_smoothness_penalty(action, prev_action)

    return 0.25 * h_r + 0.25 * u_r + 0.30 * c_r + 0.20 * cl_r - s_pen


def walking_reward(
    height: float,
    quat: np.ndarray,
    forward_vel: float,
    lateral_vel: float,
    left_contact: bool,
    right_contact: bool,
    target_velocity: float,
    action: np.ndarray,
    prev_action: np.ndarray,
) -> float:
    """
    Stage 4: Walking Reward

    Goal: Move forward at target velocity with stable gait.

    Components (weights sum to ~0.85, rest is penalties):
        0.35 * velocity_reward   - track target forward speed
        0.20 * height_reward     - don't fall
        0.15 * upright_reward    - stay level
        0.15 * contact_reward    - proper foot placement
        - energy_penalty         - efficient actuation
        - smoothness_penalty     - no jerky movements
        - lateral_penalty        - walk straight, don't crab-walk

    The velocity reward is the primary objective. Other terms
    prevent degenerate solutions like falling forward fast or
    hopping instead of walking.
    """
    v_r = velocity_tracking_reward(forward_vel, target_velocity)
    h_r = height_reward(height)
    u_r = upright_reward(quat)
    c_r = walking_contact_reward(left_contact, right_contact)

    e_pen = action_magnitude_penalty(action, scale=0.001)
    s_pen = action_smoothness_penalty(action, prev_action, scale=0.005)
    l_pen = lateral_velocity_penalty(lateral_vel)

    return 0.35 * v_r + 0.20 * h_r + 0.15 * u_r + 0.15 * c_r - e_pen - s_pen - l_pen
