#!/usr/bin/env python3
"""
All configuration, constants, and hyperparameters for GO-BDX RL walking.
"""

import numpy as np


# ============================================================
# Robot Physical Constants (measured from simulation)
# ============================================================
TARGET_HEIGHT = 0.26          # Target CoM height when standing (m)
FALL_HEIGHT = 0.12            # Below this = fallen (m)
MAX_TILT = 0.78               # ~45 deg max roll/pitch before termination (rad)

# ============================================================
# Actuator Mapping
# ============================================================
# MuJoCo XML has 15 actuators (0-14):
#   0: left_hip_roll,   1: left_hip_pitch,  2: left_hip_yaw,
#   3: left_shin,       4: left_foot,
#   5: neck,  6: head_pitch,  7: head_yaw,  8: left_antenna,  9: right_antenna,
#  10: right_hip_roll, 11: right_hip_pitch, 12: right_hip_yaw,
#  13: right_shin,     14: right_foot
#
# RL controls only the 10 leg actuators:
LEG_ACTUATOR_IDS = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]

# Action scaling per joint (hip roll limited to prevent sideways fall)
ACTION_SCALES = np.array([
    0.3,   # left_hip_roll   - LIMITED
    0.8,   # left_hip_pitch  - main walking joint
    0.5,   # left_hip_yaw
    1.0,   # left_shin (knee)
    0.5,   # left_foot (ankle)
    0.3,   # right_hip_roll  - LIMITED
    0.8,   # right_hip_pitch - main walking joint
    0.5,   # right_hip_yaw
    1.0,   # right_shin (knee)
    0.5,   # right_foot (ankle)
], dtype=np.float32)

# qpos addresses for leg joints (free joint uses 0-6, then joints sequential)
JOINT_QPOS_ADDRS = [7, 8, 9, 10, 11, 17, 18, 19, 20, 21]

# qvel addresses for leg joints (free joint uses 0-5, then joints sequential)
JOINT_QVEL_ADDRS = [6, 7, 8, 9, 10, 16, 17, 18, 19, 20]

# Initial standing pose (slightly bent knees)
INITIAL_JOINT_POSE = {
    7:  0.0,    # left_hip_roll
    8:  0.2,    # left_hip_pitch
    9:  0.0,    # left_hip_yaw
    10: -0.4,   # left_shin
    11: 0.2,    # left_foot
    17: 0.0,    # right_hip_roll
    18: 0.2,    # right_hip_pitch
    19: 0.0,    # right_hip_yaw
    20: -0.4,   # right_shin
    21: 0.2,    # right_foot
}

# Body names in MuJoCo XML
BODY_BASE = "floating_base"
BODY_LEFT_FOOT = "left_foot_link"
BODY_RIGHT_FOOT = "right_foot_link"

# ============================================================
# Environment Settings
# ============================================================
CONTROL_FREQ = 50             # Hz (agent decision rate)
PHYSICS_STEPS_PER_CTRL = 10   # physics at 500 Hz, control at 50 Hz
MAX_EPISODE_STEPS = 250       # 250 steps = 5 seconds at 50 Hz
FOOT_CONTACT_THRESHOLD = 0.03 # foot height below this = contact (m)
GAIT_PERIOD = 1.0             # seconds per full gait cycle

# ============================================================
# Observation Space Layout (43 dims)
# ============================================================
#  0     : body height                    (1)
#  1-4   : body orientation quaternion    (4)
#  5-7   : body linear velocity           (3)
#  8-10  : body angular velocity          (3)
# 11-20  : leg joint positions            (10)
# 21-30  : leg joint velocities           (10)
# 31-40  : previous action               (10)
# 41     : target velocity                (1)
# 42-43  : gait phase sin/cos            (2)
OBS_DIM = 43
ACT_DIM = 10

# ============================================================
# Curriculum
# ============================================================
CURRICULUM_STAGES = {
    1: "standing",
    2: "balance",
    3: "stepping",
    4: "walking",
}
SUCCESS_THRESHOLD = 0.80      # 80% success rate to advance
SUCCESS_HISTORY_LEN = 100     # evaluate over last 100 episodes

# Balance stage push config
PUSH_INTERVAL_MIN = 50        # steps
PUSH_INTERVAL_MAX = 100
PUSH_FORCE_RANGE = 30.0       # Newtons, applied as [-30, 30]

# Walking velocity targets (progressive)
WALKING_VELOCITIES = [0.1, 0.2, 0.3, 0.4, 0.5]

# ============================================================
# PPO Hyperparameters
# ============================================================
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.005,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {
        "net_arch": {
            "pi": [256, 256, 128],
            "vf": [256, 256, 128],
        },
        "activation_fn": "ELU",
    },
}

# ============================================================
# Training Settings
# ============================================================
NUM_ENVS = 8                  # parallel environments
TOTAL_TIMESTEPS = 5_000_000
CHECKPOINT_INTERVAL = 100_000 # save every N steps
LOG_INTERVAL = 10             # log every N PPO updates
PLOT_INTERVAL = 5000          # update plots every N steps
