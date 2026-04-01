# GO-BDX RL Walking Pipeline — Self-Contained Implementation

## Overview

This directory contains a complete, self-contained reinforcement learning pipeline
to train the GO-BDX bipedal robot to walk using PPO with curriculum learning.

**No files outside this directory are imported or modified.**

---

## File Structure

```
rl_forwalking/
├── __init__.py          # Package marker
├── PLAN.md              # This document
├── config.py            # All constants, hyperparameters, actuator mappings
├── env.py               # Gymnasium environment (GoBdxWalkingEnv)
├── rewards.py           # Reward functions for all 4 stages
├── curriculum.py        # Curriculum scheduler (auto stage advancement)
├── train.py             # Main PPO training script (SB3)
├── test_policy.py       # Evaluate a trained policy
└── logger.py            # Matplotlib-based training logger
```

---

## Curriculum Stages

| Stage | Name     | Goal                        | Success Condition              | Advance At |
|-------|----------|-----------------------------|--------------------------------|------------|
| 1     | Standing | Stay upright 5 seconds      | Episode timeout (no fall)      | 80/100     |
| 2     | Balance  | Recover from random pushes  | Timeout + survived 3+ pushes  | 80/100     |
| 3     | Stepping | Alternate foot lifts        | Timeout (maintained pattern)   | 80/100     |
| 4     | Walking  | Forward velocity tracking   | Timeout + vel error < 0.15 m/s | Final      |

Stage 4 has internal velocity progression: 0.1 → 0.2 → 0.3 → 0.4 → 0.5 m/s.

---

## Reward Function Design

### Design Principles

1. **Gaussian shaping** for continuous targets:
   `r = exp(-k * error^2)` — smooth gradient everywhere, peaks at target, never negative.

2. **Binary contact matching** for gait timing:
   Compare actual foot contacts vs. desired pattern from gait clock.

3. **Penalties are subtractive and small:**
   Action magnitude, action jerk, lateral drift. Prevent degenerate solutions.

4. **Weights sum to ~1.0** (before penalties) so total reward is interpretable.

### Stage 1: Standing

```
reward = 0.4 * height_reward + 0.4 * upright_reward + 0.2 * stillness_reward - action_penalty

height_reward   = exp(-50 * (h - 0.26)^2)        # Gaussian at target height
upright_reward  = exp(-5 * (roll^2 + pitch^2))    # Penalize tilt
stillness_reward = exp(-2 * (|v_lin| + |v_ang|))  # Penalize movement
action_penalty  = 0.01 * sum(action^2)            # Energy efficiency
```

**Intuition:** The robot gets maximum reward (~1.0/step) by standing perfectly still
at 0.26m height with no tilt. Any deviation reduces reward smoothly — no cliffs.

### Stage 2: Balance

```
reward = standing_reward + recovery_bonus

recovery_bonus = 0.3 if (was_pushed AND height > 0.20) else 0.0
```

**Intuition:** Same as standing but with a bonus for surviving pushes. Random forces
(up to 30N) are applied every 50-100 steps. The robot learns that disturbances are
normal and it should actively correct rather than lock joints rigidly.

### Stage 3: Stepping

```
reward = 0.25 * height_r + 0.25 * upright_r + 0.30 * contact_r + 0.20 * clearance_r - jerk_penalty

contact_reward:
  phase [0, pi):   left should be UP, right should be DOWN
  phase [pi, 2pi): left should be DOWN, right should be UP
  r = 0.5 * (left_correct) + 0.5 * (right_correct)

clearance_reward = clip(swing_foot_height / 0.05, 0, 1)   # Want 5cm lift
jerk_penalty     = 0.01 * sum((action - prev_action)^2)
```

**Intuition:** The gait clock (sin/cos in observation) tells the policy which phase
of the gait cycle it's in. The reward matches actual foot contacts against the desired
pattern. Clearance reward ensures the swing foot actually lifts (not just losing contact
by sliding). Jerk penalty prevents leg vibration.

### Stage 4: Walking

```
reward = 0.35 * vel_r + 0.20 * height_r + 0.15 * upright_r + 0.15 * contact_r
         - energy_pen - smooth_pen - lateral_pen

vel_reward     = exp(-4 * |v_actual - v_target|)    # Track target speed
contact_reward = 1.0 if any foot down, +0.5 bonus for alternating
energy_penalty = 0.001 * sum(action^2)
smooth_penalty = 0.005 * sum((action - prev_action)^2)
lateral_penalty = 0.1 * |v_lateral|                  # Walk straight
```

**Intuition:** Velocity tracking is now the primary objective (highest weight 0.35).
The robot must walk forward at a specific speed while staying upright and using proper
foot placement. Three penalty terms prevent degenerate solutions:
- Energy penalty: don't waste torque
- Smoothness penalty: don't vibrate
- Lateral penalty: don't crab-walk sideways

### Why These Specific Coefficients?

| Coefficient | Value | Effect |
|-------------|-------|--------|
| Height k=50 | `exp(-50*e^2)` | 2cm error → reward drops to 0.82, 6cm → 0.16 |
| Upright k=5 | `exp(-5*e^2)` | 11° tilt → 0.82, 29° → 0.29 |
| Velocity k=4 | `exp(-4*e)` | 0.05 m/s error → 0.82, 0.2 → 0.45 |
| Stillness k=2 | `exp(-2*e)` | Gentle, doesn't over-penalize small motions |

---

## Observation Space (43 dims)

| Index   | Component              | Dims | Notes                          |
|---------|------------------------|------|--------------------------------|
| 0       | Body height            | 1    | CoM z-position                 |
| 1-4     | Body orientation       | 4    | Quaternion (w,x,y,z)           |
| 5-7     | Linear velocity        | 3    | Body frame                     |
| 8-10    | Angular velocity       | 3    | Body frame                     |
| 11-20   | Joint positions        | 10   | 10 leg joints                  |
| 21-30   | Joint velocities       | 10   | 10 leg joints                  |
| 31-40   | Previous action        | 10   | For smoothness                 |
| 41      | Target velocity        | 1    | Commanded forward speed        |
| 42-43   | Gait phase (sin, cos)  | 2    | Cyclic gait clock              |

## Action Space (10 dims)

| Index | Joint            | Scale | Role             |
|-------|------------------|-------|------------------|
| 0     | left_hip_roll    | 0.3   | Side balance     |
| 1     | left_hip_pitch   | 0.8   | Main walking     |
| 2     | left_hip_yaw     | 0.5   | Leg rotation     |
| 3     | left_shin        | 1.0   | Knee             |
| 4     | left_foot        | 0.5   | Ankle            |
| 5     | right_hip_roll   | 0.3   | Side balance     |
| 6     | right_hip_pitch  | 0.8   | Main walking     |
| 7     | right_hip_yaw    | 0.5   | Leg rotation     |
| 8     | right_shin       | 1.0   | Knee             |
| 9     | right_foot       | 0.5   | Ankle            |

Actions are in [-1, 1], multiplied by scale before applying to MuJoCo actuators.
Hip roll is limited (scale=0.3) to prevent sideways collapse.

---

## PPO Hyperparameters

| Parameter     | Value   | Rationale                                 |
|---------------|---------|-------------------------------------------|
| learning_rate | 3e-4    | Standard for locomotion                   |
| n_steps       | 2048    | Enough trajectory for GAE estimation      |
| batch_size    | 64      | Standard minibatch size                   |
| n_epochs      | 10      | Multiple passes over collected data       |
| gamma         | 0.99    | Long horizon (walking is long-term)       |
| gae_lambda    | 0.95    | Bias-variance tradeoff for advantage      |
| clip_range    | 0.2     | Prevent large policy updates              |
| ent_coef      | 0.005   | Small exploration bonus                   |
| Network       | 256-256-128 | Separate actor/critic, ELU activation |

---

## Usage

```bash
# Train from scratch
python -m rl_forwalking.train --model_path go_bdx.xml

# Resume from stage 2
python -m rl_forwalking.train --model_path go_bdx.xml \
    --checkpoint rl_forwalking_output/checkpoints/curriculum/standing_complete.zip \
    --stage 2

# Test a trained policy
python -m rl_forwalking.test_policy --model_path go_bdx.xml \
    --checkpoint rl_forwalking_output/checkpoints/best/best_model.zip

# Use fewer envs for low VRAM
python -m rl_forwalking.train --model_path go_bdx.xml --num_envs 4 --cpu
```

---

## Expected Timeline (RTX 2050, 8 envs)

| Stage      | Steps    | Time       |
|------------|----------|------------|
| Standing   | 300-500k | 20-30 min  |
| Balance    | 300-500k | 20-30 min  |
| Stepping   | 500k-1M  | 40-60 min  |
| Walking    | 2-5M     | 3-6 hours  |
| **Total**  | **3-7M** | **4-8 hrs**|

---

## Output Structure

```
rl_forwalking_output/
├── checkpoints/
│   ├── periodic/              # Every 100k steps
│   │   ├── step_100000.zip
│   │   └── ...
│   ├── curriculum/            # Stage completions
│   │   ├── standing_complete.zip
│   │   ├── balance_complete.zip
│   │   ├── stepping_complete.zip
│   │   └── walking_complete.zip
│   ├── best/                  # Best episode reward
│   │   └── best_model.zip
│   └── final_model.zip        # End of training
└── logs/
    ├── training_log.json       # All metrics
    ├── training_curves.png     # Reward/success/stage plots
    └── curriculum_state.json   # Curriculum progress
```
