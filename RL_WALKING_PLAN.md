# GO-BDX Reinforcement Learning Walking Plan

## Current Status

### ✅ COMPLETED
- MuJoCo simulation working (`go_bdx.xml`)
- URDF to MuJoCo conversion pipeline (`convert_urdf.py`)
- Keyboard control for manual testing (`keyboard_control.py`)
- All 15 joints actuated and controllable
- Hip roll drift issue fixed (kp=2000)
- Git repository initialized
- **Gymnasium RL environment** (`rl/go_bdx_env.py`)
- **Curriculum scheduler** (`rl/curriculum.py`)
- **PPO training script** (`rl/train.py`)
- **Policy evaluation script** (`rl/test_policy.py`)

### 🔄 IN PROGRESS
- Training the model (run `python rl/train.py`)

### ❌ NOT YET DONE
- ROS2 integration
- Trained walking policy (requires training time)
- Policy export for deployment

---

## 1. Robot Specifications (Measured from Simulation)

### 1.1 Physical Properties

| Property | Value | Notes |
|----------|-------|-------|
| Total Mass | **43.47 kg** | Heavy robot |
| Body Mass | 28.57 kg | Main body |
| Standing Height (CoM) | **0.25-0.30 m** | Body center above ground |
| Fallen Height (CoM) | **~0.03 m** | Lying on ground |
| Fall Detection Threshold | **< 0.12 m** | Use this in termination |
| Time to Fall (no control) | **~1.6 seconds** | How fast it collapses |

### 1.2 Actuators for Walking (10 of 15)

**Leg Actuators (RL Action Space):**
| Action Index | Actuator Index | Name | kp | Role |
|--------------|----------------|------|-----|------|
| 0 | 0 | left_hip_roll | 2000 | Side balance (LIMIT to ±0.3) |
| 1 | 1 | left_hip_pitch | 1000 | **Main walking** |
| 2 | 2 | left_hip_yaw | 500 | Leg rotation |
| 3 | 3 | left_shin | 300 | Knee bend |
| 4 | 4 | left_foot | 300 | Ankle |
| 5 | 10 | right_hip_roll | 2000 | Side balance (LIMIT to ±0.3) |
| 6 | 11 | right_hip_pitch | 1000 | **Main walking** |
| 7 | 12 | right_hip_yaw | 500 | Leg rotation |
| 8 | 13 | right_shin | 300 | Knee bend |
| 9 | 14 | right_foot | 300 | Ankle |

---

## 2. Algorithm Deep Dive: Why PPO with Neural Networks?

### 2.1 RL Algorithm Comparison

| Algorithm | Type | Policy | Best For | Why NOT for GO-BDX |
|-----------|------|--------|----------|-------------------|
| **Q-Learning** | Value-based | Discrete | Simple games | ❌ Continuous action space (joint angles) |
| **DQN** | Value-based | Discrete | Atari games | ❌ Cannot handle continuous control |
| **DDPG** | Actor-Critic | Continuous | Robotics | ❌ Brittle, hard to tune |
| **TD3** | Actor-Critic | Continuous | Robotics | ⚠️ OK but less stable than PPO |
| **SAC** | Actor-Critic | Continuous | Exploration | ⚠️ Harder to tune, more hyperparameters |
| **PPO** | Actor-Critic | Continuous | Locomotion | ✅ **BEST CHOICE** |

### 2.2 Why PPO is the Best Choice

**1. Continuous Action Space:**
- GO-BDX needs continuous joint angles (not discrete)
- PPO naturally handles continuous actions
- Q-learning/DQN cannot work here (they need discrete actions)

**2. Stability:**
- PPO uses "clipped" objective to prevent large policy updates
- Walking is sensitive - large updates cause catastrophic forgetting
- Robot suddenly forgets how to stand after learning to step

**3. Sample Efficiency:**
- PPO is on-policy but very sample-efficient
- Bipedal walking research (DeepMind, OpenAI) all use PPO variants
- Faster convergence than SAC/TD3 for locomotion tasks

**4. Proven Track Record:**
- OpenAI's Humanoid: PPO
- DeepMind's walking robots: PPO
- Boston Dynamics sim-to-real: PPO variants
- 90%+ of bipedal RL papers use PPO

### 2.3 Why Neural Network (Not Linear/Tabular)?

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| **Tabular Q-Learning** | Simple | Cannot handle continuous state/action | ❌ Impossible |
| **Linear Function Approx** | Fast | Cannot learn complex walking patterns | ❌ Too simple |
| **Neural Network** | Learns complex patterns | Needs more data | ✅ **Required** |

**Walking requires a neural network because:**
1. **43-dim observation space** - too large for tabular
2. **10-dim action space** - continuous, needs function approximation
3. **Complex dynamics** - nonlinear relationship between actions and balance
4. **Temporal patterns** - need to learn gait timing

### 2.4 Policy Network Architecture (Detailed)

```
┌─────────────────────────────────────────────────────────────────┐
│                    PPO ACTOR-CRITIC NETWORK                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  OBSERVATION (43 dims)                                          │
│  ├── Body height (1)                                            │
│  ├── Body orientation quaternion (4)                            │
│  ├── Body linear velocity (3)                                   │
│  ├── Body angular velocity (3)                                  │
│  ├── Joint positions (10)                                       │
│  ├── Joint velocities (10)                                      │
│  ├── Previous action (10)                                       │
│  ├── Target velocity (1)                                        │
│  └── Gait phase sin/cos (2)                                     │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────┐                                            │
│  │ SHARED FEATURES │ (Optional - can be separate)               │
│  └────────┬────────┘                                            │
│           │                                                     │
│     ┌─────┴─────┐                                               │
│     ▼           ▼                                               │
│  ACTOR         CRITIC                                           │
│  (Policy π)    (Value V)                                        │
│     │             │                                             │
│     ▼             ▼                                             │
│  ┌──────┐     ┌──────┐                                          │
│  │FC 256│     │FC 256│  Fully Connected + ELU activation        │
│  │ ELU  │     │ ELU  │                                          │
│  └──┬───┘     └──┬───┘                                          │
│     ▼            ▼                                              │
│  ┌──────┐     ┌──────┐                                          │
│  │FC 256│     │FC 256│                                          │
│  │ ELU  │     │ ELU  │                                          │
│  └──┬───┘     └──┬───┘                                          │
│     ▼            ▼                                              │
│  ┌──────┐     ┌──────┐                                          │
│  │FC 128│     │FC 128│                                          │
│  │ ELU  │     │ ELU  │                                          │
│  └──┬───┘     └──┬───┘                                          │
│     ▼            ▼                                              │
│  ┌──────┐     ┌──────┐                                          │
│  │FC 10 │     │FC 1  │                                          │
│  │ Tanh │     │Linear│                                          │
│  └──┬───┘     └──┬───┘                                          │
│     │            │                                              │
│     ▼            ▼                                              │
│  ACTION       STATE VALUE                                       │
│  (10 dims)    (scalar)                                          │
│  [-1, +1]     V(s)                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Network Parameters:
  - Actor: 43 → 256 → 256 → 128 → 10 = ~140,000 parameters
  - Critic: 43 → 256 → 256 → 128 → 1 = ~133,000 parameters
  - Total: ~273,000 parameters (~1.1 MB)
```

**Why ELU Activation (not ReLU)?**
- ELU has smooth gradients for negative inputs
- Better for continuous control (smoother actions)
- Used in most locomotion papers

**Why Tanh Output (not Linear)?**
- Actions bounded to [-1, +1]
- Prevents exploding actions
- Scaled to actual joint limits later

---

## 3. Curriculum Learning (Success-Based Transitions)

### 3.1 Curriculum Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CURRICULUM STAGES                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STAGE 1: STANDING                                              │
│  ├── Goal: Stay upright for 250 steps (5 sec)                   │
│  ├── Success: Episode reaches timeout without falling           │
│  ├── Transition: 80/100 recent episodes successful              │
│  └── Checkpoint: standing_complete.zip                          │
│           │                                                     │
│           ▼ (80% success rate)                                  │
│                                                                 │
│  STAGE 2: BALANCE                                               │
│  ├── Goal: Recover from random pushes                           │
│  ├── Success: Survive push without falling                      │
│  ├── Transition: 80/100 recent episodes successful              │
│  └── Checkpoint: balance_complete.zip                           │
│           │                                                     │
│           ▼ (80% success rate)                                  │
│                                                                 │
│  STAGE 3: STEPPING                                              │
│  ├── Goal: Alternate foot lifts in place                        │
│  ├── Success: Correct foot contacts + no fall                   │
│  ├── Transition: 80/100 recent episodes successful              │
│  └── Checkpoint: stepping_complete.zip                          │
│           │                                                     │
│           ▼ (80% success rate)                                  │
│                                                                 │
│  STAGE 4: WALKING                                               │
│  ├── Goal: Forward velocity tracking                            │
│  ├── Success: Maintain target velocity ±0.1 m/s                 │
│  └── Final Checkpoint: walking_complete.zip                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Success Criteria Per Stage

| Stage | Success Condition | Failure Condition | Transition Trigger |
|-------|-------------------|-------------------|-------------------|
| 1. Standing | Episode timeout (250 steps) | Fall (height < 0.12) | **80/100 success** |
| 2. Balance | Survive 3+ pushes | Fall after push | **80/100 success** |
| 3. Stepping | Correct contacts >70% | Fall or wrong contacts | **80/100 success** |
| 4. Walking | Avg velocity error <0.1 m/s | Fall | N/A (final) |

### 3.3 Curriculum Scheduler Implementation

```python
class CurriculumScheduler:
    def __init__(self):
        self.stage = 1  # Start at standing
        self.stage_names = ["standing", "balance", "stepping", "walking"]
        self.success_history = deque(maxlen=100)  # Last 100 episodes
        self.success_threshold = 0.80  # 80% to advance
        
    def record_episode(self, success: bool):
        """Record episode outcome"""
        self.success_history.append(1 if success else 0)
        
        # Check for stage transition
        if len(self.success_history) >= 100:
            success_rate = sum(self.success_history) / len(self.success_history)
            
            if success_rate >= self.success_threshold and self.stage < 4:
                self.advance_stage()
    
    def advance_stage(self):
        """Move to next curriculum stage"""
        # Save checkpoint for completed stage
        self.save_checkpoint(f"{self.stage_names[self.stage-1]}_complete.zip")
        
        # Advance
        self.stage += 1
        self.success_history.clear()  # Reset history for new stage
        print(f"🎉 Advanced to Stage {self.stage}: {self.stage_names[self.stage-1]}")
    
    def get_reward_function(self):
        """Return appropriate reward for current stage"""
        return [standing_reward, balance_reward, 
                stepping_reward, walking_reward][self.stage - 1]
    
    def is_success(self, info: dict) -> bool:
        """Determine if episode was successful for current stage"""
        if self.stage == 1:  # Standing
            return info.get("termination_reason") == "timeout"
        elif self.stage == 2:  # Balance
            return info.get("survived_pushes", 0) >= 3
        elif self.stage == 3:  # Stepping
            return info.get("contact_accuracy", 0) > 0.7
        else:  # Walking
            return info.get("avg_velocity_error", 1.0) < 0.1
```

### 3.4 Checkpointing Strategy

**Automatic Checkpoints:**
```python
checkpoint_schedule = {
    "periodic": 100_000,      # Every 100k steps
    "stage_complete": True,    # When curriculum stage completes
    "best_reward": True,       # When new best reward achieved
}
```

**Checkpoint Files:**
```
checkpoints/
├── periodic/
│   ├── step_100000.zip
│   ├── step_200000.zip
│   └── ...
├── curriculum/
│   ├── standing_complete.zip    # After Stage 1
│   ├── balance_complete.zip     # After Stage 2
│   ├── stepping_complete.zip    # After Stage 3
│   └── walking_complete.zip     # Final
└── best/
    └── best_model.zip           # Highest reward ever
```

### 3.5 Resume Training from Checkpoint

```python
def resume_training(checkpoint_path: str, start_stage: int):
    """
    Resume training from a specific checkpoint and stage.
    
    Example:
        # Resume from standing checkpoint, start balance training
        resume_training("checkpoints/curriculum/standing_complete.zip", stage=2)
    """
    # Load model
    model = PPO.load(checkpoint_path)
    
    # Load normalization stats
    vec_env = VecNormalize.load(
        checkpoint_path.replace(".zip", "_vecnorm.pkl"),
        make_env()
    )
    
    # Set curriculum stage
    curriculum = CurriculumScheduler()
    curriculum.stage = start_stage
    curriculum.success_history.clear()
    
    # Update environment for new stage
    env.set_curriculum_stage(start_stage)
    
    # Continue training
    model.set_env(vec_env)
    model.learn(total_timesteps=remaining_steps)
    
    return model

# Usage examples:
# Start fresh:
#   python train.py --stage 1

# Resume after standing:
#   python train.py --checkpoint standing_complete.zip --stage 2

# Resume after balance:
#   python train.py --checkpoint balance_complete.zip --stage 3
```

---

## 4. Reward Functions (Detailed)

### 4.1 Stage 1: Standing Reward

```python
def standing_reward(env) -> float:
    """
    Reward for standing still without falling.
    Components:
      - Height maintenance (want ~0.26m)
      - Upright orientation (no tilt)
      - Minimal movement (don't wobble)
    """
    # Height reward: Gaussian centered at target height
    height = env.data.qpos[2]
    target_height = 0.26
    height_reward = np.exp(-50 * (height - target_height)**2)
    # height=0.26 → 1.0, height=0.20 → 0.08
    
    # Upright reward: penalize roll and pitch
    quat = env.data.qpos[3:7]  # wxyz
    # Convert to roll, pitch, yaw
    roll, pitch, yaw = quat_to_euler(quat)
    upright_reward = np.exp(-5 * (roll**2 + pitch**2))
    # tilt=0 → 1.0, tilt=0.3rad → 0.64
    
    # Stillness reward: penalize body velocity
    lin_vel = np.linalg.norm(env.data.qvel[0:3])
    ang_vel = np.linalg.norm(env.data.qvel[3:6])
    stillness_reward = np.exp(-2 * (lin_vel + ang_vel))
    
    # Action penalty: minimize control effort
    action_penalty = 0.01 * np.sum(env.action**2)
    
    # Combine
    reward = (
        0.4 * height_reward +
        0.4 * upright_reward +
        0.2 * stillness_reward -
        action_penalty
    )
    return reward
```

### 4.2 Stage 2: Balance Reward

```python
def balance_reward(env) -> float:
    """
    Same as standing + bonus for recovering from pushes.
    """
    base_reward = standing_reward(env)
    
    # Recovery bonus: if we were pushed but still standing
    if env.was_pushed_recently and env.data.qpos[2] > 0.20:
        recovery_bonus = 0.5
    else:
        recovery_bonus = 0.0
    
    return base_reward + recovery_bonus
```

### 4.3 Stage 3: Stepping Reward

```python
def stepping_reward(env) -> float:
    """
    Reward for alternating foot contacts (stepping in place).
    """
    phase = env.gait_phase  # 0 to 2π, cycles every ~1 second
    
    # Get foot contact states
    left_contact = env.get_foot_contact("left")   # True/False
    right_contact = env.get_foot_contact("right")
    
    # Desired pattern:
    #   phase 0-π:   left foot UP (swing), right foot DOWN (stance)
    #   phase π-2π:  left foot DOWN (stance), right foot UP (swing)
    
    if 0 <= phase < np.pi:
        # Left should be swinging (not contact), right should be stance (contact)
        desired_left = False
        desired_right = True
    else:
        desired_left = True
        desired_right = False
    
    # Contact accuracy
    left_correct = (left_contact == desired_left)
    right_correct = (right_contact == desired_right)
    contact_reward = 0.5 * left_correct + 0.5 * right_correct
    
    # Foot clearance: swing foot should be lifted
    if not desired_left:  # Left is swinging
        left_foot_z = env.get_foot_height("left")
        clearance_reward = min(left_foot_z / 0.05, 1.0)  # Target 5cm lift
    else:
        right_foot_z = env.get_foot_height("right")
        clearance_reward = min(right_foot_z / 0.05, 1.0)
    
    # Still maintain height and upright
    height = env.data.qpos[2]
    height_reward = np.exp(-50 * (height - 0.26)**2)
    
    reward = (
        0.3 * height_reward +
        0.4 * contact_reward +
        0.3 * clearance_reward
    )
    return reward
```

### 4.4 Stage 4: Walking Reward

```python
def walking_reward(env) -> float:
    """
    Reward for forward locomotion at target velocity.
    """
    target_vel = env.target_velocity  # e.g., 0.3 m/s
    actual_vel = env.data.qvel[0]     # Forward velocity (x direction)
    
    # Velocity tracking (exponential - peaks at target)
    vel_error = abs(actual_vel - target_vel)
    vel_reward = np.exp(-4 * vel_error)
    # error=0 → 1.0, error=0.1 → 0.67, error=0.3 → 0.30
    
    # Height maintenance
    height = env.data.qpos[2]
    height_reward = np.exp(-50 * (height - 0.26)**2)
    
    # Foot contact (similar to stepping)
    contact_reward = compute_contact_reward(env)
    
    # Energy efficiency (penalize large actions)
    energy_penalty = 0.001 * np.sum(env.action**2)
    
    # Action smoothness (penalize jerky movements)
    if env.prev_action is not None:
        smooth_penalty = 0.01 * np.sum((env.action - env.prev_action)**2)
    else:
        smooth_penalty = 0
    
    reward = (
        0.40 * vel_reward +
        0.25 * height_reward +
        0.20 * contact_reward -
        energy_penalty -
        smooth_penalty
    )
    return reward
```

---


---

## 5. Training Configuration

### 5.1 Hardware Setup (RTX 2050, 4GB VRAM)

| Setting | Value | Rationale |
|---------|-------|-----------|
| Parallel Envs | **8-12** | ~300 MB for 12 envs |
| Render Env | **1** | For visualization |
| Batch Size | 64 | Standard PPO |
| Device | **cuda** | PyTorch GPU acceleration |

### 5.2 PyTorch GPU Setup

```python
import torch

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# In PPO setup
model = PPO(
    "MlpPolicy",
    vec_env,
    device="cuda",  # Use GPU
    **ppo_config
)
```

### 5.3 Logging: Matplotlib (No TensorFlow/TensorBoard)

**Alternative to TensorBoard: Live Matplotlib Plots**

```python
import matplotlib.pyplot as plt
from collections import deque
import json

class TrainingLogger:
    """Simple logger using matplotlib - no TensorFlow needed"""
    
    def __init__(self, log_dir="./logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # History buffers
        self.rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.stages = []
        self.steps = []
        
        # Setup live plot
        plt.ion()  # Interactive mode
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        
    def log(self, step, reward, episode_length, success, stage):
        """Log training metrics"""
        self.steps.append(step)
        self.rewards.append(reward)
        self.episode_lengths.append(episode_length)
        self.stages.append(stage)
        
        # Calculate running success rate
        recent_success = self.success_rates[-100:] if self.success_rates else []
        recent_success.append(1 if success else 0)
        self.success_rates.append(sum(recent_success) / len(recent_success))
        
        # Save to JSON periodically
        if step % 10000 == 0:
            self.save_logs()
    
    def update_plots(self):
        """Update matplotlib plots"""
        if len(self.steps) < 2:
            return
            
        # Clear axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot 1: Episode Reward
        self.axes[0, 0].plot(self.steps, self.rewards, 'b-', alpha=0.3)
        # Rolling average
        if len(self.rewards) > 100:
            rolling = [sum(self.rewards[max(0,i-100):i])/min(i,100) 
                      for i in range(1, len(self.rewards)+1)]
            self.axes[0, 0].plot(self.steps, rolling, 'b-', linewidth=2)
        self.axes[0, 0].set_xlabel('Steps')
        self.axes[0, 0].set_ylabel('Episode Reward')
        self.axes[0, 0].set_title('Training Reward')
        self.axes[0, 0].grid(True)
        
        # Plot 2: Episode Length
        self.axes[0, 1].plot(self.steps, self.episode_lengths, 'g-', alpha=0.3)
        self.axes[0, 1].set_xlabel('Steps')
        self.axes[0, 1].set_ylabel('Episode Length')
        self.axes[0, 1].set_title('Episode Duration')
        self.axes[0, 1].grid(True)
        
        # Plot 3: Success Rate
        self.axes[1, 0].plot(self.steps, self.success_rates, 'r-')
        self.axes[1, 0].axhline(y=0.8, color='k', linestyle='--', label='Threshold')
        self.axes[1, 0].set_xlabel('Steps')
        self.axes[1, 0].set_ylabel('Success Rate (last 100)')
        self.axes[1, 0].set_title('Success Rate')
        self.axes[1, 0].set_ylim([0, 1])
        self.axes[1, 0].grid(True)
        
        # Plot 4: Curriculum Stage
        self.axes[1, 1].plot(self.steps, self.stages, 'm-')
        self.axes[1, 1].set_xlabel('Steps')
        self.axes[1, 1].set_ylabel('Stage')
        self.axes[1, 1].set_title('Curriculum Stage')
        self.axes[1, 1].set_yticks([1, 2, 3, 4])
        self.axes[1, 1].set_yticklabels(['Standing', 'Balance', 'Stepping', 'Walking'])
        self.axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.pause(0.01)  # Update display
    
    def save_logs(self):
        """Save logs to JSON file"""
        data = {
            'steps': self.steps,
            'rewards': self.rewards,
            'episode_lengths': self.episode_lengths,
            'success_rates': self.success_rates,
            'stages': self.stages,
        }
        with open(f"{self.log_dir}/training_log.json", 'w') as f:
            json.dump(data, f)
    
    def save_final_plots(self):
        """Save plots as images"""
        self.update_plots()
        plt.savefig(f"{self.log_dir}/training_curves.png", dpi=150)
        print(f"Saved plots to {self.log_dir}/training_curves.png")
```

**Usage in Training:**
```python
logger = TrainingLogger("./logs")

for step in range(total_steps):
    # ... training code ...
    
    # Log every episode
    if episode_done:
        logger.log(step, episode_reward, episode_length, success, curriculum.stage)
    
    # Update plots every 1000 steps
    if step % 1000 == 0:
        logger.update_plots()

# Save final plots
logger.save_final_plots()
```

### 5.4 Dependencies (No TensorFlow!)

```bash
# Required packages
pip install torch           # PyTorch (GPU support)
pip install stable-baselines3  # PPO implementation (uses PyTorch)
pip install gymnasium       # RL environment interface
pip install mujoco          # Physics simulation
pip install matplotlib      # Plotting (instead of TensorBoard)
pip install numpy
```

**Note:** Stable-Baselines3 v2.0+ is pure PyTorch - no TensorFlow dependency!



## 6. Expected Training Timeline

| Stage | Steps | Time (12 envs, RTX 2050) | What to Expect |
|-------|-------|--------------------------|----------------|
| 1. Standing | 300k-500k | ~20-30 min | Robot learns to not fall |
| 2. Balance | 300k-500k | ~20-30 min | Robot recovers from pushes |
| 3. Stepping | 500k-1M | ~40-60 min | Robot lifts feet alternately |
| 4. Walking | 2M-5M | ~3-6 hours | Robot walks forward |
| **Total** | **3M-7M** | **~4-8 hours** | |

---

## 7. File Structure

```
go_bdx/
├── README.md
├── RL_WALKING_PLAN.md           # This document
├── go_bdx.urdf
├── go_bdx.xml
├── convert_urdf.py
├── keyboard_control.py
├── meshes/
│
├── rl/
│   ├── go_bdx_env.py            # Gymnasium environment
│   ├── rewards.py               # Reward functions per stage
│   ├── curriculum.py            # Curriculum scheduler
│   ├── train.py                 # Main training script
│   ├── test_policy.py           # Evaluation script
│   └── visualize.py             # Training visualization
│
├── checkpoints/
│   ├── periodic/                # Every 100k steps
│   ├── curriculum/              # Stage completion
│   │   ├── standing_complete.zip
│   │   ├── balance_complete.zip
│   │   ├── stepping_complete.zip
│   │   └── walking_complete.zip
│   └── best/                    # Best reward
│
└── logs/                        # TensorBoard logs
```

---

## 8. Quick Reference Commands

```bash
# Convert URDF (if needed)
python convert_urdf.py

# Test manually with keyboard
python keyboard_control.py

# Train from scratch
python rl/train.py --stage 1 --num_envs 12

# Resume training
python rl/train.py --checkpoint checkpoints/curriculum/standing_complete.zip --stage 2

# Test trained policy
python rl/test_policy.py --model checkpoints/best/best_model.zip

# View training logs
# View plots in matplotlib window (opens automatically)
```
