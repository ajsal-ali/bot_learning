# GO-BDX Reinforcement Learning Training

Train the GO-BDX bipedal robot to walk using PPO (Proximal Policy Optimization) with curriculum learning.

## 📋 Dependencies

### Required Packages

```bash
pip install gymnasium mujoco numpy stable-baselines3 matplotlib tqdm rich
```

### Versions Tested
- Python 3.10+
- gymnasium >= 0.29.0
- mujoco >= 3.0.0
- stable-baselines3 >= 2.0.0
- matplotlib >= 3.7.0
- tqdm >= 4.65.0
- rich >= 13.0.0

### Quick Install

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. Train with Visualization (Recommended)

```bash
cd ~/Documents/rl_task_ws/go_bdx
python3 -m rl.train_visual
```

This will:
- Train 4 parallel environments
- Show live MuJoCo rendering
- Display real-time training plots
- Save checkpoints every 100k steps
- Use curriculum learning (4 stages)

### 2. Test a Trained Policy

```bash
python3 -m rl.test_policy checkpoints/final_model.zip --episodes 10
```

---

## 📚 Scripts Overview

### `train_visual.py` - Main Training Script

Train with live visualization and plotting.

**Usage:**
```bash
python3 -m rl.train_visual [OPTIONS]
```

**Arguments:**
| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--timesteps` | `-t` | int | 2,000,000 | Total training timesteps |
| `--n-envs` | `-n` | int | 4 | Number of parallel environments |
| `--stage` | `-s` | int | 1 | Starting curriculum stage (1-4) |
| `--resume` | `-r` | str | None | Resume from checkpoint path |
| `--save-dir` | | str | ./checkpoints | Checkpoint save directory |
| `--log-dir` | | str | ./logs | Logging directory |
| `--realtime` | | flag | False | Run at real-time speed (for visualization) |

**Examples:**
```bash
# Train with 8 parallel environments for 5M steps
python3 -m rl.train_visual -n 8 -t 5000000

# Resume from checkpoint
python3 -m rl.train_visual --resume checkpoints/periodic/rl_model_500000_steps.zip

# Start from walking stage (skip standing/balance)
python3 -m rl.train_visual --stage 4

# Real-time visualization (slow, for checking)
python3 -m rl.train_visual -n 1 --realtime
```

---

### `train.py` - Headless Training

Train without visualization (faster, for long runs).

**Usage:**
```bash
python3 -m rl.train [OPTIONS]
```

**Arguments:** (Same as `train_visual.py` minus `--realtime`)

**Examples:**
```bash
# Headless training with 16 envs (faster)
python3 -m rl.train -n 16 -t 10000000

# Save to custom directory
python3 -m rl.train --save-dir ./my_checkpoints --log-dir ./my_logs
```

---

### `test_policy.py` - Policy Evaluation

Test a trained policy and measure performance.

**Usage:**
```bash
python3 -m rl.test_policy MODEL_PATH [OPTIONS]
```

**Arguments:**
| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `model_path` | | str | required | Path to .zip model file |
| `--episodes` | `-n` | int | 5 | Number of episodes to run |
| `--no-render` | | flag | False | Disable rendering |

**Examples:**
```bash
# Test with rendering (default)
python3 -m rl.test_policy checkpoints/final_model.zip

# Run 20 test episodes
python3 -m rl.test_policy checkpoints/final_model.zip -n 20

# Test without visualization
python3 -m rl.test_policy checkpoints/final_model.zip --no-render
```

---

## 🎓 Curriculum Learning Stages

The training uses a 4-stage curriculum:

| Stage | Task | Success Criteria | Reward Focus |
|-------|------|------------------|--------------|
| **1. Standing** | Stay upright without falling | Survive 250 steps (5s) | Height + upright + stillness |
| **2. Balance** | Withstand random pushes | Survive 3+ pushes | Standing reward + recovery |
| **3. Stepping** | Alternate foot contact | >70% contact accuracy | Gait phase + foot clearance |
| **4. Walking** | Move at target velocity | Track velocity ±0.15 m/s | Velocity tracking + stability |

**Progression:** Advances to next stage when **80% success rate** over last 100 episodes.

---

## 🧠 Environment Details

### Observation Space (54 dims)

| Component | Dims | Description |
|-----------|------|-------------|
| Body height | 1 | CoM height above ground |
| Body orientation | 4 | Quaternion (wxyz) |
| Linear velocity | 3 | Body velocity (xyz) |
| Angular velocity | 3 | Body rotation rate |
| Joint positions | 10 | Leg joint angles |
| Joint velocities | 10 | Leg joint speeds |
| Previous action (t-1) | 10 | Action history |
| Previous action (t-2) | 10 | Action history |
| Target velocity | 1 | Desired forward speed |
| Gait phase | 2 | sin/cos of gait cycle |

### Action Space (10 dims)

- 10 leg joint position targets
- Normalized to [-1, 1]
- Scaled per joint (hip_pitch=±0.8, knee=±1.0, etc.)
- Rate limited to 0.1 change per step (smooth movements)

### Environment Parameters

```python
from rl.go_bdx_env import GoBdxEnv

env = GoBdxEnv(
    render_mode="human",        # "human" or None
    max_episode_steps=250,      # Steps per episode (5s at 50Hz)
    curriculum_stage=1,         # 1-4
    target_velocity=0.0,        # m/s (for walking stage)
    randomize=True,             # Random init perturbations
    obs_noise=False,            # Sensor noise (for sim2real)
)
```

**Observation Noise** (optional, for sim2real):
- Set `obs_noise=True` to add sensor noise
- Joint position: ±0.03 rad
- Joint velocity: ±2.5 rad/s
- IMU: ±0.1 rad/s
- Helps policy transfer to real hardware

---

## 📊 Training Outputs

### Checkpoints Directory
```
checkpoints/
├── periodic/
│   ├── rl_model_100000_steps.zip
│   ├── rl_model_200000_steps.zip
│   └── ...
├── curriculum/
│   ├── stage_1_completed.zip
│   ├── stage_2_completed.zip
│   └── ...
└── final_model.zip
```

### Logs Directory
```
logs/
└── run_20260402_141523/
    ├── training_progress.png    # Live-updated plot
    ├── episode_data.csv         # Raw episode data
    └── curriculum_state.json    # Current curriculum progress
```

---

## 🎮 Control Frequency

- **Policy frequency:** 50 Hz (20ms per action)
- **Physics frequency:** 500 Hz (2ms timestep)
- **Control ratio:** 10 physics steps per action

---

## 🤖 PPO Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning rate | 3e-4 | Linear decay to 0 |
| n_steps | 2048 | Rollout length per env |
| batch_size | 256 | Minibatch size |
| n_epochs | 10 | Gradient updates per rollout |
| gamma | 0.99 | Discount factor |
| gae_lambda | 0.95 | GAE parameter |
| clip_range | 0.2 | PPO clipping |
| ent_coef | 0.005 | Entropy bonus |
| Policy network | [256, 256, 128] | 3 hidden layers with ELU |

---

## 📈 Expected Training Time

| Stage | Success Criteria | Typical Time (4 envs) |
|-------|------------------|----------------------|
| **Standing** | Survive 5s | 2-4 hours |
| **Balance** | Survive pushes | 4-8 hours |
| **Stepping** | Gait coordination | 8-12 hours |
| **Walking** | Track velocity | 12-24 hours |

**Total:** ~30-50 hours for full curriculum

**Tips for faster training:**
- Increase `--n-envs` (diminishing returns after 8-16)
- Use headless training (`train.py` instead of `train_visual.py`)
- Skip early stages if resuming: `--stage 3`

---

## 🛠️ Troubleshooting

### "Model not found: go_bdx.xml"
```bash
# Make sure you're in the go_bdx directory
cd ~/Documents/rl_task_ws/go_bdx

# Check if go_bdx.xml exists
ls go_bdx.xml
```

### "Observation space mismatch"
If resuming training with a different observation space (e.g., old 44-dim model vs new 54-dim):
- Train from scratch, or
- Use matching model architecture

### Training is very slow
- Reduce `--n-envs` (MuJoCo is CPU-bound)
- Use headless mode (`train.py`)
- Check CPU usage (should be near 100%)

### Robot immediately falls
- Normal in early training (<1 hour)
- Check reward is improving (use plots)
- Try reducing action rate limit if too jerky

### GPU not used
- Expected - MuJoCo simulation dominates (95% CPU time)
- Neural network is small (~273k params)
- GPU usage will be low (5-10%)

---

## 📝 File Structure

```
rl/
├── README.md              # This file
├── __init__.py            # Package exports
├── go_bdx_env.py          # Gymnasium environment
├── curriculum.py          # Curriculum scheduler
├── train_visual.py        # Training with visualization
├── train.py               # Headless training
└── test_policy.py         # Policy evaluation
```

---

## 🎯 Next Steps After Training

1. **Test the policy:**
   ```bash
   python3 -m rl.test_policy checkpoints/final_model.zip -n 20
   ```

2. **Resume from checkpoint:**
   ```bash
   python3 -m rl.train_visual --resume checkpoints/periodic/rl_model_500000_steps.zip
   ```

3. **Train with noise for sim2real:**
   - Modify `go_bdx_env.py`: set `obs_noise=True` in environment creation
   - Retrain from scratch or fine-tune existing policy

4. **Export policy for deployment:**
   ```python
   from stable_baselines3 import PPO
   model = PPO.load("checkpoints/final_model.zip")
   # Deploy to ROS2, real hardware, etc.
   ```

---

## 📖 References

- **Algorithm:** [PPO Paper](https://arxiv.org/abs/1707.06347)
- **Curriculum Learning:** Staged progression from simple to complex tasks
- **Sim2Real:** Observation noise + domain randomization
- **Inspiration:** Open Duck Mini, Disney BDX

---

## 💬 Support

For issues or questions:
1. Check this README
2. Review training plots in `logs/`
3. Test with minimal config: `python3 -m rl.train_visual -n 1 -t 100000`

**Happy Training! 🤖🚶**
