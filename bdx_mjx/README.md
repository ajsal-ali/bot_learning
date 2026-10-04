# bdx_mjx — GO-BDX walking with MJX on the GPU

Thousands of robots simulated in parallel on the GPU (MuJoCo MJX), trained with
Brax PPO. Replaces the old `rl/` pipeline (kept for reference, not used).

```
bdx_mjx/
  assets/build_model.py   rebuilds the robot XMLs from ../go_bdx.xml (masses, feet, motors, sensors)
  assets/xmls/            go_bdx_train.xml (no meshes, MJX) + go_bdx_view.xml (same physics + meshes)
  joystick.py             the task: follow a (vx, vy, yaw-rate) command with a periodic gait
  randomize.py            per-env domain randomization (friction, masses, motor gains, ...)
  train.py                training entry point (presets, checkpoints, logs, policy export)
  policy.py               NumPy-only policy + WalkController (same code for sim and real robot)
  play.py                 watch it at real time in the MuJoCo viewer, or record a video
```

## Setup

JAX only runs on the GPU on **Linux** (or WSL2 on Windows). Native Windows works,
but on the CPU: fine for testing and `play.py`, far too slow for real training.

**Training PC (Linux + NVIDIA):**
```bash
conda env create -f bdx_mjx/environment.yml      # creates env "bdx"
conda activate bdx
python -c "import jax; print(jax.devices())"     # must show CudaDevice
```

**This Windows PC** (already set up in the `grounding` conda env, CPU JAX):
```bash
pip install -r bdx_mjx/requirements.txt
```
To use the RTX 2050 here: `wsl --install` (Ubuntu), then do the Linux setup inside WSL.

## Train

Run from the repo root:
```bash
python -m bdx_mjx.train --preset gpu         # >= 8 GB VRAM: 8192 envs, 200M steps
python -m bdx_mjx.train --preset gpu_small   # 4-6 GB VRAM (e.g. RTX 2050 in WSL2): 2048 envs
python -m bdx_mjx.train --preset cpu         # no GPU: ~30 min, only checks that it learns
python -m bdx_mjx.train --preset cpu_test    # 2-minute smoke test of the whole pipeline

# options
--num-timesteps 300000000   --num-envs 4096   --seed 1   --run-name myrun
--resume runs/<run>/checkpoints/latest.pkl    # continue from a checkpoint
--impl warp                                   # MuJoCo Warp backend (Linux GPU, untested here)
```
The first iteration spends a few minutes compiling. After that it prints one line per
eval. Watch:
- **episode length** climbing to the max (1000 = 20 s without falling)
- **tracking_lin_vel / tracking_ang_vel** going up (it's following the command)
- `runs/<run>/progress.png` for the curves

Copy `runs/<run>/` back to any PC to watch it, since `policy.npz` needs only NumPy and MuJoCo.

## Watch / record

```bash
python -m bdx_mjx.play runs/<run>                       # viewer, real time (1x)
python -m bdx_mjx.play runs/<run> --cmd 0.3 0 0         # start walking forward at 0.3 m/s
python -m bdx_mjx.play runs/<run> --video walk.mp4 --seconds 10 --cmd 0.3 0 0
```
Viewer keys: ↑/↓ forward speed, ←/→ turn, PgUp/PgDn sideways, Home = stop, End = reset.

## Why it's built this way

**Time scale.** Training runs as fast as the GPU allows. The physics uses a fixed
time step, so speed has no effect on what is learned. What must match the real
robot is the **control rate (50 Hz)**, which is fixed in the config. Only `play.py`
runs at 1x, because that's for watching.

**Robot model.** The CAD export had placeholder physics: a 28 kg base (its mass
came from the mesh volume at water density), 1e-4 inertias everywhere, collision
on the visual meshes, unlimited torque, and kp=2000. `assets/build_model.py`
rebuilds it:
- ~11.6 kg total, inertia computed from each link's mesh
- flat box collision under each foot, fitted to the sole
- torque-limited PD motors (23.7 Nm, GO-M8010-6-like)
- head joints frozen
- IMU site with x forward

**The link masses are estimates.** Weigh your parts, edit `MASSES` in
`build_model.py`, and run `python bdx_mjx/assets/build_model.py`. Domain
randomization covers ±10%, plus up to 1 kg of torso payload.

**Joint names lie.** Going by the CAD names: `*_hip_roll` turns about the vertical
axis (yaw), `*_hip_pitch` about the forward axis (roll), and `*_hip_yaw` about the
lateral axis (pitch). The names are kept so they match your hardware config.
`constants.py` groups the joints by what they really do.

**One task, no stage curriculum.** The policy always gets a velocity command,
and standing is just the zero command. The reward is velocity tracking plus
gait-clock foot height and feet air time (these break the "stand still and
collect reward" optimum), minus penalties for tilt, jerk, torque, slip and
legs crossing.

**Sim-to-real.** The actor only sees what the robot can measure: gyro, gravity
direction from the IMU, joint encoders, last action, command and gait clock.
It also gets observation noise, random pushes and randomized friction, gains and
masses. The critic gets privileged sim state (true velocity, contacts) during
training only. `policy.WalkController` is the deploy code: give it the IMU and
encoder readings at 50 Hz and it returns 10 joint targets.
`play.py` runs exactly that loop, and it is verified to match the training env's
observations and actions to float precision.

## Things to tune if it doesn't walk well

All of these live in `joystick.default_config()`:
- `reward_config.scales`: e.g. raise `feet_air_time` if it shuffles, raise
  `orientation` if it leans
- `gait_config.freq_range` and `swing_height`: step frequency and foot lift
- `command_config`: the speed ranges it trains on
- `action_scale`: how far each action can move a joint from the standing pose
