# bdx_mjx — GO-BDX walking with MJX on the GPU

Thousands of robots simulated in parallel on the GPU (MuJoCo MJX), trained with
Brax PPO. Replaces the old `rl/` pipeline (kept for reference, not used).

```
bdx_mjx/
  assets/build_model.py   rebuilds the robot XMLs from ../go_bdx.xml (masses, feet, motors, sensors)
  assets/xmls/            go_bdx_train.xml (no meshes, MJX) + go_bdx_view.xml (same physics + meshes)
  joystick.py             the task: follow a (vx, vy, yaw-rate) command with a periodic gait
  randomize.py            per-env domain randomization (friction, masses, motor gains, ...)
  train.py                training entry point (curriculum phases, presets, checkpoints, logs, export)
  policy.py               NumPy-only policy + WalkController (same code for sim and real robot)
  play.py                 watch it at real time, record a video, or run the push-survival test
  mjx_env.py, wrapper.py  env base class + Brax wrappers, vendored from MuJoCo Playground
  slurm/                  ParamShakti scripts (see Setup)
  requirements-lock-linux.txt   exact package versions for the cluster
```

## Setup

JAX only runs on the GPU on **Linux**. Native Windows works, but on the CPU:
fine for testing and `play.py`, far too slow for real training.

**ParamShakti (CentOS 7, V100).** JAX can't run on the host (glibc 2.17 is
too old), so it runs in a small Apptainer container (Debian, Python 3.11),
with the packages in a venv on scratch. One-time setup:
```bash
# 1. get the ~93 Linux wheels into /scratch/scratch26/23me36008/bdx_env/wheels/, either
bash bdx_mjx/slurm/fetch_env.sh                  # on the login node (slow network), or
python bdx_mjx/slurm/download_wheels_pc.py       # on your PC, then scp wheels_linux/* there
                                                 # (then run fetch_env.sh once: it pulls the image)
# 2. install them offline in a job (shared partition)
mkdir -p logs && sbatch bdx_mjx/slurm/install_env.slurm
```
Then train with `sbatch bdx_mjx/slurm/train_gpu.slurm`. Its header explains the
CPU/RAM/env-count sizing. `slurm/env.sh` holds the paths and picks a working
apptainer module.

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
--start-phase 3                               # ...skipping curriculum phases already done
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
python -m bdx_mjx.play runs/<run> --push-test           # % of pushes survived, by force
```
Viewer keys: ↑/↓ forward speed, ←/→ turn, PgUp/PgDn sideways, Home = stop, End = reset,
Insert = push the torso (`--push-force`, default 40 N). Or use MuJoCo's own: double-click
the body, then Ctrl + right-drag.

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

**One task, with a push curriculum.** The policy always gets a velocity command,
and standing is just the zero command (20% of the time). The reward is velocity
tracking plus gait-clock foot height and feet air time, minus penalties for
tilt, jerk, torque, slip and legs crossing. The gait-clock and air-time terms
are what stop it from learning to "stand still and collect reward".

There are no separate standing/balance/stepping stages: the reward never changes,
because swapping reward functions mid-training breaks the value function. What
ramps up is the pushes. A push is a horizontal force on the torso, in a random
direction, for 0.1-0.2 s, every 4-8 s, while standing or walking:

| phase | share of steps | push force | why |
|---|---|---|---|
| 1 walk | 30% | 5-20 N | below what the stiff stance survives, so it learns the gait first |
| 2 push | 30% | 10-40 N | moderate pushes; some need a step |
| 3 hard_push | 40% | 10-55 N | beyond the passive limit, so it has to step to recover |

These limits are measured on this model. With zero action, a 0.2 s push topples
it at 30 N forward and 50 N backward or sideways. 55 N for 0.2 s is about 0.95 m/s
of velocity change, which by the capture point needs one big or two normal steps.
Much bigger pushes are physically unrecoverable for this leg length and would only
teach it that falling is unavoidable. `play.py --push-test` reports how many it
actually survives.

**Sim-to-real.** The actor only sees what the robot can measure: gyro, gravity
direction from the IMU, joint encoders, last action, command and gait clock.
It also gets observation noise, random pushes, randomized friction, gains and
masses, a 0 or 20 ms actuation delay, and per-joint encoder offsets (±0.03 rad).
At zero command it is rewarded for standing still with both feet down, and it
only steps when it needs to, for example after a push. The critic gets privileged sim state (true velocity, contacts) during
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
