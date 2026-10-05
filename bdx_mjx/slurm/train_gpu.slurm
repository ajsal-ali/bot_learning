#!/bin/bash
#SBATCH --job-name=bdx_walk
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/bdx_%j.out
#SBATCH --error=logs/bdx_%j.err

# GO-BDX walking, MJX + Brax PPO on one V100 (16 GB).
#
#   cd /scratch/scratch26/23me36008/bot_learning
#   mkdir -p logs                      # SLURM will not create it for --output
#   sbatch bdx_mjx/slurm/train_gpu.slurm
#   tail -f logs/bdx_<jobid>.out
#
# Sizing
# ------
# Unlike the mavrl jobs, this one is GPU-bound: physics, policy and PPO update
# all run on the GPU inside one jitted XLA program. The CPU only launches it,
# compiles it once at the start (~2-5 min, XLA uses a few threads for that) and
# writes logs/checkpoints at each eval. So:
#   --cpus-per-task=8   enough for XLA compilation; 40 would just sit idle.
#   --mem=32G           host RAM: python + JAX + compile peaks at ~6-10 GB.
#                       32G leaves headroom; lower it if the queue is tight.
#
# Envs per GPU
# ------------
# The robot is small (16 DoF, 2 foot boxes, <=8 contacts), so one env's state
# is a few tens of KB. GPU memory is not the limit; 8192 envs fits on 16 GB
# with lots of room. 8192 is the standard for PPO here: more envs means more
# steps/s, but each PPO update sees the same batch_size*num_minibatches, so
# going higher mostly spends compute without learning faster.
#   NUM_ENVS=8192   (default, recommended, and the max for the gpu preset)
#   NUM_ENVS=4096   if you ever hit GPU out-of-memory.
# Brax requires batch_size * num_minibatches (256 * 32 = 8192) to be a
# multiple of NUM_ENVS, so use 8192, 4096, 2048, ...
#
# Time: expected a few hundred thousand env steps/s on a V100 (not measured
# yet - the eval lines print the real steps/s). 200M steps -> roughly 20-60 min,
# plus ~2-5 min of XLA compilation at the start of each of the 3 curriculum
# phases (walk -> push -> hard_push, see train.py). --time=06:00:00 is a
# generous cap; the job ends when training does.
#
# What to watch in logs/bdx_<jobid>.out:
#   episode length  -> should reach 1000 (20 s, no falls) during phase 1 and
#                      stay near it as pushes get harder in phases 2-3
#   tracking_lin_vel / tracking_ang_vel -> rising = following the command
#   the push-test table at the end -> % of pushes survived per force.

set -e

echo "===== JOB ====="
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

PROJECT=/scratch/scratch26/23me36008/bot_learning
cd "$PROJECT"
mkdir -p logs runs
# Container + venv (see env.sh: CentOS 7's glibc is too old for JAX on the host).
source bdx_mjx/slurm/env.sh
[ -x "$VENV/bin/python" ] || { echo "no venv - run fetch_env.sh + install_env.slurm first" >&2; exit 1; }
# --nv passes the host's NVIDIA driver and the allocated V100 into the container.
PY="in_container --nv $SIF $VENV/bin/python"

# JAX/XLA: pre-allocate 90% of the GPU for itself (fast). Exported env vars
# pass into the container automatically.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
# Keep the few CPU-side libraries from spawning 40-thread pools.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

echo "===== CHECK ====="
$PY - <<'EOF'
import jax, brax, mujoco
print("jax", jax.__version__, "brax", brax.__version__, "mujoco", mujoco.__version__)
print("devices", jax.devices())
assert jax.default_backend() == "gpu", "JAX is not on the GPU inside the container"
EOF

RUN_NAME=walk_${SLURM_JOB_ID}
NUM_ENVS=8192
TIMESTEPS=200000000
# To continue an earlier run, set e.g.
#   RESUME="--resume runs/walk_1234567/checkpoints/latest.pkl"
RESUME=""
# ...and to skip phases already done, add e.g. "--start-phase 3" to RESUME.

echo "===== TRAIN ====="
$PY -u -m bdx_mjx.train \
    --preset         gpu \
    --num-envs       "$NUM_ENVS" \
    --num-timesteps  "$TIMESTEPS" \
    --run-name       "$RUN_NAME" \
    --runs-dir       runs \
    --seed           0 \
    $RESUME

echo "===== PUSH TEST ====="
# Plain MuJoCo on the CPU, no rendering: survival rate vs push force.
$PY -m bdx_mjx.play "runs/$RUN_NAME" --push-test | tee "runs/$RUN_NAME/push_test.txt"

# No videos here: the minimal container has no OpenGL. Copy runs/$RUN_NAME/
# to your PC and use play.py there (viewer, or --video).

echo "===== DONE ====="
echo "run dir: $PROJECT/runs/$RUN_NAME"
echo "copy it to your PC and run:  python -m bdx_mjx.play runs/$RUN_NAME"
