#!/bin/bash
#SBATCH --job-name=bdx_env
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/bdx_env_%j.out
#SBATCH --error=logs/bdx_env_%j.err

# Installs the MJX/Brax stack into the existing conda env "bdx". Needs no GPU:
# conda only solves, downloads and unpacks. Run as a job because a conda solve
# on the login node gets killed (>30 min of CPU), and the account gets locked
# if that repeats.
#
# Partition: any CPU partition. "shared" is a guess - check the names with
# `sinfo -s` and change the line above if needed (a wrong name fails instantly).
#
#   cd /scratch/scratch26/23me36008/bot_learning
#   mkdir -p logs
#   sbatch bdx_mjx/slurm/install_env.slurm
#   tail -f logs/bdx_env_<jobid>.out
#
# Only 4 CPUs: the solve is single-threaded and the rest is downloading.
# The GPU itself is checked at the start of train_gpu.slurm, which stops
# immediately if JAX cannot see it.
#
# Why python=3.11 is in the package list although the env already has 3.11:
# it is a pin, not a reinstall. Without it the solver is free to change Python
# to satisfy some package, which would break everything else in the env. It
# also shrinks the search space, so the solve is faster.

set -e

ENV_NAME=bdx
# CUDA packages are several GB; keep the download cache off the home quota.
export CONDA_PKGS_DIRS=/scratch/scratch26/23me36008/conda_pkgs
mkdir -p "$CONDA_PKGS_DIRS"
# This node has no NVIDIA driver, so conda would see no CUDA and refuse every
# cuda12 jaxlib build. Tell it to solve as if a CUDA 12 driver were present
# (the V100 nodes have one; train_gpu.slurm verifies that at runtime).
export CONDA_OVERRIDE_CUDA=12.0

echo "===== JOB ====="
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)"

source ~/.bashrc
conda activate "$ENV_NAME"
echo "conda $(conda --version)   env: $CONDA_PREFIX   python: $(python --version 2>&1)"

echo "===== INTERNET ====="
if ! curl -sSI -m 30 https://conda.anaconda.org/conda-forge/ > /dev/null; then
    echo "!! This compute node cannot reach conda.anaconda.org."
    echo "!! Install from the login node instead, with the fast solver only:"
    echo "!!   conda install -n $ENV_NAME --solver=libmamba ...   (see the README)"
    exit 1
fi
echo "conda-forge reachable"

echo "===== SOLVER ====="
# libmamba solves this in ~1 min; the classic solver can take an hour.
SOLVER_ARG=""
if conda list -n base 2>/dev/null | grep -q "^conda-libmamba-solver"; then
    SOLVER_ARG="--solver=libmamba"
    echo "using libmamba"
else
    echo "conda-libmamba-solver not installed in base - using the classic solver (slow)"
fi

echo "===== INSTALL ====="
# jax < 0.10      : brax calls jax.device_put_replicated, removed in jax 0.10
# *cuda12* jaxlib : CUDA 13 builds do not support the V100 (sm_70)
conda install -y -n "$ENV_NAME" $SOLVER_ARG --override-channels -c conda-forge \
    "python=3.11" \
    "jax<0.10" "jaxlib<0.10=*cuda12*" "cuda-version=12.*" \
    "flax=0.12.6" optax "brax=0.12.5" \
    "mujoco=3.14.0" "mujoco-mjx=3.14.0" \
    orbax-checkpoint ml-collections \
    matplotlib pillow ffmpeg mediapy

echo "===== INSTALLED ====="
conda list -n "$ENV_NAME" | grep -E "^(python|jax|jaxlib|flax|brax|mujoco|mujoco-mjx|cuda-version|ml-collections) "

echo "===== IMPORT CHECK (CPU) ====="
conda list -n "$ENV_NAME" jaxlib | grep -q cuda12     && echo "jaxlib is a cuda12 build"     || { echo "!! jaxlib is NOT a cuda12 build - it will not use the GPU"; exit 1; }
# No GPU here, so JAX falls back to CPU (it prints a warning about CUDA - expected).
python - <<'EOF'
import jax, jax.numpy as jp, brax, flax, mujoco
from mujoco import mjx
print("jax", jax.__version__, "| brax", brax.__version__, "| flax", flax.__version__,
      "| mujoco", mujoco.__version__)
m = mujoco.MjModel.from_xml_string(
    "<mujoco><worldbody><geom type='plane' size='1 1 .1'/>"
    "<body pos='0 0 .2'><freejoint/><geom size='.05'/></body></worldbody></mujoco>")
mx = mjx.put_model(m)
d = jax.jit(jax.vmap(lambda _: mjx.step(mx, mjx.make_data(m))))(jp.arange(1024))
print("mjx ok: stepped 1024 envs on", d.qpos.devices())
EOF

echo "===== DONE - env $ENV_NAME is installed; submit bdx_mjx/slurm/train_gpu.slurm ====="
