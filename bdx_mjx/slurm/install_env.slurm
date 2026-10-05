#!/bin/bash
#SBATCH --job-name=bdx_env
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/bdx_env_%j.out
#SBATCH --error=logs/bdx_env_%j.err

# STEP 2 of 2: install into your existing conda env "bdx", OFFLINE, from the packages that
# bdx_mjx/slurm/fetch_env.sh downloaded on the login node (compute nodes have
# no internet). No GPU needed - this only unpacks and links files.
#
#   cd /scratch/scratch26/23me36008/bot_learning
#   bash bdx_mjx/slurm/fetch_env.sh            # login node, first
#   mkdir -p logs
#   sbatch bdx_mjx/slurm/install_env.slurm
#
# Installs into the existing "bdx" env in place (nothing is deleted or
# recreated). train_gpu.slurm activates "bdx".

set -e

PROJECT=/scratch/scratch26/23me36008/bot_learning
ENV_NAME=bdx
export MAMBA_ROOT_PREFIX=/scratch/scratch26/23me36008/micromamba
export CONDA_OVERRIDE_CUDA=12.0
MM="$HOME/.local/bin/micromamba"
EXPLICIT=$PROJECT/bdx_mjx/slurm/env_explicit.txt

echo "===== JOB ====="
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)"
source ~/.bashrc
ENV_PREFIX=$(conda env list | awk -v n="$ENV_NAME" '$1 == n {print $NF}')
[ -d "$ENV_PREFIX" ] || { echo "conda env '$ENV_NAME' not found" >&2; exit 1; }
echo "installing into $ENV_PREFIX"
[ -f "$EXPLICIT" ] || { echo "missing $EXPLICIT - run fetch_env.sh on the login node first" >&2; exit 1; }
[ -x "$MM" ] || { echo "missing $MM - run fetch_env.sh on the login node first" >&2; exit 1; }

echo "===== INSTALL (offline) ====="
"$MM" install -y --offline -p "$ENV_PREFIX" -f "$EXPLICIT"

echo "===== CHECK ====="
"$MM" list -p "$ENV_PREFIX" | grep -E " (python|jax|jaxlib|flax|brax|mujoco|mujoco-mjx) "
# No GPU on this node: JAX warns and falls back to CPU - expected. The GPU
# itself is checked at the start of train_gpu.slurm.
"$ENV_PREFIX/bin/python" - <<'EOF'
import jax, jax.numpy as jp, brax, flax, mujoco
from mujoco import mjx
print("jax", jax.__version__, "| brax", brax.__version__, "| flax", flax.__version__,
      "| mujoco", mujoco.__version__)
m = mujoco.MjModel.from_xml_string(
    "<mujoco><worldbody><geom type='plane' size='1 1 .1'/>"
    "<body pos='0 0 .2'><freejoint/><geom size='.05'/></body></worldbody></mujoco>")
mx = mjx.put_model(m)
d = jax.jit(jax.vmap(lambda _: mjx.step(mx, mjx.make_data(m))))(jp.arange(256))
print("mjx ok")
EOF

echo "===== DONE - now: sbatch bdx_mjx/slurm/train_gpu.slurm ====="
