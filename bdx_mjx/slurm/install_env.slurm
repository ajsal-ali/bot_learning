#!/bin/bash
#SBATCH --job-name=bdx_env
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/bdx_env_%j.out
#SBATCH --error=logs/bdx_env_%j.err

# STEP 2 of 2: install the downloaded wheels into $VENV (on scratch), offline,
# inside the container. No GPU or internet needed.
#
#   cd /scratch/scratch26/23me36008/bot_learning
#   bash bdx_mjx/slurm/fetch_env.sh        # login node, first
#   mkdir -p logs
#   sbatch bdx_mjx/slurm/install_env.slurm

set -e
cd /scratch/scratch26/23me36008/bot_learning
source bdx_mjx/slurm/env.sh

echo "===== JOB ====="
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)"
[ -f "$SIF" ] && [ -d "$WHEELS" ] || { echo "run bdx_mjx/slurm/fetch_env.sh on the login node first" >&2; exit 1; }

echo "===== INSTALL (offline) ====="
rm -rf "$VENV"
in_container "$SIF" python -m venv "$VENV"
in_container "$SIF" "$VENV/bin/pip" install --no-deps --no-index --find-links "$WHEELS" \
    -r "$PROJECT/bdx_mjx/requirements-lock-linux.txt"
# --no-deps skipped the resolver; confirm nothing is missing or mismatched.
in_container "$SIF" "$VENV/bin/pip" check

echo "===== CHECK ====="
# No GPU on this node: JAX warns and falls back to CPU - expected. The GPU
# itself is checked at the start of train_gpu.slurm.
in_container "$SIF" "$VENV/bin/python" - <<'EOF'
import jax, jax.numpy as jp, brax, flax, mujoco
from mujoco import mjx
print("jax", jax.__version__, "| brax", brax.__version__, "| flax", flax.__version__,
      "| mujoco", mujoco.__version__)
m = mujoco.MjModel.from_xml_string(
    "<mujoco><worldbody><geom type='plane' size='1 1 .1'/>"
    "<body pos='0 0 .2'><freejoint/><geom size='.05'/></body></worldbody></mujoco>")
mx = mjx.put_model(m)
jax.jit(jax.vmap(lambda _: mjx.step(mx, mjx.make_data(m))))(jp.arange(256))
print("mjx ok")
EOF

echo "===== DONE - now: sbatch bdx_mjx/slurm/train_gpu.slurm ====="
