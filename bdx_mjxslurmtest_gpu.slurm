#!/bin/bash
#SBATCH --job-name=bdx_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/bdx_test_%j.out
#SBATCH --error=logs/bdx_test_%j.err

# ~10 min end-to-end test on the V100 before committing to a long run:
#   1. JAX sees the GPU inside the container
#   2. MJX steps 8192 envs of the real robot on the GPU (and how fast)
#   3. train.py runs all 3 curriculum phases + checkpoints at a tiny size
#   4. play.py --push-test runs on the result
# Every step prints PASS or stops the job with the error.
#
#   sbatch bdx_mjx/slurm/test_gpu.slurm
#   tail -f logs/bdx_test_<jobid>.out

set -e
cd /scratch/scratch26/23me36008/bot_learning
source bdx_mjx/slurm/env.sh
[ -f "$VENV/pyvenv.cfg" ] || { echo "no venv at $VENV - run install_env.slurm first" >&2; exit 1; }
PY="in_container --nv $SIF $VENV/bin/python"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

echo "===== JOB ====="
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "===== 1. JAX on the GPU ====="
$PY - <<'EOF'
import jax, jax.numpy as jp
print("devices:", jax.devices())
assert jax.default_backend() == "gpu", "JAX is not on the GPU"
x = jp.ones((4096, 4096)); print("matmul:", float((x @ x).sum()))
print("PASS 1")
EOF

echo "===== 2. MJX speed, real robot, 8192 envs ====="
$PY - <<'EOF'
import time, jax, jax.numpy as jp, mujoco
from mujoco import mjx
from bdx_mjx import constants as consts
m = mujoco.MjModel.from_xml_path(consts.TRAIN_XML)
mx = mjx.put_model(m, impl="jax")
d0 = mjx.make_data(m, impl="jax").replace(qpos=jp.array(m.keyframe("home").qpos))
n = 8192
batch = jax.vmap(lambda _: d0)(jp.arange(n))
step = jax.jit(jax.vmap(lambda d: mjx.step(mx, d)))
t = time.time(); batch = step(batch); jax.block_until_ready(batch); print(f"compile {time.time()-t:.0f} s")
t = time.time()
for _ in range(50): batch = step(batch)
jax.block_until_ready(batch); dt = time.time() - t
phys = n * 50 / dt
print(f"{phys:,.0f} physics steps/s  ->  ~{phys / 5:,.0f} env steps/s (5 substeps per action)")
print(f"=> 200M env steps would take ~{200e6 / (phys / 5) / 60:.0f} min of physics")
print("PASS 2")
EOF

echo "===== 3. training pipeline (tiny) ====="
rm -rf runs/gpu_test
$PY -u -m bdx_mjx.train --preset cpu_test --num-envs 64 --num-timesteps 200000 \
    --run-name gpu_test --runs-dir runs
ls runs/gpu_test/checkpoints | grep -q phase3 && echo "PASS 3"

echo "===== 4. push test on the result ====="
$PY -m bdx_mjx.play runs/gpu_test --push-test
echo "PASS 4"

echo "===== ALL PASSED - submit bdx_mjx/slurm/train_gpu.slurm ====="

