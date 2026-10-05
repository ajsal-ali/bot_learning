#!/bin/bash
# STEP 1 of 2 - run on the LOGIN node (compute nodes have no internet):
#   bash bdx_mjx/slurm/fetch_env.sh
# STEP 2 - build the env in a job (no internet needed):
#   sbatch bdx_mjx/slurm/install_env.slurm
#
# This step is light: micromamba resolves the package list in seconds (the
# >30 min that got flagged was conda's classic solver), then it only downloads.
# It writes bdx_mjx/slurm/env_explicit.txt (exact package URLs) and fills the
# package cache on scratch; the job installs from those offline.

set -euo pipefail

ROOT=/scratch/scratch26/23me36008/micromamba
export MAMBA_ROOT_PREFIX=$ROOT
PKGS=$ROOT/pkgs
HERE="$(cd "$(dirname "$0")" && pwd)"
MM="$HOME/.local/bin/micromamba"
mkdir -p "$PKGS"

# The login node has no GPU driver; without this the cuda12 jaxlib is rejected.
export CONDA_OVERRIDE_CUDA=12.0

if [ ! -x "$MM" ]; then
  echo "==> downloading micromamba to $MM"
  mkdir -p "$HOME/.local"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xj -C "$HOME/.local" bin/micromamba
fi

# jax < 0.10: brax calls jax.device_put_replicated, removed in jax 0.10.
# *cuda12* jaxlib: CUDA 13 builds do not support the V100.
SPECS=(
  "python=3.11"
  "jax<0.10" "jaxlib<0.10=*cuda12*" "cuda-version=12.*"
  "flax=0.12.6" optax "brax=0.12.5"
  "mujoco=3.14.0" "mujoco-mjx=3.14.0"
  orbax-checkpoint ml-collections
  matplotlib pillow ffmpeg mediapy
)

echo "==> resolving (seconds)"
nice -n 10 "$MM" create -n bdx_resolve --dry-run --json \
    --override-channels -c conda-forge "${SPECS[@]}" > "$HERE/env_resolve.json"

python3 - "$HERE/env_resolve.json" "$HERE/env_explicit.txt" <<'EOF'
import json, sys
actions = json.load(open(sys.argv[1]))["actions"]
pkgs = actions.get("LINK", [])
assert pkgs, "solver returned no packages - see env_resolve.json"
with open(sys.argv[2], "w") as f:
    f.write("@EXPLICIT\n")
    for p in pkgs:
        f.write(f'{p["url"]}#{p["md5"]}\n' if p.get("md5") else f'{p["url"]}\n')
names = {p["name"]: p for p in pkgs}
jl = names.get("jaxlib", {})
print(f"{len(pkgs)} packages; jaxlib {jl.get('version')} {jl.get('build_string')}")
assert "cuda12" in jl.get("build_string", ""), "jaxlib is not a cuda12 build"
EOF

echo "==> downloading packages to $PKGS"
grep -v '^@' "$HERE/env_explicit.txt" | cut -d'#' -f1 | while read -r url; do
  f="$PKGS/$(basename "$url")"
  [ -s "$f" ] || curl -sSfL --retry 3 -o "$f" "$url"
done
echo "==> $(ls "$PKGS" | wc -l) files in cache, $(du -sh "$PKGS" | cut -f1)"
echo "==> now: sbatch bdx_mjx/slurm/install_env.slurm"
