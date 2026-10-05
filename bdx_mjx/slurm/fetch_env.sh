#!/bin/bash
# STEP 1 of 2 - run on the LOGIN node (compute nodes have no internet):
#   bash bdx_mjx/slurm/fetch_env.sh
# STEP 2 - install in a job (no internet needed):
#   sbatch bdx_mjx/slurm/install_env.slurm
#
# Downloads only: the container image (~50 MB) and the Python wheels (~3-4 GB,
# mostly NVIDIA CUDA libraries). The only CPU work is squashing the small image
# and turning the one pure-Python source package (gym, needed by brax) into a
# wheel - seconds, far from the login node's 30-minute limit.

set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

if [ ! -f "$SIF" ]; then
  echo "==> pulling container image -> $SIF"
  "$CT" pull "$SIF" docker://python:3.11-slim-bookworm
fi
in_container "$SIF" python --version

echo "==> downloading wheels -> $WHEELS"
mkdir -p "$WHEELS"
# `pip wheel` = download, plus build wheels for any source-only packages, so
# the offline install in the job never needs to build anything.
in_container "$SIF" nice -n 10 python -m pip wheel --quiet --wheel-dir "$WHEELS" \
    -r "$PROJECT/bdx_mjx/requirements.txt" "jax[cuda12]==0.9.2"
echo "==> $(ls "$WHEELS" | wc -l) wheels, $(du -sh "$WHEELS" | cut -f1)"
echo "==> now: sbatch bdx_mjx/slurm/install_env.slurm"
