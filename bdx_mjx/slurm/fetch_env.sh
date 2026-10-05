#!/bin/bash
# STEP 1 of 2 - run on the LOGIN node (compute nodes have no internet):
#   bash bdx_mjx/slurm/fetch_env.sh
# STEP 2 - install in a job (no internet needed):
#   sbatch bdx_mjx/slurm/install_env.slurm
#
# Downloads only: the container image (~50 MB) and the ~95 Python wheels
# (~3-4 GB, mostly NVIDIA CUDA libraries) listed with exact versions in
# bdx_mjx/requirements-lock-linux.txt. --no-deps means pip does no dependency
# resolving at all (that resolve is what ran for an hour before), so the only
# work is the download itself. Re-running skips files already downloaded.

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
# One package at a time so every step shows in the terminal:
#   [12/95] nvidia-cudnn-cu12==9.27.0.42   + pip's progress bar
mapfile -t PKGS < <(grep -vE '^\s*(#|$)' "$PROJECT/bdx_mjx/requirements-lock-linux.txt")
N=${#PKGS[@]}
START=$SECONDS
for i in "${!PKGS[@]}"; do
  echo ""
  echo "[$((i + 1))/$N] ${PKGS[$i]}   (total so far: $(du -sh "$WHEELS" | cut -f1), $((SECONDS - START)) s)"
  in_container "$SIF" python -m pip download --no-deps --only-binary=:all: \
      --progress-bar on --disable-pip-version-check --dest "$WHEELS" "${PKGS[$i]}"
done
echo ""
echo "==> $(ls "$WHEELS" | wc -l) wheels, $(du -sh "$WHEELS" | cut -f1), $((SECONDS - START)) s"
echo "==> now: sbatch bdx_mjx/slurm/install_env.slurm"
