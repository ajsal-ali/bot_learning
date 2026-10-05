# Shared settings for the ParamShakti scripts. Sourced, not run.
#
# Why a container: ParamShakti is CentOS 7 (glibc 2.17) and every modern
# jaxlib needs glibc >= 2.28, so JAX cannot run on the host - in any conda
# env. A small Debian container (python:3.11-slim, ~50 MB) supplies a new
# glibc; `--nv` passes the host's NVIDIA driver in, so the V100 works inside.
# The Python packages live in a venv on scratch, built with the container's
# Python, so the image itself never needs rebuilding.

PROJECT=/scratch/scratch26/23me36008/bot_learning
BDX_ENV=/scratch/scratch26/23me36008/bdx_env
SIF=$BDX_ENV/py311.sif          # the container image
WHEELS=$BDX_ENV/wheels          # downloaded packages (login node)
VENV=$BDX_ENV/venv              # installed packages (install job)

# Keep apptainer's cache/temp off the home quota.
export APPTAINER_CACHEDIR=$BDX_ENV/apptainer_cache
export APPTAINER_TMPDIR=$BDX_ENV/apptainer_tmp
mkdir -p "$BDX_ENV" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# Batch shells may not have `module` defined yet.
type module >/dev/null 2>&1 || source ~/.bashrc
type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh
module load apps/apptainer/1.4.0

# Run a command inside the container. Scratch is bound so the project,
# wheels and venv are visible; home is bound by default.
in_container() {
  apptainer exec --bind /scratch/scratch26/23me36008 "$@"
}
