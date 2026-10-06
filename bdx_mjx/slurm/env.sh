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
# So `import bdx_mjx` works inside the container from any working directory.
export PYTHONPATH=$PROJECT${PYTHONPATH:+:$PYTHONPATH}

# Keep apptainer's cache/temp off the home quota.
export APPTAINER_CACHEDIR=$BDX_ENV/apptainer_cache
export APPTAINER_TMPDIR=$BDX_ENV/apptainer_tmp
mkdir -p "$BDX_ENV" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# Batch shells may not have `module` defined yet.
type module >/dev/null 2>&1 || source ~/.bashrc
type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh
# Use the first container module that actually runs (apptainer 1.4.0 on
# ParamShakti fails with a missing libsubid.so.3).
CT=""
for mod in apps/apptainer/1.2.5:apptainer apps/singularity/3.4.1:singularity \
           apps/apptainer/1.4.0:apptainer; do
  module purge >/dev/null 2>&1
  module load "${mod%%:*}" >/dev/null 2>&1 || continue
  if "${mod##*:}" --version >/dev/null 2>&1; then CT="${mod##*:}"; break; fi
done
[ -n "$CT" ] || { echo "!! no working apptainer/singularity module" >&2; return 1 2>/dev/null || exit 1; }
echo "container runtime: $($CT --version)"
export SINGULARITY_CACHEDIR=$APPTAINER_CACHEDIR SINGULARITY_TMPDIR=$APPTAINER_TMPDIR

# Run a command inside the container. Scratch is bound so the project,
# wheels and venv are visible; home is bound by default.
in_container() {
  "$CT" exec --bind /scratch/scratch26/23me36008 "$@"
}
