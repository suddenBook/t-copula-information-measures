#!/bin/bash
# run_phase.sh <phase1|phase2|phase3|phase4|phase5|phase6|phase7>
#
# Runs one phase inside a Slurm allocation.
# phase.sbatch calls it, and it also runs by hand on an interactive node.
# Nothing here checks a result: the exit status is MATLAB's own, and a phase that has nothing left to do exits in seconds because every phase resumes from its checkpoints.
#
# Environment (phase.sbatch sets the per-task part, and everything else has a default):
#   TC_BASE             remote base directory              TC_HPC_BASE, ~/tcopula
#   TC_CODE_ROOT        the rsynced source tree            $TC_BASE/code
#   TC_DATA_ROOT        raw inputs                         $TC_BASE/data
#   TC_RESULTS_ROOT     output tree, shared storage        $TC_BASE/results
#   TC_WORKERS          parallel workers                   $SLURM_CPUS_PER_TASK - 1
#   TC_RECORDING_INDEX, TC_SHARD, TC_FINALIZE, TC_OVERWRITE, TC_RENDER_FIGURES   passed straight through to the phase script
#   TC_HPC_MATLAB_MODULE  the module to load; submit.sh exports it into the job, and site.env below is the fallback for a hand run
set -u

# The cluster's own values live in site.env next to this script, a copy of site.env.example; a TC_HPC_* already in the environment wins over the file.
site_file="$(dirname "$0")/site.env"
# shellcheck source=/dev/null
[ -f "$site_file" ] && . "$site_file"

phase="${1:?phase1..phase7}"
case "$phase" in
    phase1) script=Phase1_preprocessing ;;
    phase2) script=Phase2_t_copula_fit ;;
    phase3) script=Phase3_gaussian_surrogates ;;
    phase4) script=Phase4_information_measures ;;
    phase5) script=Phase5_pair_density ;;
    phase6) script=Phase6_autocorrelation_timescale ;;
    phase7) script=Phase7_simulation_study ;;
    *)
        echo "run_phase.sh: unknown phase '$phase' (expected phase1..phase7)" >&2
        exit 2
        ;;
esac

base="${TC_BASE:-${TC_HPC_BASE:-$HOME/tcopula}}"
case "$base" in \~/*) base="$HOME/${base#\~/}" ;; esac # site.env may write ~/
code="${TC_CODE_ROOT:-$base/code}"
export TC_DATA_ROOT="${TC_DATA_ROOT:-$base/data}"
export TC_RESULTS_ROOT="${TC_RESULTS_ROOT:-$base/results}"
export TC_WORKERS="${TC_WORKERS:-$((${SLURM_CPUS_PER_TASK:-2} - 1))}"

# MATLAB comes from the module stack of site.env: TC_HPC_MODULE_PREREQS (loaded before the MATLAB module) and TC_HPC_MATLAB_MODULE.
# Loads are best effort, because a module rename is not a scientific fact and an absent MATLAB ends the job on its own below.
# Python is uv's own interpreter (see below).
if type module >/dev/null 2>&1; then
    [ -n "${TC_HPC_MATLAB_MODULE:-}" ] || {
        echo "run_phase.sh: no MATLAB module name; set TC_HPC_MATLAB_MODULE, or copy scripts/hpc/site.env.example to scripts/hpc/site.env" >&2
        exit 2
    }
    module purge 2>/dev/null
    for m in ${TC_HPC_MODULE_PREREQS:-}; do module load "$m" 2>/dev/null; done
    module load "$TC_HPC_MATLAB_MODULE"
else
    echo "[run_phase] no 'module' command in this shell; using whatever matlab is on PATH" >&2
fi
export PATH="$HOME/.local/bin:$PATH" # uv lives here
# The figure renderers run under uv with the uv-managed interpreter that prewarm_figures.sh installed (TC_HPC_UV_PYTHON, default 3.12) and the script environments it built.
# Jobs never resolve or build an environment, so concurrent tasks cannot race on the shared cache and corrupt it.
# A renderer whose environment is missing fails on its own, never the phase.
export UV_PYTHON="${TC_HPC_UV_PYTHON:-3.12}" UV_PYTHON_DOWNLOADS=never UV_NO_SYNC=1 UV_OFFLINE=1
unset MATLABPATH PYTHONPATH PYTHONHOME
# Workers that each run a multithreaded BLAS oversubscribe the task's cores (31 workers on a 32-core task, for example), so the numerical libraries run single threaded, as MATLAB's workers do either way.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Everything per job goes on node-local /scratch (or under TMPDIR, default /tmp, when /scratch cannot be created) and is removed on exit: MATLAB's preferences, the MCR cache, matplotlib's cache and the parallel pool's JobStorageLocation.
# A fresh MATLAB_PREFDIR also means a fresh `local` cluster profile per job, so two array tasks sharing a node cannot collide over one pool directory.
# Checkpoints never live here: they are written to TC_RESULTS_ROOT on shared storage, because the later jobs of the chain and the finalizer must see them.
scratch="/scratch/${USER:-$(id -un)}/${SLURM_JOB_ID:-local}.${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$scratch" 2>/dev/null || scratch="${TMPDIR:-/tmp}/tc-${SLURM_JOB_ID:-local}.${SLURM_ARRAY_TASK_ID:-0}"
export MATLAB_PREFDIR="$scratch/matlab-prefs"
export MCR_CACHE_ROOT="$scratch/mcr"
export MPLCONFIGDIR="$scratch/matplotlib"
export TC_JOB_STORAGE="$scratch/pool"
mkdir -p "$MATLAB_PREFDIR" "$MCR_CACHE_ROOT" "$MPLCONFIGDIR" "$TC_JOB_STORAGE"
trap 'rm -rf "$scratch"' EXIT

cd "$code" || exit 1
printf '[run_phase] %s script=%s host=%s job=%s task=%s recording=%s shard=%s finalize=%s workers=%s scratch=%s %s\n' \
    "$phase" "$script" "$(hostname -s)" "${SLURM_JOB_ID:-local}" \
    "${SLURM_ARRAY_TASK_ID:-0}" "${TC_RECORDING_INDEX:-all}" "${TC_SHARD:-1/1}" \
    "${TC_FINALIZE:-0}" "$TC_WORKERS" "$scratch" "$(date -u +%FT%TZ)"

# SHELL and MATLAB_JAVA can only be repaired at exec time, so MATLAB starts with SHELL and MATLAB_SHELL set to /bin/bash and MATLAB_JAVA unset.
# A launcher that exports SHELL as a bare basename makes MATLAB exec it without a PATH search, so every system() call returns 127 and the MATLAB-to-Python figure bridge silently dies.
# An inherited MATLAB_JAVA commonly names a JVM this environment does not have.
# The prelude pins the pool's job storage to the node-local directory above, raises the profile's NumWorkers to TC_WORKERS when the requested count exceeds the physical-core default (hyperthreads count as CPUs here), and saves it into this job's own profile.
prelude="try, c = parcluster('local'); c.JobStorageLocation = getenv('TC_JOB_STORAGE'); w = str2double(getenv('TC_WORKERS')); if isfinite(w) && w > c.NumWorkers, c.NumWorkers = w; end; saveProfile(c); catch pool_err, fprintf(2, '[run_phase] pool profile left at its default: %s\n', pool_err.message); end; "
env -u MATLAB_JAVA SHELL=/bin/bash MATLAB_SHELL=/bin/bash \
    matlab -batch "addpath('scripts'); ${prelude}${script}"
status=$?
printf '[run_phase] %s exit %d %s\n' "$phase" "$status" "$(date -u +%FT%TZ)"
exit "$status"
