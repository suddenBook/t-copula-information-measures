#!/bin/bash
# submit.sh <phase1..phase7> [options]: run on the cluster login node.
#
# Queues one phase.
# There is no gate, smoke run or validation: sbatch's own exit status is the only thing that can refuse a submission, and every job resumes from the checkpoints on shared storage.
#
#   --chain N            submit the array N times, each waiting on the previous one with afterany, so a job the wall cuts is followed by one that resumes it (default 1)
#   --after <id[:id...]> extra afterany dependency, for chaining phases
#   --recordings K       array width over recordings (default 10)
#   --shards S           shards of phase 3 (default 2, one per surrogate realization) and of phase 7 (default 9); each memory class of phase 7 is sharded on its own, and the phase gives a shard its recovery jobs heaviest first, so any S balances the observations a shard fits
#   --class C            phase 7 only: light or heavy, the memory class this submission computes (TC_SIM_CLASS); without it the job gets TC_SIM_CLASS empty, meaning every class, so a value left in this shell can never leak into a job.
#                        A class run never publishes, so --class implies --finalize no; queue the one finalize job after both classes with --finalize only --after <heavy>:<light>.
#   --workers W          TC_WORKERS inside the job (default TC_HPC_WORKERS)
#   --mem G              memory per task (default TC_HPC_MEM)
#   --time D-HH:MM:SS    wall limit per task (default TC_HPC_TIME)
#   --tasks LIST         the array tasks to submit, in sbatch --array syntax (1,3,5,7, 9-20, or 1-8%4 to run at most 4 at once); default every task of the shape below.
#                        It lets one phase go to two partitions in two calls, for example the large recordings to large-memory nodes and the small ones to other nodes.
#   --finalize MODE      phases 3 and 7: yes (default) queues the finalize job after the arrays; no queues the arrays only; only queues the finalize job alone (give it --after the arrays of the earlier calls).
#                        Phase 3's finalize computes nothing and needs small resources; phase 7's merges the whole study (millions of replicate rows) and renders it, so give it a whole node's memory.
#   --dry-run            print the sbatch command lines and submit nothing
#
# Shapes:
#   phase1 phase2 phase4 phase5 phase6   array 1-K, one recording per task
#   phase3                               array 1-(K*S) over recording x shard, then a finalize array 1-K after it
#   phase7                               array 1-S over shards, then one finalize job
#
# The account, the partition, the QOS, the MATLAB module and the resource defaults come from site.env; TC_BASE (default TC_HPC_BASE) sets the remote layout; TC_OVERWRITE and TC_RENDER_FIGURES are passed through from this shell if you export them.
#
# TC_OVERWRITE reaches the first array of a --chain and nothing else: every successor array and the finalize job are exported TC_OVERWRITE=0.
# A successor exists to resume what the wall cut, so one that recomputed from the beginning would repeat the first allocation of work for ever and never reach the rest; the finalize job only merges checkpoints, so it has nothing to recompute.
# The job ids are printed and appended to $TC_BASE/logs/<phase>-<stamp>.jobs.
set -u

# The cluster's own values (account, partition, QOS, MATLAB module, base directory, resource defaults) live in site.env next to this script, a copy of site.env.example.
# A TC_HPC_* already in the environment, and any option on the command line, win over the file.
site_file="$(dirname "$0")/site.env"
# shellcheck source=/dev/null
[ -f "$site_file" ] && . "$site_file"

phase="${1:?usage: submit.sh <phase1..phase7> [--chain N] [--after ids] [--recordings K] [--shards S] [--class light|heavy] [--workers W] [--mem G] [--time D-HH:MM:SS] [--tasks LIST] [--finalize yes|no|only] [--dry-run]}"
shift
case "$phase" in
    phase1 | phase2 | phase3 | phase4 | phase5 | phase6 | phase7) ;;
    *)
        echo "submit.sh: unknown phase '$phase' (expected phase1..phase7)" >&2
        exit 2
        ;;
esac

chain=1
after=""
recordings=10
shards=""
workers="${TC_HPC_WORKERS:-63}"
mem="${TC_HPC_MEM:-240G}"
wall="${TC_HPC_TIME:-7-00:00:00}"
tasks=""
finalize=yes
finalize_given=0
class=""
dry=0
while [ $# -gt 0 ]; do
    case "$1" in
        --chain)
            chain="${2:?}"
            shift 2
            ;;
        --after)
            after="${2:?}"
            shift 2
            ;;
        --recordings)
            recordings="${2:?}"
            shift 2
            ;;
        --shards)
            shards="${2:?}"
            shift 2
            ;;
        --workers)
            workers="${2:?}"
            shift 2
            ;;
        --mem)
            mem="${2:?}"
            shift 2
            ;;
        --time)
            wall="${2:?}"
            shift 2
            ;;
        --tasks)
            tasks="${2:?}"
            shift 2
            ;;
        --finalize)
            finalize="${2:?}"
            finalize_given=1
            shift 2
            ;;
        --class)
            class="${2:?}"
            shift 2
            ;;
        --dry-run)
            dry=1
            shift
            ;;
        *)
            echo "submit.sh: unknown option '$1'" >&2
            exit 2
            ;;
    esac
done
case "$finalize" in
    yes | no | only) ;;
    *)
        echo "submit.sh: --finalize takes yes, no or only; got '$finalize'" >&2
        exit 2
        ;;
esac
case "$phase" in
    phase3 | phase7) ;;
    *) [ "$finalize" = only ] && {
        echo "submit.sh: $phase has no finalize job" >&2
        exit 2
    } ;;
esac
case "$class" in
    '' | light | heavy) ;;
    *)
        echo "submit.sh: --class takes light or heavy; got '$class'" >&2
        exit 2
        ;;
esac
[ -n "$class" ] && [ "$phase" != phase7 ] && {
    echo "submit.sh: --class applies to phase7 only" >&2
    exit 2
}
if [ -n "$class" ]; then
    [ "$finalize_given" = 1 ] && [ "$finalize" != no ] && {
        echo "submit.sh: a --class run never publishes; queue the finalize job once, after both classes, with --finalize only (no --class)" >&2
        exit 2
    }
    finalize=no
fi
if [ -n "$tasks" ]; then
    case "$tasks" in
        *[!0-9,%-]*)
            echo "submit.sh: --tasks takes an sbatch array list such as 1,3,5, 9-20 or 1-8%4; got '$tasks'" >&2
            exit 2
            ;;
    esac
fi
case "$phase" in
    phase3) shards="${shards:-2}" ;;
    phase7) shards="${shards:-9}" ;;
    *) shards=1 ;;
esac

missing=""
[ -n "${TC_HPC_ACCOUNT:-}" ] || missing="$missing TC_HPC_ACCOUNT"
[ -n "${TC_HPC_PARTITION:-}" ] || missing="$missing TC_HPC_PARTITION"
[ -n "${TC_HPC_QOS:-}" ] || missing="$missing TC_HPC_QOS"
[ -n "${TC_HPC_MATLAB_MODULE:-}" ] || missing="$missing TC_HPC_MATLAB_MODULE"
[ -z "$missing" ] || {
    echo "submit.sh: no value for$missing; set them, or copy scripts/hpc/site.env.example to scripts/hpc/site.env" >&2
    exit 2
}

base="${TC_BASE:-${TC_HPC_BASE:-$HOME/tcopula}}"
case "$base" in \~/*) base="$HOME/${base#\~/}" ;; esac # site.env may write ~/
code="${TC_CODE_ROOT:-$base/code}"
logs="$base/logs"
sbatch_file="$code/scripts/hpc/phase.sbatch"
mkdir -p "$logs"
jobs_file="$logs/$phase-$(date -u +%Y%m%dT%H%M%SZ).jobs"

case "$phase" in
    phase3) array="1-$((recordings * shards))" ;;
    phase7) array="1-$shards" ;;
    *) array="1-$recordings" ;;
esac
# A task list selects tasks of the same shape: phase.sbatch still maps task t to its recording and shard through TC_SHARDS, so a subset runs exactly the work those tasks would run in the full array.
[ -n "$tasks" ] && array="$tasks"
common="ALL,TC_PHASE=$phase,TC_BASE=$base,TC_CODE_ROOT=$code"
common="$common,TC_DATA_ROOT=${TC_DATA_ROOT:-$base/data}"
common="$common,TC_RESULTS_ROOT=${TC_RESULTS_ROOT:-$base/results}"
common="$common,TC_WORKERS=$workers,TC_SHARDS=$shards"
common="$common,TC_RENDER_FIGURES=${TC_RENDER_FIGURES:-1}"
common="$common,TC_HPC_MATLAB_MODULE=$TC_HPC_MATLAB_MODULE" # run_phase.sh loads it
# Phase 7's memory class travels explicitly, empty meaning every class, so a TC_SIM_CLASS left in this shell cannot reach a job by --export=ALL.
job_name="tc-$phase"
if [ "$phase" = phase7 ]; then
    common="$common,TC_SIM_CLASS=$class"
    [ -n "$class" ] && job_name="tc-$phase-$class"
fi
# The caller's recomputation request is the first array's alone; see the header.
overwrite_first="TC_OVERWRITE=${TC_OVERWRITE:-0}"
overwrite_resume="TC_OVERWRITE=0"
# The optional module lists travel the same way; a space-separated list is quoted for sbatch's comma-separated --export syntax.
[ -n "${TC_HPC_PYTHON_MODULES:-}" ] && common="$common,TC_HPC_PYTHON_MODULES=\"$TC_HPC_PYTHON_MODULES\""
[ -n "${TC_HPC_MODULE_PREREQS:-}" ] && common="$common,TC_HPC_MODULE_PREREQS=\"$TC_HPC_MODULE_PREREQS\""
resources=(--nodes=1 --ntasks=1 --cpus-per-task="${TC_HPC_CPUS:-64}" --mem="$mem" --time="$wall"
    --account="$TC_HPC_ACCOUNT" --partition="$TC_HPC_PARTITION" --qos="$TC_HPC_QOS")
# Free-form extra sbatch options from site.env (for example --hint=nomultithread to pin one worker per physical core), word-split on purpose.
# shellcheck disable=SC2206
[ -n "${TC_HPC_SBATCH_EXTRA:-}" ] && resources+=(${TC_HPC_SBATCH_EXTRA})

job_id=""
final_id=""
n_dry=0
# submit <label> <sbatch args...>: print the command line and, unless this is a dry run, submit it and record its id in the jobs file; job_id is set either way.
submit() {
    local label="$1"
    shift
    printf '%s\n  sbatch %s %s\n' "$label" "$*" "$sbatch_file"
    if [ "$dry" = 1 ]; then
        n_dry=$((n_dry + 1))
        job_id="DRY$n_dry"
        return 0
    fi
    job_id=$(sbatch --parsable "$@" "$sbatch_file") \
        || {
            echo "sbatch refused $label" >&2
            exit 1
        }
    job_id="${job_id%%;*}"
    echo "$label $job_id" >>"$jobs_file"
    echo "  -> $job_id"
}

echo "$phase array=$array chain=$chain shards=$shards workers=$workers finalize=$finalize${class:+ class $class}${after:+ after $after}"
# Every other TC_SIM_* in this shell reaches the jobs through --export=ALL and changes the study they run, so they are printed here, where a pilot's leftover setting is seen.
sim_env=$(env | grep '^TC_SIM_' | grep -v '^TC_SIM_CLASS=' | sort | tr '\n' ' ')
[ -n "$sim_env" ] && echo "  exported to the jobs from this shell: $sim_env"
# A dry run submits nothing, so it leaves no record in the jobs file either.
if [ "$dry" != 1 ]; then
    echo "$phase submitted $(date -u +%FT%TZ) array=$array chain=$chain shards=$shards finalize=$finalize${class:+ class=$class}${sim_env:+ env: $sim_env}${after:+ after $after}" \
        >>"$jobs_file"
fi
previous="$after"
i=1
# --finalize only queues no array: the finalize job waits on --after alone.
[ "$finalize" = only ] && i=$((chain + 1))
while [ "$i" -le "$chain" ]; do
    if [ "$i" = 1 ]; then overwrite="$overwrite_first"; else overwrite="$overwrite_resume"; fi
    args=(--job-name="$job_name" --output="$logs/%x-%A_%a.out" --array="$array"
        --export="$common,$overwrite,TC_FINALIZE=0" "${resources[@]}")
    [ -n "$previous" ] && args+=(--dependency="afterany:$previous")
    submit "$phase array $i/$chain" "${args[@]}"
    previous="$job_id"
    i=$((i + 1))
done
final_after=()
[ -n "$previous" ] && final_after=(--dependency="afterany:$previous")
[ "$finalize" = no ] || case "$phase" in
    phase3)
        submit "$phase finalize" \
            --job-name="tc-$phase-final" --output="$logs/%x-%A_%a.out" --array="1-$recordings" \
            ${final_after[@]+"${final_after[@]}"} \
            --export="$common,$overwrite_resume,TC_FINALIZE=1" "${resources[@]}"
        final_id="$job_id"
        ;;
    phase7)
        submit "$phase finalize" \
            --job-name="tc-$phase-final" --output="$logs/%x-%j.out" \
            ${final_after[@]+"${final_after[@]}"} \
            --export="$common,$overwrite_resume,TC_FINALIZE=1" "${resources[@]}"
        final_id="$job_id"
        ;;
esac

if [ "$dry" = 1 ]; then
    echo "dry run: nothing submitted"
elif [ "$finalize" = only ]; then
    echo "finalize job id $final_id${after:+, after $after}; ids in $jobs_file"
else
    echo "last array job id $previous${final_id:+, finalize job id $final_id}; ids in $jobs_file"
fi
