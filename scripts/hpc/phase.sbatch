#!/bin/bash -l
#SBATCH --job-name=tc-phase
#SBATCH --nodes=1
#SBATCH --ntasks=1
# One task of one phase.
# submit.sh passes the phase, the roots, the shard count and every site value (account, partition, QOS, cores, memory and wall, all from site.env) through --export and the sbatch command line, so no cluster-specific value lives in this file.
# This file only turns the array index into the TC_* variables the phase script reads, then hands over to run_phase.sh.
#
#   phase1, phase2, phase4, phase5, phase6 : array 1..K over recordings
#   phase3 (work)                  : array 1..K*S over recording x shard
#   phase3 (finalize)              : array 1..K over recordings, TC_FINALIZE=1
#   phase7 (work)                  : array 1..S over shards
#   phase7 (finalize)              : one task, TC_FINALIZE=1
set -u
phase="${TC_PHASE:?phase1..phase7}"
shards="${TC_SHARDS:-1}"
task="${SLURM_ARRAY_TASK_ID:-1}"
unset TC_RECORDING_INDEX TC_SHARD # only this file decides them

if [ "${TC_FINALIZE:-0}" = 1 ]; then
    if [ "$phase" = phase3 ]; then
        export TC_RECORDING_INDEX="$task"
    fi
else
    case "$phase" in
        phase3)
            export TC_RECORDING_INDEX=$(((task - 1) / shards + 1))
            export TC_SHARD="$(((task - 1) % shards + 1))/$shards"
            ;;
        phase7)
            export TC_SHARD="$task/$shards"
            ;;
        *)
            export TC_RECORDING_INDEX="$task"
            ;;
    esac
fi

exec bash "${TC_CODE_ROOT:?}/scripts/hpc/run_phase.sh" "$phase"
