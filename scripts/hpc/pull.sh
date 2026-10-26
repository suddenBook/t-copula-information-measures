#!/bin/bash
# pull.sh [host] [all|phase1..phase7]: run on the local machine.
#
# Copies the cluster's output tree into the local one (TC_RESULTS_ROOT, default results/) and files each Slurm log beside the outputs it records, in <recording>/PhaseN/text/cluster_logs/.
# --checksum verifies every byte, --partial resumes an interrupted transfer, and there is no --delete anywhere, so this never removes a local file.
# It is safe to run while jobs are still computing, and a later run copies only what changed.
#
#   host   ssh host (default TC_HPC_HOST from site.env); a path containing "/" is a local directory, which is how a rehearsal is done
#   phase  restrict the transfer to one phase's subtrees (plus recording.json) and to that phase's logs
set -u

# The cluster's own values live in site.env next to this script, a copy of site.env.example; this script reads the ssh alias and the base directory from it.
# A TC_HPC_* already in the environment, and the host argument, win over the file.
site_file="$(dirname "$0")/site.env"
# shellcheck source=/dev/null
[ -f "$site_file" ] && . "$site_file"

host="${1:-${TC_HPC_HOST:-}}"
[ -n "$host" ] || {
    echo "pull.sh: no cluster host; set TC_HPC_HOST or copy scripts/hpc/site.env.example to scripts/hpc/site.env" >&2
    exit 2
}
what="${2:-all}"
repo="$(cd "$(dirname "$0")/../.." && pwd)"
dest="${TC_RESULTS_ROOT:-$repo/results}"
remote="${TC_HPC_BASE:-~/tcopula}"
case "$host" in
    */*) base="$host" ;;
    # A leading ~/ would reach the remote shell unexpanded; without it the path is relative to the remote home, which is what it means.
    *) base="$host:${remote#\~/}" ;;
esac
mkdir -p "$dest"

args=(-a --checksum --partial --info=progress2)
case "$what" in
    all) ;;
    phase[1-7])
        args+=(--prune-empty-dirs --include='*/' --include="Phase${what#phase}/***"
            --include='recording.json' --exclude='*')
        ;;
    *)
        echo "pull.sh: second argument is 'all' or phase1..phase7, got '$what'" >&2
        exit 2
        ;;
esac
rsync "${args[@]}" "$base/results/" "$dest/" || exit 1

# Every log opens with "[run_phase] phaseN ..." (run_phase.sh), and the phase script then names the recording's folder on a line of its own ("[PhaseN] <folder>", printed by openPhaseOutput), so a log is filed under the first folder it names whose PhaseN output exists locally.
# A log that names none (a resume job that found its work already done, or a job that failed before writing anything) stays on the cluster and is only counted.
stage="$(mktemp -d)" || exit 1
trap 'rm -rf "$stage"' EXIT
rsync -a --include='*.out' --exclude='*' "$base/logs/" "$stage/" || exit 1
filed=0
left=0
for log in "$stage"/*.out; do
    [ -e "$log" ] || continue
    phase="$(grep -m1 -aoE '^\[run_phase\] phase[1-7]' "$log")"
    phase="${phase##*phase}"
    case "$what" in phase[1-7]) [ "$phase" = "${what#phase}" ] || continue ;; esac
    folder=""
    if [ -n "$phase" ]; then
        while IFS= read -r line; do
            candidate="${line#"[Phase$phase] "}"
            case "$candidate" in "" | /* | *..*) continue ;; esac
            if [ -d "$dest/$candidate/Phase$phase" ]; then
                folder="$candidate"
                break
            fi
        done < <(grep -aE "^\[Phase$phase\] " "$log")
    fi
    if [ -z "$folder" ]; then
        left=$((left + 1))
        continue
    fi
    out="$dest/$folder/Phase$phase/text/cluster_logs"
    mkdir -p "$out" || exit 1
    cmp -s "$log" "$out/${log##*/}" || cp -p "$log" "$out/" || exit 1
    filed=$((filed + 1))
done
echo "pulled $what from $base into $dest; Slurm logs: $filed filed in <folder>/PhaseN/text/cluster_logs, $left naming no local output folder left on the cluster"
