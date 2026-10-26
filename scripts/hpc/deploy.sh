#!/bin/bash
# deploy.sh [--data|--data-all] [--dry-run] [host]: run on the local machine.
#
# Sends the source tree (src/, scripts/, tests/, README.md, LICENSE, .gitignore) to <host>:<base>/code/, site.env included, which is how submit.sh reads the site values there.
# results/, data/ and .git are never part of the code transfer: they are simply not in the list.
# --delete applies to the code tree only, so a file deleted from the local tree disappears there too.
#
#   --data       also send the Phase 1 inputs of the ten recordings to <host>:<base>/data/, keeping their paths under data/ (about 2.35 GB); this is what the pipeline reads
#   --data-all   send the whole of data/ instead, raw archives and papers included (about 102 GB, which on a home quota of a few hundred GB leaves little room for the results); only for archive work
#   --dry-run    list what rsync would transfer and change nothing
#   host         ssh host (default TC_HPC_HOST from site.env); a path containing "/" is treated as a local directory, which is how a rehearsal is done
#
# The four directories of the remote layout (code, data, results, logs) are created before the first transfer.
# rsync creates the last component of a destination when its parent exists and fails with code 11 when it does not, so a first deployment into an empty home needs them made first.
set -u

# The cluster's own values live in site.env next to this script, a copy of site.env.example; this script reads the ssh alias and the base directory from it.
# A TC_HPC_* already in the environment, and the host argument, win over the file.
site_file="$(dirname "$0")/site.env"
# shellcheck source=/dev/null
[ -f "$site_file" ] && . "$site_file"

# The Phase 1 inputs of the ten recordings in scope, relative to the data root.
# Phase 1 reads only <family>/preprocessed/; the families' other folders, such as raw/, belong to data preparation and stay local.
# Update this list when the recording table at the top of Phase1_preprocessing.m gains a family.
inputs=(human_SEEG/preprocessed
    monkey_LFP/preprocessed
    monkey_ECoG/preprocessed)

data=none
dry=0
host=""
while [ $# -gt 0 ]; do
    case "$1" in
        --data)
            data=inputs
            shift
            ;;
        --data-all)
            data=all
            shift
            ;;
        --dry-run)
            dry=1
            shift
            ;;
        -*)
            echo "deploy.sh: unknown option '$1'" >&2
            exit 2
            ;;
        *)
            host="$1"
            shift
            ;;
    esac
done
host="${host:-${TC_HPC_HOST:-}}"
[ -n "$host" ] || {
    echo "deploy.sh: no cluster host; set TC_HPC_HOST or copy scripts/hpc/site.env.example to scripts/hpc/site.env" >&2
    exit 2
}
repo="$(cd "$(dirname "$0")/../.." && pwd)"
remote="${TC_HPC_BASE:-~/tcopula}"
case "$host" in
    */*)
        ssh_host=""
        base_path="$host"
        ;;
    # A leading ~/ would reach the remote shell unexpanded, so strip it: the path is then relative to the remote home, which is what it means.
    *)
        ssh_host="$host"
        base_path="${remote#\~/}"
        ;;
esac
base="${ssh_host:+$ssh_host:}$base_path"

# Make the layout before the first transfer; see the header.
mkdir_command="mkdir -p '$base_path/code' '$base_path/data' '$base_path/results' '$base_path/logs'"
if [ -n "$ssh_host" ]; then
    if [ "$dry" = 1 ]; then
        echo "would run: ssh $ssh_host \"$mkdir_command\""
    else
        # Expanding locally is the point: the base comes from the local site.env and travels as literal single-quoted words the remote shell does not re-expand.
        # shellcheck disable=SC2029
        ssh "$ssh_host" "$mkdir_command" || exit 1
    fi
elif [ "$dry" = 1 ]; then
    echo "would run: $mkdir_command"
else
    mkdir -p "$base_path/code" "$base_path/data" "$base_path/results" "$base_path/logs" \
        || exit 1
fi

code_args=(-a --delete --exclude=__pycache__ --exclude='*.asv' --exclude='*.DS_Store')
[ "$dry" = 1 ] && code_args+=(--dry-run --itemize-changes)
for item in src scripts tests README.md LICENSE .gitignore; do
    [ -e "$repo/$item" ] && code_args+=("$repo/$item")
done
echo "code -> $base/code/"
rsync "${code_args[@]}" "$base/code/" || exit 1

# The remote tree carries no .git, so the commit travels as a plain file that writeRunConfig.m reads into every run_config.json (provenance only, never checked).
sha="$(git -C "$repo" rev-parse HEAD 2>/dev/null || echo unknown)"
sha_tmp="$(mktemp)"
printf '%s\n' "$sha" >"$sha_tmp"
if [ "$dry" = 1 ]; then
    echo "GIT_SHA ($sha) -> $base/code/GIT_SHA"
else
    rsync -a "$sha_tmp" "$base/code/GIT_SHA" || {
        rm -f "$sha_tmp"
        exit 1
    }
fi
rm -f "$sha_tmp"

# Build the figure renderers' uv environments once on the login node so the jobs never race on them (prewarm_figures.sh); a local rehearsal skips it.
case "$host" in
    */*) ;;
    *) if [ "$dry" = 1 ]; then
        echo "would run: ssh $ssh_host \"bash '$base_path/code/scripts/hpc/prewarm_figures.sh'\""
    else
        # shellcheck disable=SC2029
        ssh "$ssh_host" "bash '$base_path/code/scripts/hpc/prewarm_figures.sh'" || exit 1
    fi ;;
esac

if [ "$data" != none ]; then
    data_root="${TC_DATA_ROOT:-$repo/data}"
    data_args=(-a --partial --info=progress2 --exclude='*.DS_Store')
    [ "$dry" = 1 ] && data_args+=(--dry-run --itemize-changes)
    if [ "$data" = all ]; then
        echo "all of data/ -> $base/data/"
        data_args+=("$data_root/")
    else
        echo "${inputs[*]} -> $base/data/"
        # --relative with the /./ pivot keeps <family>/preprocessed/ under data/ without sending the families' other subdirectories.
        data_args+=(--relative)
        for item in "${inputs[@]}"; do
            data_args+=("$data_root/./$item")
        done
    fi
    rsync "${data_args[@]}" "$base/data/" || exit 1
fi

echo "local HEAD $(git -C "$repo" rev-parse HEAD 2>/dev/null || echo unknown) (the remote tree carries no .git)"
[ "$dry" = 1 ] && echo "dry run: nothing was written"
exit 0
