#!/bin/bash
# prewarm_figures.sh: run on the cluster login node (deploy.sh does it after every transfer).
#
# Installs the uv-managed interpreter the figure renderers use and builds the environment of every renderer driver once, sequentially, in the shared uv cache.
# Jobs then run the renderers with UV_NO_SYNC=1 and only read those environments, so concurrent jobs never build the same environment at the same time on the shared filesystem, which would corrupt it.
# It needs internet access on the login node (PyPI), and a renderer whose dependency header changed gets its new environment at the next deploy.
set -u
site_file="$(dirname "$0")/site.env"
# shellcheck source=/dev/null
[ -f "$site_file" ] && . "$site_file"
export PATH="$HOME/.local/bin:$PATH"
python="${TC_HPC_UV_PYTHON:-3.12}"
code="$(cd "$(dirname "$0")/../.." && pwd)"
uv python install "$python" || exit 1
export UV_PYTHON="$python" UV_PYTHON_DOWNLOADS=never
status=0
for driver in "$code"/src/visualization/python_figures/render_phase*_figures.py \
    "$code"/src/visualization/python_figures/render_cross_recording.py; do
    [ -f "$driver" ] || continue
    printf '[prewarm] %s\n' "$(basename "$driver")"
    uv sync --script "$driver" || status=1
done
exit "$status"
