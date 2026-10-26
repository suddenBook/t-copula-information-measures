"""Phase 1 figure driver: render every Phase 1 QC figure from one preprocessed_data.mat.

This is the one command scripts/Phase1_preprocessing.m runs, through src/visualization/renderPhaseFigures.m.
Each phase has a driver like this one: it imports the phase's renderers and calls each `render(mat_path, out_dir)` in one interpreter, so a phase resolves its uv environment once rather than once per figure.
Under uv run, only the PEP 723 header of the entry script builds the environment, so it lists the dependencies of every renderer it imports.

Figures are descriptive QC and provenance evidence, not hypothesis tests or model-fit verdicts.
A renderer that raises is reported and skipped; it never aborts the run or the remaining renderers.
The exit code is the number of renderers that failed (2 for a usage error), so the MATLAB side can record a partial result honestly.

Renderers with nothing to draw return an empty list and count as successful: without an applied band-pass there is no raw-vs-filtered comparison, and without excluded channels there are no excluded-channel pages.

Run:
    uv run src/visualization/python_figures/render_phase1_figures.py <preprocessed_data.mat> [out_dir]
"""

# /// script
# requires-python = ">=3.10"
# dependencies = ["h5py>=3.10", "matplotlib>=3.8", "numpy>=1.26", "scipy>=1.11"]
# ///
from __future__ import annotations

import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import render_phase1_band_selection_psd as R_bandsel  # noqa: E402
import render_phase1_psd as R_psd  # noqa: E402
import render_phase1_stationarity as R_stationarity  # noqa: E402
import render_phase1_marginals as R_marginals  # noqa: E402
import render_phase1_time_domain as R_time_domain  # noqa: E402
import render_phase1_statistics as R_statistics  # noqa: E402
import render_phase1_uniformity as R_uniformity  # noqa: E402
import render_phase1_cleaning as R_cleaning  # noqa: E402
import render_phase1_excluded_channels as R_excluded  # noqa: E402
import render_phase1_artifact_quality as R_artifact  # noqa: E402

# (label, module) pairs in reading order.
RENDERERS = [("band_selection_psd", R_bandsel), ("psd", R_psd), ("time_domain", R_time_domain), ("stationarity", R_stationarity), ("marginals", R_marginals), ("statistics", R_statistics), ("uniformity", R_uniformity), ("artifact_quality", R_artifact), ("cleaning", R_cleaning), ("excluded_channels", R_excluded)]


def render_all(mat_path: str, out_dir: str) -> int:
    """Run every Phase 1 renderer on one MAT file, print one status line per renderer and return the number that raised."""
    os.makedirs(out_dir, exist_ok=True)
    failures, produced = [], []
    for label, mod in RENDERERS:
        t0 = time.time()
        try:
            saved = mod.render(mat_path, out_dir) or []
            names = sorted({os.path.basename(p) for p in saved})
            produced.extend(names)
            note = ", ".join(sorted({os.path.splitext(n)[0] for n in names})) or "(no figure)"
            print(f"  [ok]   {label:20s} {time.time() - t0:5.1f}s  {note}", flush=True)
        except Exception:
            failures.append(label)
            print(f"  [FAIL] {label:20s} {time.time() - t0:5.1f}s", flush=True)
            traceback.print_exc()

    n_fig = len({os.path.splitext(n)[0] for n in produced})
    print(f"phase1 figures: {n_fig} figures ({len(produced)} files) from {len(RENDERERS) - len(failures)}/{len(RENDERERS)} renderers -> {out_dir}")
    if failures:
        print(f"phase1 figures: FAILED renderers: {', '.join(failures)}", file=sys.stderr)
    return len(failures)


def main() -> int:
    """Command-line entry point: render one MAT file into out_dir (default: the MAT file's folder) and return the exit code."""
    if len(sys.argv) < 2:
        print("usage: render_phase1_figures.py <preprocessed_data.mat> [out_dir]")
        return 2
    mat_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(mat_path))
    return render_all(mat_path, out_dir)


if __name__ == "__main__":
    sys.exit(main())
