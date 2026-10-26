"""Phase 2 figure driver: every t-copula fit figure of one recording, from its phase directory.

`src/visualization/renderPhaseFigures.m` runs one command per phase, `uv run --quiet render_phase2_figures.py <Phase2 dir> <images dir>`, then lists the image files itself and records them in `text/figures_manifest.json`; this driver writes its own `figures_manifest.json` beside the images.

Figure stems and the question each answers (each is also written as `<stem>_log`):

  phase2_shape_vs_dimension     Where does the fitted shape sit at each interaction order, what tail index is that, and how much of the mass is exactly Gaussian?
  phase2_pair_shape_vs_rho      At d = 2, does the fitted shape follow concordance, and what likelihood does the extra parameter buy?
  phase2_heavy_boundary         How often does a fit land at the far end of the domain?
  phase2_status_by_dimension    How did the fits end, in the status precedence of tFitUnit?
  phase2_fit_seconds            What does one fit cost, and what does the sweep cost?
  phase2_repair_shift           How far does the eigenvalue-floor repair move R?
  phase2_profiles_by_dimension  What does the profile likelihood look like near the estimate (no log twin, because it is a likelihood difference)?

Under uv run, only the PEP 723 header of the entry script builds the environment, so it lists the dependencies of every renderer it imports.

A renderer that raises is reported and skipped and never aborts the rest, because figures are exposition and the phase's data is already on disk when this runs.
The exit code is the number of renderers that failed.
"""

# /// script
# requires-python = ">=3.10"
# dependencies = ["h5py>=3.10", "matplotlib>=3.10", "numpy>=1.26", "pandas>=2.0"]
# ///
from __future__ import annotations

import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _phase_io as IO  # noqa: E402
import _style as S  # noqa: E402
import render_phase2_shape as R_shape  # noqa: E402
import render_phase2_status as R_status  # noqa: E402
import render_phase2_profiles as R_profiles  # noqa: E402

RENDERERS = [("shape", R_shape), ("status", R_status), ("profiles", R_profiles)]


def render_all(source, out_dir, driver="render_phase2_figures"):
    os.makedirs(out_dir, exist_ok=True)
    failures, produced = [], []
    for label, module in RENDERERS:
        started = time.time()
        try:
            saved = module.render(source, out_dir) or []
            produced.extend(saved)
            stems = sorted({os.path.splitext(os.path.basename(p))[0] for p in saved})
            print(f"  [ok]   {label:10s} {time.time() - started:5.1f}s  {', '.join(stems) or '(no figure)'}", flush=True)
        except Exception:
            failures.append(label)
            print(f"  [FAIL] {label:10s} {time.time() - started:5.1f}s", flush=True)
            traceback.print_exc()
    stems = {os.path.splitext(os.path.basename(p))[0] for p in produced}
    S.manifest(out_dir, produced, driver=driver, source=source, ok=not failures)
    print(f"phase2 figures: {len(stems)} figures ({len(produced)} files) from {len(RENDERERS) - len(failures)}/{len(RENDERERS)} renderers -> {out_dir}")
    if failures:
        print(f"phase2 figures: FAILED renderers: {', '.join(failures)}", file=sys.stderr)
    return len(failures)


def main():
    if len(sys.argv) < 2:
        print("usage: render_phase2_figures.py <Phase2 dir> [images dir]")
        return 2
    source = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else IO.phase_paths(source)["images_dir"]
    return render_all(source, out_dir)


if __name__ == "__main__":
    sys.exit(main())
