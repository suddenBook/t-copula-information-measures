"""Phase 3 figure driver: the Gaussian surrogate reference of one recording.

`renderPhaseFigures.m` runs `uv run --quiet render_phase3_figures.py <Phase3 dir> <images dir>`, then lists the image files itself and records them in `text/figures_manifest.json`; the driver writes its own `figures_manifest.json` beside the images.

Figure stems and the question each answers, with a `<stem>_log` twin where a log axis applies:

  phase3_distributions_by_dimension      Per dimension, do the observed distribution of p_hat = 1/nu_hat and the surrogate one lie on each other, the way two observed halves and two surrogate realizations do?
  phase3_overlap_vs_dimension            The overlap coefficient of observed against surrogate, and of the two yardsticks, against d, with the mass outside the overlap.
  phase3_surrogate_floor                 How large a difference does the reference itself produce, per measure and per dimension?
  phase3_gaussian_fraction_vs_dimension  How much of each distribution sits exactly on the Gaussian?

Under uv run, only the PEP 723 header of the entry script builds the environment, so it lists the dependencies of every renderer it imports.

A renderer that raises is reported and skipped; the exit code is the number of renderers that failed.
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
import render_phase3_surrogate as R_surrogate  # noqa: E402
import render_phase3_units as R_units  # noqa: E402

RENDERERS = [("surrogate", R_surrogate), ("units", R_units)]


def render_all(source, out_dir, driver="render_phase3_figures"):
    """Run every renderer on one Phase 3 output folder, write the figure manifest, and return the number of renderers that failed."""
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
    print(f"phase3 figures: {len(stems)} figures ({len(produced)} files) from {len(RENDERERS) - len(failures)}/{len(RENDERERS)} renderers -> {out_dir}")
    if failures:
        print(f"phase3 figures: FAILED renderers: {', '.join(failures)}", file=sys.stderr)
    return len(failures)


def main():
    """Command-line entry: the Phase 3 folder, then optionally the images folder, which defaults to the phase's own."""
    if len(sys.argv) < 2:
        print("usage: render_phase3_figures.py <Phase3 dir> [images dir]")
        return 2
    source = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else IO.phase_paths(source)["images_dir"]
    return render_all(source, out_dir)


if __name__ == "__main__":
    sys.exit(main())
