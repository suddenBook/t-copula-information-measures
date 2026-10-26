"""Phase 4 figure driver: the information measures of one recording, from its phase folder.

`renderPhaseFigures.m` runs `uv run --quiet render_phase4_figures.py <Phase4 dir> <images dir>`, then lists the image files itself and records them in `text/figures_manifest.json`; the driver writes its own `figures_manifest.json` beside the images.

Figure stems and the question each answers, each also written as `<stem>_log`:

  phase4_differences_vs_dimension_nats  How large is each difference, in nats, at each interaction order?
  phase4_differences_vs_dimension_pct   The same as a percentage of the Gaussian value.
  phase4_setwise_vs_gaussian            Channel set by channel set, the difference against the Gaussian value, with constant-percentage guides and the surrogate floor.
  phase4_pct_by_decile                  Does a large percentage only appear where the Gaussian denominator is small?
  phase4_o_sign_flips                   How often does the O-information change sign?
  phase4_ksg_vs_model                   Does the closed-form value agree with the model-free one at d = 2 and d = 3?
  phase4_theory_grid                    The shape term and the four differences as functions of p; the log twin reads them in nu.

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
import render_phase4_differences as R_differences  # noqa: E402
import render_phase4_checks as R_checks  # noqa: E402
import render_phase4_theory as R_theory  # noqa: E402

RENDERERS = [("differences", R_differences), ("checks", R_checks), ("theory", R_theory)]


def render_all(source, out_dir, driver="render_phase4_figures"):
    """Run every renderer on one Phase 4 output folder, write the figure manifest, and return the number of renderers that failed."""
    os.makedirs(out_dir, exist_ok=True)
    failures, produced = [], []
    for label, module in RENDERERS:
        started = time.time()
        try:
            saved = module.render(source, out_dir) or []
            produced.extend(saved)
            stems = sorted({os.path.splitext(os.path.basename(p))[0] for p in saved})
            print(f"  [ok]   {label:12s} {time.time() - started:5.1f}s  {', '.join(stems) or '(no figure)'}", flush=True)
        except Exception:
            failures.append(label)
            print(f"  [FAIL] {label:12s} {time.time() - started:5.1f}s", flush=True)
            traceback.print_exc()
    stems = {os.path.splitext(os.path.basename(p))[0] for p in produced}
    S.manifest(out_dir, produced, driver=driver, source=source, ok=not failures)
    print(f"phase4 figures: {len(stems)} figures ({len(produced)} files) from {len(RENDERERS) - len(failures)}/{len(RENDERERS)} renderers -> {out_dir}")
    if failures:
        print(f"phase4 figures: FAILED renderers: {', '.join(failures)}", file=sys.stderr)
    return len(failures)


def main():
    """Command-line entry: the Phase 4 folder, then optionally the images folder, which defaults to the phase's own."""
    if len(sys.argv) < 2:
        print("usage: render_phase4_figures.py <Phase4 dir> [images dir]")
        return 2
    source = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else IO.phase_paths(source)["images_dir"]
    return render_all(source, out_dir)


if __name__ == "__main__":
    sys.exit(main())
