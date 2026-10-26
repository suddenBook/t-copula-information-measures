"""Phase 5 figure driver: the pair copula densities of one recording, read from its Phase 5 folder.

`renderPhaseFigures.m` runs `uv run --quiet render_phase5_figures.py <Phase5 dir> <images dir>` and copies the driver's printed output into the phase log.
The driver lists the files it wrote in `figures_manifest.json` beside the images.

Figure stems and the question each answers (each is also written as `<stem>_log`, the same figure on a logarithmic colour scale):

  phase5_density_<band>_<rank>  For one selected pair: what does the empirical copula density look like, what do the Gaussian and the fitted t say, and what does each of them miss?
  phase5_density_summary        The same for one pair per concordance tertile, so the three regimes are read side by side.

Under uv run, only the PEP 723 header of the entry script builds the environment, so it lists the dependencies of every renderer it imports.

A renderer that raises is reported and skipped, and the exit code is the number of renderers that failed.
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
import render_phase5_density as R_density  # noqa: E402

RENDERERS = [("density", R_density)]


def render_all(source, out_dir, driver="render_phase5_figures"):
    """Run every renderer on one Phase 5 folder, update the manifest, and return the number of renderers that failed."""
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
    print(f"phase5 figures: {len(stems)} figures ({len(produced)} files) from {len(RENDERERS) - len(failures)}/{len(RENDERERS)} renderers -> {out_dir}")
    if failures:
        print(f"phase5 figures: FAILED renderers: {', '.join(failures)}", file=sys.stderr)
    return len(failures)


def main():
    """Command-line entry; the images folder defaults to the phase's own, and a missing argument returns 2."""
    if len(sys.argv) < 2:
        print("usage: render_phase5_figures.py <Phase5 dir> [images dir]")
        return 2
    source = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else IO.phase_paths(source)["images_dir"]
    return render_all(source, out_dir)


if __name__ == "__main__":
    sys.exit(main())
