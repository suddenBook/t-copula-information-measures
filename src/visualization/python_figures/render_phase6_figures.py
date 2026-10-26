"""Phase 6 figure driver: the autocorrelation time scales of one recording, from its phase folder.

`renderPhaseFigures.m` runs `uv run --quiet render_phase6_figures.py <Phase6 dir> <images dir>`, then lists the image files itself and records them in `text/figures_manifest.json`; the driver writes its own `figures_manifest.json` beside the images.

The figure stems and the question each answers; each figure is also written as `<stem>_log`, the same figure on a logarithmic axis.

phase6_acf
    How long does the recording remember its own signal and its own amplitude, does the fitted exponential follow the envelope of the autocorrelation it summarises, and how much longer is the amplitude memory than its Gaussian reference (the excess tau_amp / tau_amp_ref, 1 for a Gaussian signal)?
phase6_channel_timescales
    How far do the channels' own time scales and amplitude excesses spread around the values of their average, and which channels have none?

Under uv run, only the PEP 723 header of the entry script builds the environment, so it lists the dependencies of every renderer it imports.

A renderer that raises is reported and skipped; the exit code is the number that failed.
"""

# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.10", "numpy>=1.26", "pandas>=2.0"]
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
import render_phase6_timescale as R_timescale  # noqa: E402

RENDERERS = [("timescale", R_timescale)]


def render_all(source, out_dir, driver="render_phase6_figures"):
    """Run every renderer on the phase folder `source`, write the figure manifest, and return the number of renderers that failed."""
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
    print(f"phase6 figures: {len(stems)} figures ({len(produced)} files) from {len(RENDERERS) - len(failures)}/{len(RENDERERS)} renderers -> {out_dir}")
    if failures:
        print(f"phase6 figures: FAILED renderers: {', '.join(failures)}", file=sys.stderr)
    return len(failures)


def main():
    if len(sys.argv) < 2:
        print("usage: render_phase6_figures.py <Phase6 dir> [images dir]")
        return 2
    source = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else IO.phase_paths(source)["images_dir"]
    return render_all(source, out_dir)


if __name__ == "__main__":
    sys.exit(main())
