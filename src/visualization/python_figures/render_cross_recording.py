"""Cross-recording driver: every recording of a results root, rolled up into one tree.

Run:
    uv run src/visualization/python_figures/render_cross_recording.py <results root> [output base]

Recordings are discovered through the `recording.json` sidecar Phase 1 writes at each recording's base (`<results root>/*/*/*/recording.json`) and ordered by `seed_base`, so the rollup does not depend on a manifest and picks up exactly what has been run.

What it writes, under `<results root>/cross_recording/` unless an output base is given.
The phase a figure lands in is the phase whose numbers it draws.

  Phase2/images/cross_f1_shape_vs_dimension
      F1: p_hat and nu_hat against dimension, one line per recording, one panel per modality plus an all-recordings panel, with the envelope of the recordings' own surrogate intervals behind the p_hat row.
  Phase2/text/recording_table
      The recording table, one row per recording.
  Phase3/images/cross_f1_overlap_vs_dimension
      How much of each recording's observed p_hat distribution its own surrogate covers, per dimension, beside its two yardstick overlaps.
  Phase3/text/overlap_by_recording_and_dimension
  Phase4/images/cross_f2_differences_vs_dimension_nats
  Phase4/images/cross_f2_differences_vs_dimension_pct
  Phase4/images/cross_f2_setwise_vs_gaussian
      F2, pooled over recordings, with the constant-percentage guides and the floor.
  Phase4/text/recording_medians_at_low_and_high_d
  Phase5/images/cross_f3_density_by_modality
      F3: one density row per modality.
  Phase6/text/timescale_table
      tau_env, tau_amp, its Gaussian reference and the amplitude excess, fitted and integrated, samples per tau, T/tau and the lag window in tau_env per recording, with the within-trial-centring twins and the ranges the simulation's grid has to cover.
  Phase7/text/timescale_matched_window
      Per recording and model nu, the model lag window matched to the recording's, in tau and in the model's own tau_env.
  Phase6/images/cross_f6_timescales
      How long does each recording remember its signal and its amplitude, and how much longer is its amplitude memory than a Gaussian signal's (the excess, fitted and integrated)?
  Phase7/images/cross_f7_recordings_on_time_axis
      What window and rate does an accurate estimate need, and where do the recordings sit?
      The error of 1/nu_hat against T/tau, one line per sampling rate, with each recording placed at its own T/tau_env and its samples per tau_env.
  Phase7/images/cross_f7_amplitude_memory
      Does each recording carry more amplitude memory than the model at its shape?
      The model's excess, fitted and integrated, against nu at the lag windows matched to each recording per nu, with each recording's excess.

The Phase 7 table and figures read the study under `<results root>/simulation/t_copula/all/Phase7` and are skipped without it.
Every figure is written as PDF and PNG at 300 dpi, with a `<stem>_log` twin where a logarithmic axis applies, every table as CSV and Markdown, and each images directory gets a `figures_manifest.json` that lists its figure files.

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
import render_cross_shape as R_shape  # noqa: E402
import render_cross_differences as R_differences  # noqa: E402
import render_cross_density as R_density  # noqa: E402
import render_cross_tables as R_tables  # noqa: E402
import render_cross_timescale as R_timescale  # noqa: E402

PHASES = (2, 3, 4, 5, 6, 7)
RENDERERS = [("shape", R_shape), ("differences", R_differences), ("density", R_density), ("tables", R_tables), ("timescale", R_timescale)]


def render_all(results_root, output_base, driver="render_cross_recording"):
    """Run every renderer on the recordings under `results_root` and write the manifests; return the number of renderers that failed."""
    recordings = IO.discover_recordings(results_root)
    if not recordings:
        print(f"cross-recording: no recording.json under {results_root}/*/*/*; nothing to do")
        return 0
    print(f"cross-recording: {len(recordings)} recordings -> {output_base}")
    for phase in PHASES:
        for child in ("images", "text"):
            os.makedirs(os.path.join(output_base, f"Phase{phase}", child), exist_ok=True)

    failures, produced = [], []
    for label, module in RENDERERS:
        started = time.time()
        try:
            saved = module.render(recordings, output_base) or []
            produced.extend(saved)
            stems = sorted({os.path.splitext(os.path.basename(p))[0] for p in saved})
            print(f"  [ok]   {label:12s} {time.time() - started:5.1f}s  {', '.join(stems) or '(nothing written)'}", flush=True)
        except Exception:
            failures.append(label)
            print(f"  [FAIL] {label:12s} {time.time() - started:5.1f}s", flush=True)
            traceback.print_exc()

    by_directory = {}
    for path in produced:
        by_directory.setdefault(os.path.dirname(path), []).append(path)
    for directory, paths in by_directory.items():
        if os.path.basename(directory) == "images":
            S.manifest(directory, paths, driver=driver, source=results_root, ok=not failures)
    stems = {os.path.splitext(os.path.basename(p))[0] for p in produced}
    print(f"cross-recording: {len(stems)} figures and tables ({len(produced)} files) from {len(RENDERERS) - len(failures)}/{len(RENDERERS)} renderers -> {output_base}")
    if failures:
        print(f"cross-recording: FAILED renderers: {', '.join(failures)}", file=sys.stderr)
    return len(failures)


def main():
    if len(sys.argv) < 2:
        print("usage: render_cross_recording.py <results root> [output base]")
        return 2
    results_root = os.path.abspath(sys.argv[1])
    output_base = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(results_root, "cross_recording")
    return render_all(results_root, output_base)


if __name__ == "__main__":
    sys.exit(main())
