"""Phase 7 figure driver: the simulation study, from its phase directory.

`renderPhaseFigures.m` runs `uv run --quiet render_phase7_figures.py <Phase7 dir> <images dir>`, then lists the image files itself and records them in `text/figures_manifest.json`; the driver writes its own `figures_manifest.json` beside the images.

The figure stems and the question each answers are listed below.
A stem has a `<stem>_log` twin where a log axis applies, and the panels where a log axis is the point carry no linear twin.
The replicates are paths of the Student-t diffusion in units of tau, so the amount of data is the observation window T/tau.
The samples per tau are a nested axis of the recovery grid (every rate reads the same paths), so a stem marked `_sXX` is written once per rate the tables carry, with the rate zero-padded to two digits (`rate_suffix`): `_s05`, `_s10`, `_s20`, `_s40`, `_s80` and `_s160` at the declared rates.
A per-rate stem that has a `_log` twin writes it after the rate (`phase7_recovery_bias_s10_log`); `phase7_recovery_spread_sXX` and the information figures `_estimated_vs_true_sXX` and `_relative_error_sXX` have none.
Figure text writes the shape as 1/nu, never p.

  phase7_recovery_spread_sXX                Does the relative spread fall like (T/tau)^-1/2, and what is the spread law at each dimension?
  phase7_recovery_bias_sXX                  Where in p does the estimate carry bias?
  phase7_recovery_gaussian_truth_sXX        At a Gaussian truth, how much mass lands exactly on p = 0 and how large is the rest?
  phase7_recovery_heavy_sXX                 How often does a fit land at the heavy end?
  phase7_recovery_sampling                  Does denser sampling of the same window change the estimate?
                                            Median error with its interquartile band and sd of p_hat against the samples per tau, for representative truths and dimensions, one line per T/tau.
  phase7_factorial_bias                     Does a change of correlation design move the bias, at one window and one rate?
  phase7_information_estimated_vs_true_sXX  Do the estimated t-minus-Gaussian corrections track the true ones?
  phase7_information_relative_error_sXX     How fast does the relative error fall with T/tau, for the correction and for the full measure?
  phase7_information_components_sXX         Shape, correlation, or their interaction: where does the full-measure error come from?
  phase7_information_floor_sXX              What correction does a Gaussian truth still produce by (d, T/tau), and how large is the full-measure error beside it?
  phase7_check_law                          Is the one-time law of the simulated paths the Student law it should be, at time 0 and later, read against the Bonferroni critical value?
  phase7_check_coarse                       At a forced coarse step in the heavy corner, does the Metropolis chain keep the Student law where the frozen chain, the negative control, loses it?
  phase7_check_refinement                   Does refining the integration step move the mean or the spread of p_hat, read as one pooled interval per nu and step pair against the tolerance, and how slow does the chain's clock run?
  phase7_check_discretisation               How far can the clock error move p_hat, in sd of p_hat, against the tolerance?
  phase7_check_timescale                    What do the Phase 6 estimator's tau_env, tau_amp, Gaussian reference and amplitude excess read on the model's own paths over the recordings' lag windows?
  phase7_check_sampling                     Does the estimate still change with more samples per tau at a fixed window, per nu?

Under uv run, only the PEP 723 header of the entry script builds the environment, so it lists the dependencies of every renderer it imports.

# debt: `recovery_long.csv` is read whole into a DataFrame by two renderers, which at the declared grid (some 8200 cells at six rates x 1000 replicates x ~80 columns) takes several gigabytes of RAM.
# Reading it per measure with `usecols`, or reading `recovery_summary.csv` instead and resolving its column spellings, is the upgrade path for the production grid.

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
import render_phase7_shape_recovery as R_recovery  # noqa: E402
import render_phase7_factorial as R_factorial  # noqa: E402
import render_phase7_information as R_information  # noqa: E402
import render_phase7_simulator_check as R_check  # noqa: E402

RENDERERS = [("recovery", R_recovery), ("factorial", R_factorial), ("information", R_information), ("check", R_check)]


def render_all(source, out_dir, driver="render_phase7_figures"):
    """Run every Phase 7 renderer on `source`, write the figure manifest in `out_dir`, and return the number of renderers that failed."""
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
    print(f"phase7 figures: {len(stems)} figures ({len(produced)} files) from {len(RENDERERS) - len(failures)}/{len(RENDERERS)} renderers -> {out_dir}")
    if failures:
        print(f"phase7 figures: FAILED renderers: {', '.join(failures)}", file=sys.stderr)
    return len(failures)


def main():
    """Command-line entry: the Phase 7 directory, and optionally the images directory, which defaults to the phase's own."""
    if len(sys.argv) < 2:
        print("usage: render_phase7_figures.py <Phase7 dir> [images dir]")
        return 2
    source = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else IO.phase_paths(source)["images_dir"]
    return render_all(source, out_dir)


if __name__ == "__main__":
    sys.exit(main())
