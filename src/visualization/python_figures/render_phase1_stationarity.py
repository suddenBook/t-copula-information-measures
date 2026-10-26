"""Phase 1 QC figure: raw-input sliding-window amplitude summary.

A two-panel descriptive diagnostic of the `stationarity_info` that checkStationarity.m computes on the signal before artifact cleaning and band-pass filtering.
The persisted struct contains every plotted array, so this renderer reads it directly and recomputes nothing:
    panel 1: channel-averaged window variance against time, with the mean and mean*(1 +/- cv_var) guides;
    panel 2: channel-averaged absolute window mean against time.

Run:
    uv run src/visualization/python_figures/render_phase1_stationarity.py <preprocessed_data.mat> [out_dir]
"""

# /// script
# requires-python = ">=3.10"
# dependencies = ["h5py>=3.10", "matplotlib>=3.8", "numpy>=1.26"]
# ///
from __future__ import annotations

import os
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _phase_data as D  # noqa: E402


def render(mat_path, out_dir):
    """Draw the stationarity figure of one preprocessed_data.mat and return the saved paths ([] without stationarity_info)."""
    os.makedirs(out_dir, exist_ok=True)
    D.apply_publication_style()
    with h5py.File(mat_path, "r") as h:
        if not D.has(h, "stationarity_info"):
            print("stationarity: stationarity_info absent, skipping")
            return []
        s = D.read_struct(h, "stationarity_info")
    t = s["window_times"]
    wvar = s["window_var"]
    wmean = s["window_mean"]
    cv = float(s["cv_var"].ravel()[0]) if "cv_var" in s else float("nan")
    mean_var = float(np.mean(wvar))
    cv_label = f"{cv:.3f}" if np.isfinite(cv) else "undefined"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True)
    ax1.plot(t, wvar, "-o", color=D.PALETTE["blue_main"], ms=3, lw=1.3)
    ax1.axhline(mean_var, ls="-", color=D.PALETTE["neutral"], lw=1.2, label="mean")
    if np.isfinite(cv):
        ax1.axhline(mean_var * (1 + cv), ls="--", color=D.PALETTE["red_strong"], lw=1.0)
        ax1.axhline(mean_var * (1 - cv), ls="--", color=D.PALETTE["red_strong"], lw=1.0, label=r"mean $\pm$ 1 window SD")
    ax1.set_ylabel("window variance")
    ax1.set_title(f"Raw-input sliding-window variance  (descriptive CV = {cv_label})", fontsize=11)
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.25)

    ax2.plot(t, wmean, "-o", color=D.PALETTE["green_3"], ms=3, lw=1.3)
    ax2.set_xlabel("time  [s]")
    ax2.set_ylabel("window mean |.|")
    ax2.grid(True, alpha=0.25)

    fig.text(0.01, 0.005, "Raw input before artifact cleaning and band-pass filtering. Sliding-window variance/mean and CV are descriptive, non-exclusionary QC: no stationarity test, verdict, or inclusion rule is applied.", fontsize=7, color="#555555", wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return D.finalize(fig, out_dir, "phase1_stationarity")


def main() -> None:
    """Command-line entry point: render one MAT file into out_dir (default: the MAT file's folder)."""
    if len(sys.argv) < 2:
        print("usage: render_phase1_stationarity.py <preprocessed_data.mat> [out_dir]")
        return
    mat_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(mat_path))
    print("saved:", render(mat_path, out_dir))


if __name__ == "__main__":
    main()
