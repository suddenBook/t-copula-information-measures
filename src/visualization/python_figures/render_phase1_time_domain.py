"""Phase 1 QC figures: time-domain signal traces.

Up to five channels stacked over the whole recording: the artifact-cleaned signal before the band-pass and, when X_filtered has the shape of X and differs from it, a comparison with the band-pass-filtered signal.
The renderer reads the persisted arrays from results/<family>/<subject>/<condition>/Phase1/mat/preprocessed_data.mat and computes no science.

Run:
    uv run src/visualization/python_figures/render_phase1_time_domain.py <preprocessed_data.mat> [out_dir]
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

MAX_CHANNELS = 5


def _plot_stack(mat, fs, out_dir, name, title, overlay=None, overlay_label=None):
    """Plot the first MAX_CHANNELS channels of mat ([channels x samples]) in stacked axes, optionally overlaid with the same channels of overlay, and save the figure under name."""
    n_ch = min(MAX_CHANNELS, mat.shape[0])
    t = np.arange(mat.shape[1]) / fs
    fig, axes = plt.subplots(n_ch, 1, figsize=(9.0, 1.7 * n_ch + 0.6), squeeze=False, sharex=True)
    axes = axes.ravel()
    for ch in range(n_ch):
        ax = axes[ch]
        ax.plot(t, mat[ch], color=D.PALETTE["blue_main"], lw=0.7, label="artifact-cleaned pre-band-pass" if overlay is not None else None)
        if overlay is not None:
            ax.plot(t, overlay[ch], color=D.PALETTE["red_strong"], lw=0.7, label=overlay_label)
        yr = 1.1 * np.max(np.abs(mat[ch])) if np.any(mat[ch]) else 1.0
        ax.set_ylim(-yr, yr)
        ax.set_ylabel(f"Ret. col {ch + 1}", fontsize=9)
        ax.grid(True, alpha=0.25)
    if overlay is not None:
        axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("Time  [s]")
    fig.suptitle(title, fontsize=12, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return D.finalize(fig, out_dir, name)


def render(mat_path, out_dir):
    """Draw the trace figures of one preprocessed_data.mat and return the saved paths."""
    os.makedirs(out_dir, exist_ok=True)
    D.apply_publication_style()
    saved = []
    with h5py.File(mat_path, "r") as h:
        fs = D.read_scalar(h, "fs")
        X = D.read_matrix(h, "X")  # cleaned, unfiltered
        saved += _plot_stack(X, fs, out_dir, "phase1_time_domain_raw", "Time-domain traces (artifact-cleaned, pre-band-pass)")
        if D.has(h, "X_filtered"):
            Xf = D.read_matrix(h, "X_filtered")
            if Xf.shape == X.shape and not np.allclose(Xf, X):
                saved += _plot_stack(X, fs, out_dir, "phase1_time_domain_raw_vs_filtered", ("Time-domain traces: artifact-cleaned pre-band-pass vs Phase-1 band-pass-filtered"), overlay=Xf, overlay_label="Phase-1 band-pass filtered")
    return saved


def main() -> None:
    """Command-line entry point: render one MAT file into out_dir (default: the MAT file's folder)."""
    if len(sys.argv) < 2:
        print("usage: render_phase1_time_domain.py <preprocessed_data.mat> [out_dir]")
        return
    mat_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(mat_path))
    print("saved:", render(mat_path, out_dir))


if __name__ == "__main__":
    main()
