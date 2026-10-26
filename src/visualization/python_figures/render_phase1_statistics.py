"""Phase 1 QC figures: amplitude distribution and per-channel box plot.

Each figure pairs a pooled histogram (1000 bins over the data range, normalized to probabilities) with a per-channel box plot.
It is drawn for two signals: the artifact-cleaned signal X before the band-pass, read directly, and the rank-uniform transform of the analysis signal X_filtered.
The rank-uniform values are recomputed from the persisted X_filtered with _phase_data.rank_uniform (rank / (n + 1), average ranks for ties), because Phase 1 stores no rank-transformed matrices.
The histogram and box statistics are recomputed here as QC diagnostics, not results.
Axis limits are data-driven, so both raw (uV) and uniform ([0, 1]) inputs render legibly.

Run:
    uv run src/visualization/python_figures/render_phase1_statistics.py <preprocessed_data.mat> [out_dir]
"""

# /// script
# requires-python = ">=3.10"
# dependencies = ["h5py>=3.10", "matplotlib>=3.8", "numpy>=1.26", "scipy>=1.11"]
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

N_BINS = 1000
# (persisted source, output figure suffix, transform)
INPUTS = [("X", "X", None), ("X_filtered", "X_filtered_uniform", "uniform")]


def _signal_label(h, source, transform):
    """Figure label naming the signal, whether Phase 1 band-pass filtered it, and the rank-uniform transform when applied."""
    if source == "X":
        label = "artifact-cleaned signal before Phase-1 band-pass"
    else:
        applied = bool(D.read_scalar(h, "filter_applied")) if D.has(h, "filter_applied") else D.has(h, "filter_range")
        if applied:
            label = "Phase-1 band-pass-filtered analysis signal"
        elif D.has(h, "native_band"):
            label = "analysis signal (Phase-1 pass-through; upstream band-limited)"
        else:
            label = "analysis signal (Phase-1 pass-through)"
    return f"{label}; rank-uniform" if transform == "uniform" else label


def _stats_figure(mat, out_dir, name, label):
    """Draw the pooled histogram and per-channel box plot of mat ([channels x samples]) and save them under name."""
    pooled = mat.ravel()
    quantity = "rank-uniform value" if "rank-uniform" in label else "amplitude"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    counts, edges = np.histogram(pooled, bins=N_BINS)
    prob = counts / max(counts.sum(), 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax1.bar(centers, prob, width=np.diff(edges), color=D.PALETTE["blue_main"], align="center")
    lo, hi = np.percentile(pooled, [0.2, 99.8])
    if hi > lo:
        ax1.set_xlim(lo, hi)
    ax1.set_xlabel(quantity)
    ax1.set_ylabel("probability")
    ax1.set_title(f"Pooled {quantity} distribution ({label}, {N_BINS} bins)", fontsize=11)

    per_channel = [mat[ch] for ch in range(mat.shape[0])]
    ax2.boxplot(per_channel, showfliers=False, widths=0.6, medianprops=dict(color=D.PALETTE["red_strong"]))
    ylo, yhi = np.percentile(pooled, [1, 99])
    if yhi > ylo:
        m = 0.1 * (yhi - ylo)
        ax2.set_ylim(ylo - m, yhi + m)
    ax2.set_xlabel("retained-column position")
    ax2.set_ylabel(quantity)
    ax2.set_title(f"Per-channel spread ({label})", fontsize=11)
    if mat.shape[0] > 20:
        ax2.set_xticks([])

    fig.tight_layout()
    return D.finalize(fig, out_dir, name)


def render(mat_path, out_dir):
    """Draw the statistics figures of one preprocessed_data.mat for each signal it holds and return the saved paths."""
    os.makedirs(out_dir, exist_ok=True)
    D.apply_publication_style()
    saved = []
    with h5py.File(mat_path, "r") as h:
        for source, suffix, transform in INPUTS:
            if not D.has(h, source):
                continue
            label = _signal_label(h, source, transform)
            mat = D.read_matrix(h, source)
            if transform == "uniform":
                mat = D.rank_uniform(mat)
            saved += _stats_figure(mat, out_dir, f"phase1_statistics_{suffix}", label)
    return saved


def main() -> None:
    """Command-line entry point: render one MAT file into out_dir (default: the MAT file's folder)."""
    if len(sys.argv) < 2:
        print("usage: render_phase1_statistics.py <preprocessed_data.mat> [out_dir]")
        return
    mat_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(mat_path))
    print("saved:", render(mat_path, out_dir))


if __name__ == "__main__":
    main()
