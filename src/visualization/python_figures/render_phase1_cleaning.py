"""Phase 1 QC figure: artifact-cleaning summary.

A six-panel summary of the channel cleaning (cleanSEEGArtifacts.m): channel- and time-retention pies, a few original and cleaned sample traces, cut-segment durations and a retention-statistics bar.
It reads the persisted arrays from results/<family>/<subject>/<condition>/Phase1/mat/preprocessed_data.mat and computes no science.

Fields read (all in Phase 1's explicit save list):
  X_raw  -- pre-clean raw matrix [samples x channels] (original traces)
  X      -- artifact-cleaned analysis matrix before the band-pass
  clean_info.time_retention_pct / channel_retention_pct / cut_segments_time
  fs
Channel retention counts come from the matrix shapes (n_total = channels of X_raw, n_retained = channels of X).
In the default channel_only strategy they equal the lengths of the retained and excluded channel lists, and they avoid the -v7.3 empty-array ambiguity.

Run:
    uv run src/visualization/python_figures/render_phase1_cleaning.py <preprocessed_data.mat> [out_dir]
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

MAX_TRACE_CHANNELS = 3
TRACE_OFFSET = 200.0  # vertical offset between stacked traces, in signal units


def _ci_scalar(h, field, default=float("nan")):
    """A scalar field of the persisted clean_info struct, or `default` (NaN) when it is absent or unreadable."""
    try:
        return float(np.asarray(h["clean_info"][field]).ravel()[0])
    except Exception:
        return default


def _cut_durations(h):
    """Durations (s) of the cut segments in clean_info.cut_segments_time, or an empty array when nothing was cut (as in channel_only mode).

    MATLAB stores the [n x 2] matrix of start and end times, which h5py sees as (2, n); an empty [] carries the MATLAB_empty attribute."""
    if "clean_info" not in h or "cut_segments_time" not in h["clean_info"]:
        return np.array([])
    d = h["clean_info"]["cut_segments_time"]
    if bool(d.attrs.get("MATLAB_empty", 0)):
        return np.array([])
    a = np.asarray(d, dtype=float)
    if a.ndim != 2 or a.size == 0:
        return np.array([])
    if a.shape[0] == 2:
        return a[1, :] - a[0, :]
    if a.shape[1] == 2:
        return a[:, 1] - a[:, 0]
    return np.array([])


def _plot_traces(ax, mat, fs, title):
    """Plot the first MAX_TRACE_CHANNELS channels of mat ([channels x samples]), stacked TRACE_OFFSET apart."""
    n_plot = min(MAX_TRACE_CHANNELS, mat.shape[0])
    t = np.arange(mat.shape[1]) / fs
    for i in range(n_plot):
        ax.plot(t, mat[i] + i * TRACE_OFFSET, lw=0.6)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Time  [s]")
    ax.set_ylabel("Amplitude")


def render(mat_path, out_dir):
    """Draw the cleaning summary of one preprocessed_data.mat and return the saved paths."""
    os.makedirs(out_dir, exist_ok=True)
    D.apply_publication_style()
    with h5py.File(mat_path, "r") as h:
        fs = D.read_scalar(h, "fs")
        X_raw = D.read_matrix(h, "X_raw")
        X = D.read_matrix(h, "X")

        n_total = X_raw.shape[0]
        n_retained = X.shape[0]
        n_excluded = max(0, n_total - n_retained)
        time_ret = _ci_scalar(h, "time_retention_pct", 100.0)
        chan_ret = _ci_scalar(h, "channel_retention_pct", 100.0 * n_retained / max(n_total, 1))
        durations = _cut_durations(h)

    fig, axes = plt.subplots(3, 2, figsize=(11.0, 11.0))

    # (1) channel retention pie
    axes[0, 0].pie([n_retained, n_excluded], labels=["Retained", "Excluded"], colors=[D.PALETTE["blue_main"], D.PALETTE["red_strong"]], autopct="%1.0f%%")
    axes[0, 0].set_title("Channel Retention", fontsize=11)

    # (2) time retention pie
    axes[0, 1].pie([time_ret, max(0.0, 100.0 - time_ret)], labels=["Retained", "Cut"], colors=[D.PALETTE["blue_main"], D.PALETTE["neutral"]], autopct="%1.1f%%")
    axes[0, 1].set_title("Time Retention", fontsize=11)

    # (3) original sample traces / (4) cleaned sample traces
    _plot_traces(axes[1, 0], X_raw, fs, "Detector Input (Sample Channels)")
    _plot_traces(axes[1, 1], X, fs, "Analysis Signal After Channel Cleaning, Pre-Band-Pass")

    # (5) cut segment durations
    ax5 = axes[2, 0]
    if durations.size > 0:
        ax5.bar(np.arange(1, durations.size + 1), durations, color=D.PALETTE["blue_secondary"])
        ax5.set_title("Cut Segment Durations", fontsize=11)
        ax5.set_xlabel("Segment Index")
        ax5.set_ylabel("Duration (s)")
    else:
        ax5.text(0.5, 0.5, "No segments cut", ha="center", va="center")
        ax5.set_title("Cut Segments", fontsize=11)
        ax5.set_xticks([])
        ax5.set_yticks([])

    # (6) retention statistics bar
    ax6 = axes[2, 1]
    ax6.bar(["Channel Retention %", "Time Retention %"], [chan_ret, time_ret], color=[D.PALETTE["blue_main"], D.PALETTE["teal"]])
    ax6.set_ylim(0, 100)
    ax6.set_ylabel("Percentage")
    ax6.set_title("Retention Statistics", fontsize=11)

    fig.suptitle(f"Artifact cleaning results — {D.recording_name(mat_path)}", fontsize=15, fontweight="bold", y=0.997)
    fig.text(0.01, 0.004, "Frozen automatic policy: remove abnormal-kurtosis-flagged channels only; retain low-power, isolation, redundancy, temporal-rate, and powerline findings as audit diagnostics. The channel_only strategy deletes no samples.", fontsize=7, color="#555555", wrap=True)
    fig.tight_layout(rect=(0, 0.045, 1, 0.98))
    return D.finalize(fig, out_dir, "phase1_cleaning")


def main() -> None:
    """Command-line entry point: render one MAT file into out_dir (default: the MAT file's folder)."""
    if len(sys.argv) < 2:
        print("usage: render_phase1_cleaning.py <preprocessed_data.mat> [out_dir]")
        return
    mat_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(mat_path))
    print("saved:", render(mat_path, out_dir))


if __name__ == "__main__":
    main()
