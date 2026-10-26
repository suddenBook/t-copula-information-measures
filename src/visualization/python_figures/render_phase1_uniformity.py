"""Phase 1 QC figures: uniform-marginal range check.

The rank-uniform transform of a signal, flattened channel by channel and plotted on [0, 2], must lie inside [0, 1].
It is recomputed from the persisted X (artifact-cleaned) and X_filtered with _phase_data.rank_uniform (rank / (n + 1), average ranks for ties), because Phase 1 stores no rank-transformed matrices and the fitting phases rank the data themselves.
The figure is a deterministic diagnostic, not a result.

Run:
    uv run src/visualization/python_figures/render_phase1_uniformity.py <preprocessed_data.mat> [out_dir]
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

# (source signal to rank -> uniform, output figure suffix)
FIELDS = [("X", "X_raw_uniform"), ("X_filtered", "X_filtered_uniform")]


def _signal_label(h, source):
    """Figure label naming the signal and whether Phase 1 band-pass filtered it."""
    if source == "X":
        return "artifact-cleaned signal before Phase-1 band-pass"
    applied = bool(D.read_scalar(h, "filter_applied")) if D.has(h, "filter_applied") else D.has(h, "filter_range")
    if applied:
        return "Phase-1 band-pass-filtered analysis signal"
    if D.has(h, "native_band"):
        return "analysis signal (Phase-1 pass-through; upstream band-limited)"
    return "analysis signal (Phase-1 pass-through)"


def render(mat_path, out_dir):
    """Draw the range-check figure for each signal one preprocessed_data.mat holds and return the saved paths."""
    os.makedirs(out_dir, exist_ok=True)
    D.apply_publication_style()
    saved = []
    with h5py.File(mat_path, "r") as h:
        for source, field in FIELDS:
            if not D.has(h, source):
                continue
            label = _signal_label(h, source)
            U = D.rank_uniform(D.read_matrix(h, source)).ravel(order="C")  # channel-by-channel
            fig, ax = plt.subplots(figsize=(9.0, 3.2))
            ax.plot(np.arange(U.size), U, color=D.PALETTE["teal"], lw=0.4)
            ax.axhline(0.0, ls="--", color=D.PALETTE["neutral"], lw=1.0)
            ax.axhline(1.0, ls="--", color=D.PALETTE["neutral"], lw=1.0)
            ax.set_ylim(0, 2)
            ax.set_xlabel("flattened sample index (retained-column-by-column)")
            ax.set_ylabel("uniform value")
            ax.set_title(f"Rank-uniform range check ({label}) — constructed in [0, 1]", fontsize=11)
            saved += D.finalize(fig, out_dir, f"phase1_uniformity_{field}")
    return saved


def main() -> None:
    """Command-line entry point: render one MAT file into out_dir (default: the MAT file's folder)."""
    if len(sys.argv) < 2:
        print("usage: render_phase1_uniformity.py <preprocessed_data.mat> [out_dir]")
        return
    mat_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(mat_path))
    print("saved:", render(mat_path, out_dir))


if __name__ == "__main__":
    main()
