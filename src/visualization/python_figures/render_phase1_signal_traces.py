"""Phase 1 signal traces for one prespecified cross-recording stratum.

The source is a role-specific cross-recording ``text/recordings.csv``, read and validated by _cross_dataset.read_role_table.
Every recording of the table gets a panel: the ten non-ictal recordings in the primary figure, or the two ictal recordings in the secondary-sensitivity figure.
The two roles never share a table or a figure.

Each panel shows three deterministic, evenly spaced retained channels of X_filtered over a 5 s window starting at 1 s.
Each channel is standardized separately for display, so amplitudes and vertical offsets are not comparable.
The traces are illustrative and are not load-bearing evidence.

Run:
    uv run src/visualization/python_figures/render_phase1_signal_traces.py <role Phase1/text> [out_dir]
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
import _cross_dataset as C  # noqa: E402

MODALITY_COLOR = {"SEEG": D.PALETTE["blue_main"], "LFP": D.PALETTE["teal"], "ECoG": D.PALETTE["red_strong"], "iEEG": D.PALETTE["violet"]}

WINDOW_START_S = 1.0
WINDOW_LEN_S = 5.0


def _mat_path(results_root, result_relpath):
    """Path of a recording's Phase 1 preprocessed_data.mat from its result_relpath, which may already name the file."""
    base = C.safe_result_path(results_root, result_relpath)
    if base.name == "preprocessed_data.mat":
        return base
    return base / "Phase1" / "mat" / "preprocessed_data.mat"


def _band_caption(h):
    """The band this recording actually carries, straight from its provenance."""
    if D.has(h, "filter_applied") and bool(D.read_scalar(h, "filter_applied")) and D.has(h, "filter_range"):
        b = D.read_vector(h, "filter_range")
        return f"{b[0]:g}–{b[1]:g} Hz (Phase 1)"
    if D.has(h, "native_band"):
        b = D.read_vector(h, "native_band")
        return f"{b[0]:g}–{b[1]:g} Hz (upstream)"
    if D.has(h, "filter_range"):  # no applied or native band recorded
        b = D.read_vector(h, "filter_range")
        return f"{b[0]:g}–{b[1]:g} Hz"
    return "band unknown"


def load_recording(path):
    """Read one recording's X_filtered, fs, band caption and channel labels, and return up to three standardized traces over the display window."""
    with h5py.File(path, "r") as h:
        X = D.read_matrix(h, "X_filtered")  # [channels x samples]
        fs = D.read_scalar(h, "fs")
        band = _band_caption(h)
        labels_all = D.read_str_cell(h, "channel_labels") if D.has(h, "channel_labels") else []
        channel_ids = D.read_vector(h, "channel_ids") if D.has(h, "channel_ids") else np.array([])

    n_ch, n_samples = X.shape
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    Xz = (X - X.mean(axis=1, keepdims=True)) / sd  # per-channel z-scores, one display scale for every modality

    i0 = int(round(WINDOW_START_S * fs))
    i1 = min(i0 + int(round(WINDOW_LEN_S * fs)), n_samples)
    t = (np.arange(i0, i1) - i0) / fs

    safe = tuple(np.unique(np.linspace(0, n_ch - 1, min(3, n_ch), dtype=int)))
    ids_are_valid = channel_ids.size == n_ch and np.all(np.isfinite(channel_ids))

    def channel_id_label(index):
        """Display label of a channel: its stored label with its ID, else 'ch <ID>' (the 1-based position when the IDs are unusable)."""
        value = channel_ids[index] if ids_are_valid else index + 1
        numeric = f"{int(value)}" if float(value).is_integer() else f"{value:g}"
        fallback = f"ch {numeric}"
        if index < len(labels_all) and labels_all[index]:
            return f"{labels_all[index]} (ID {numeric})"
        return fallback

    labels = [channel_id_label(i) for i in safe]
    return {"t": t, "traces": Xz[list(safe), i0:i1], "labels": labels, "band": band, "fs": fs, "n_channels": n_ch, "n_samples": n_samples, "display_scaling": "per_channel_standardized"}


def _plot_row(ax, rec, colour, header):
    """Draw one recording's traces, vertically offset, with their channel labels and a header line."""
    traces, t, labels = rec["traces"], rec["t"], rec["labels"]
    offsets = np.arange(traces.shape[0])[::-1] * 6.0
    for trace, label, y in zip(traces, labels, offsets):
        ax.plot(t, trace + y, color=colour, linewidth=0.8, alpha=0.95)
        ax.text(t[0] - 0.05, y, label, ha="right", va="center", fontsize=9, color="#3a3a3a")
    ax.set_xlim(t[0] - 0.35, t[-1])
    ax.set_ylim(offsets.min() - 4, offsets.max() + 4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Time  [s]", labelpad=2)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_title(f"{header}   |   {rec['band']}   |   fs = {rec['fs']:g} Hz", color=colour, fontsize=10, fontweight="bold", loc="right", pad=8)


def render(source, out_dir):
    """Draw the trace figure of one role's recordings.csv and return the saved paths; a missing recording raises FileNotFoundError."""
    table, role, _ = C.read_role_table(source, "recordings.csv", required_columns=("result_relpath", "modality", "recording", "state"))
    results_root = C.results_root_from_cross_source(source)
    row_indices = C.unique_record_rows(table)

    os.makedirs(out_dir, exist_ok=True)
    D.apply_publication_style(font_size=10, axes_lw=1.1)

    rows = []
    for index in row_indices:
        dataset = str(table["dataset"][index])
        path = _mat_path(results_root, table["result_relpath"][index])
        if not path.is_file():
            raise FileNotFoundError(f"required Phase-1 recording is missing for {dataset}: {path}")
        rec = load_recording(path)
        modality = str(table["modality"][index])
        recording = str(table["recording"][index])
        state = str(table["state"][index])
        rows.append({"dataset": dataset, "label": f"{modality} | {recording} | {state}", "colour": MODALITY_COLOR.get(modality, D.PALETTE["neutral"]), "rec": rec})

    ncols = 2
    nrows = int(np.ceil(len(rows) / ncols))
    fig = plt.figure(figsize=(14.0, 2.7 * nrows + 1.0))
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols, hspace=0.95, wspace=0.24, left=0.075, right=0.985, top=0.91, bottom=0.08)
    for index, row in enumerate(rows):
        _plot_row(fig.add_subplot(gs[index // ncols, index % ncols]), row["rec"], row["colour"], row["label"])
    for index in range(len(rows), nrows * ncols):
        fig.add_subplot(gs[index // ncols, index % ncols]).axis("off")

    fig.suptitle(f"Illustrative retained-channel traces — {C.ROLE_LABELS[role]}", fontsize=13)
    fig.text(
        0.01,
        0.008,
        "Each retained-channel trace is centered and standardized separately for display, then vertically offset. Amplitudes and offsets are not comparable across channels or recordings; channel IDs preserve the post-cleaning projection of the original channel indexing.",
        fontsize=7,
        color="#555555",
        wrap=True,
    )
    for row in rows:
        rec = row["rec"]
        print(f"  {row['dataset']:<52} fs={rec['fs']:>7.1f}  ch={rec['n_channels']:>4d}  n={rec['n_samples']:>8d}  band={rec['band']}")
    return D.finalize(fig, out_dir, C.role_stem("phase1_signal_traces", role))


def main() -> int:
    """Command-line entry point: render one role's Phase1/text folder into out_dir (default: the sibling images folder)."""
    if len(sys.argv) < 2:
        print("usage: render_phase1_signal_traces.py <role Phase1/text> [out_dir]")
        return 2
    source = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else D.phase_paths(source)["images_dir"]
    saved = render(source, out_dir)
    print(f"signal_traces: {len(saved)} files -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
