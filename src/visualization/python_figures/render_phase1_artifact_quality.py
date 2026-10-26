"""Phase 1 QC figures: artifact detection and channel quality.

Three figures, one purpose each, all read straight from the persisted structs:

  1. phase1_artifact_channel_quality -- per-channel temporal artifact load, with the policy removals and the retained diagnostic candidates marked.
  2. phase1_artifact_channel_flags   -- the boolean diagnostic-flag matrix (low power, abnormal kurtosis, isolated, redundant, temporal rate, powerline) per channel.
  3. phase1_artifact_time_profile    -- [channels x time-window] heat map of the artifact-sample fraction, which shows when the artifacts occur.

Figures 1 and 3 read `artifact_info.time_profile`, the per-window artifact fraction that Phase 1 stores (computeArtifactTimeProfile.m) because the per-sample `final_artifact_mask` is too large to persist.
A file without time_profile gets no figure 3, and figure 1 then plots `channel_artifact_pct` with the caveat below stated on its axis.

Figure 1 does not plot `channel_artifact_pct` directly.
detectSEEGArtifacts first fills that field with the true fraction of artifact samples; its section 8b then overwrites it with a flat `100` sentinel for every channel that carries a channel-level diagnostic flag (low power, abnormal kurtosis, isolated, redundant).
A 100 there is therefore not evidence that every sample is an artifact, and it is not an automatic removal decision.
Figure 1 reconstructs the true temporal load from `time_profile`, marks the sentinel channels separately and shows the reason-specific cleaning decision.

The reconstruction is exact and can be checked independently: weighting each window by the samples it holds reproduces `sum(mask)/n_samples`, and the resulting load satisfies `flag_temporal_rate == (load > temporal_rate_threshold_pct)`, a flag detectSEEGArtifacts computes from the same mask before the sentinel overwrite.

All channel indices here are the original, pre-cleaning ones (the channel set of X_raw), matching `channel_artifact_pct`, `excluded_channels` and `time_profile`.

Run:
    uv run src/visualization/python_figures/render_phase1_artifact_quality.py <preprocessed_data.mat> [out_dir]
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

FLAG_FIELDS = [("flag_low_power", "low power"), ("flag_abnormal_kurt", "abnormal kurtosis"), ("flag_isolated", "isolated"), ("flag_redundant", "redundant"), ("flag_temporal_rate", "temporal rate")]


def _fig_channel_quality(load_pct, is_true_load, sentinel, excluded, candidates, name, out_dir):
    """Bar chart of the temporal artifact load per channel, coloured by the cleaning decision."""
    n = load_pct.size
    ch = np.arange(1, n + 1)
    retained_candidates = np.setdiff1d(candidates, excluded)

    colors = np.array([D.PALETTE["blue_main"]] * n, dtype=object)
    colors[retained_candidates] = "#E08A1E"
    colors[excluded] = D.PALETTE["red_strong"]

    fig, ax = plt.subplots(figsize=(max(9.0, n * 0.06), 4.8))
    ax.bar(ch, load_pct, color=list(colors), width=0.85)
    handles = []
    handles.append(plt.Line2D([], [], color=D.PALETTE["blue_main"], lw=6, label="retained"))
    if retained_candidates.size:
        handles.append(plt.Line2D([], [], color="#E08A1E", lw=6, label=(f"detector candidate retained ({retained_candidates.size})")))
    if excluded.size:
        handles.append(plt.Line2D([], [], color=D.PALETTE["red_strong"], lw=6, label=f"removed ({excluded.size})"))

    ymax = max(1.0, float(np.nanmax(load_pct)))
    if is_true_load and sentinel.size:
        ax.plot(ch[sentinel], np.full(sentinel.size, ymax * 1.06), marker="v", ls="none", ms=4, color=D.PALETTE["violet"])
        handles.append(plt.Line2D([], [], marker="v", ls="none", ms=5, color=D.PALETTE["violet"], label=f"channel_artifact_pct = 100 sentinel ({sentinel.size})"))
        ax.set_ylim(0, ymax * 1.14)

    ax.set_xlabel("Channel (original indexing)")
    ax.set_ylabel("Artifact samples  [%]" if is_true_load else "channel_artifact_pct  [%]  (sentinel-contaminated)")
    ax.set_title(f"Per-channel temporal artifact load — {name}", fontsize=12)
    ax.set_xlim(0.5, n + 0.5)
    ax.legend(handles=handles, fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, axis="y", alpha=0.25)
    caption = "Frozen auto mode removes only channels carrying the abnormal-kurtosis flag; low-power, isolation, redundancy, temporal-rate, and powerline findings remain audit diagnostics and do not delete samples."
    if is_true_load:
        caption += "  Bars are the TRUE temporal load, reconstructed from artifact_info.time_profile; ▼ marks a legacy channel_artifact_pct=100 diagnostic sentinel, not 100% temporal contamination or an automatic removal rule."
    else:
        caption += "  This legacy artifact lacks time_profile, so bar heights use the sentinel-inclusive compatibility field and are not interpretable as temporal percentages for sentinel-marked channels."
    fig.text(0.01, 0.005, caption, fontsize=7, color="#555555", wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return D.finalize(fig, out_dir, "phase1_artifact_channel_quality")


def _fig_channel_flags(flags, labels, excluded, name, out_dir):
    """Heat map of the boolean diagnostic flags per channel, with the removed channels marked by red lines."""
    n = flags.shape[1]
    fig, ax = plt.subplots(figsize=(max(9.0, n * 0.06), 0.5 * len(labels) + 2.4))
    ax.imshow(flags, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=1, extent=(0.5, n + 0.5, len(labels) - 0.5, -0.5))
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Channel (original indexing)")
    counts = flags.sum(axis=1).astype(int)
    for i, c in enumerate(counts):
        ax.text(n + 0.5, i, f"  n={c}", va="center", fontsize=8, color="#555555")
    for e in excluded:
        ax.axvline(e + 1, color=D.PALETTE["red_strong"], lw=0.7, alpha=0.55)
    ax.set_title(f"Channel diagnostic flags — {name}" + ("  (red lines = policy removals)" if excluded.size else ""), fontsize=12)
    fig.tight_layout()
    return D.finalize(fig, out_dir, "phase1_artifact_channel_flags")


def _fig_time_profile(frac, centers, excluded, candidates, name, out_dir):
    """[channels x window] artifact occupancy.

    Channel markers sit in a gutter outside the heat map, so they do not cover the rows they mark.
    """
    n_ch, n_win = frac.shape
    retained_candidates = np.setdiff1d(candidates, excluded)
    fig, ax = plt.subplots(figsize=(10.0, max(3.6, n_ch * 0.045)))
    t0, t1 = (centers[0], centers[-1]) if centers.size else (0.0, float(n_win))
    im = ax.imshow(frac, aspect="auto", interpolation="nearest", cmap="magma_r", vmin=0, vmax=max(1e-6, float(frac.max())), extent=(t0, t1, n_ch + 0.5, 0.5))
    gutter = t0 - 0.012 * (t1 - t0)
    for idx, colour, lbl in ((excluded, D.PALETTE["red_strong"], "removed by frozen policy"), (retained_candidates, "#E08A1E", "detector candidate retained")):
        if idx.size:
            ax.plot(np.full(idx.size, gutter), idx + 1, marker="_", ls="none", ms=7, mew=2, color=colour, clip_on=False, label=lbl)
    ax.set_xlim(t0, t1)
    if excluded.size or retained_candidates.size:
        ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label("artifact sample fraction in window")
    ax.set_xlabel("Time  [s]")
    ax.set_ylabel("Channel (original indexing)")
    ax.set_title(f"Artifact occupancy over time — {name}", fontsize=12)
    fig.tight_layout()
    return D.finalize(fig, out_dir, "phase1_artifact_time_profile")


def render(mat_path, out_dir):
    """Draw the artifact figures of one preprocessed_data.mat and return the saved paths ([] when artifact_info or channel_artifact_pct is absent)."""
    os.makedirs(out_dir, exist_ok=True)
    D.apply_publication_style()
    name = D.recording_name(mat_path)
    saved = []

    with h5py.File(mat_path, "r") as h:
        if "artifact_info" not in h:
            print("artifact_quality: artifact_info absent, skipping")
            return []
        ai = h["artifact_info"]
        if "channel_artifact_pct" not in ai:
            print("artifact_quality: channel_artifact_pct absent, skipping")
            return []
        pct = np.asarray(ai["channel_artifact_pct"], dtype=float).ravel()
        n = pct.size

        in_range = lambda v: v[(v >= 0) & (v < n)]  # noqa: E731
        excluded = in_range(D.channel_indices(h, "clean_info/excluded_channels"))
        candidates = in_range(D.channel_indices(h, "clean_info/auto_detected_severe_channels"))

        rows, labels = [], []
        for field, label in FLAG_FIELDS:
            if field in ai:
                v = np.asarray(ai[field], dtype=float).ravel()
                if v.size == n:
                    rows.append((v > 0).astype(float))
                    labels.append(label)
        pl = in_range(D.channel_indices(h, "artifact_info/powerline_channels"))
        if pl.size:
            row = np.zeros(n)
            row[pl] = 1.0
            rows.append(row)
            labels.append("powerline")

        # The true temporal load, reconstructed from time_profile by _phase_data: the value of channel_artifact_pct before section 8b writes the sentinel.
        profile = D.artifact_time_profile(h)
        load_pct, is_true_load = D.true_temporal_load_pct(h)
        sentinel = D.sentinel_channels(h)

    saved += _fig_channel_quality(load_pct, is_true_load, sentinel, excluded, candidates, name, out_dir)
    if rows:
        saved += _fig_channel_flags(np.vstack(rows), labels, excluded, name, out_dir)
    if profile is not None:
        saved += _fig_time_profile(profile["fraction"], profile["centers"], excluded, candidates, name, out_dir)
    else:
        print("artifact_quality: artifact_info.time_profile absent (mat predates it); skipping the time-profile figure")
    return saved


def main() -> None:
    """Command-line entry point: render one MAT file into out_dir (default: the MAT file's folder)."""
    if len(sys.argv) < 2:
        print("usage: render_phase1_artifact_quality.py <preprocessed_data.mat> [out_dir]")
        return
    mat_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(mat_path))
    print("saved:", render(mat_path, out_dir))


if __name__ == "__main__":
    main()
