"""Phase 1 QC traces for policy removals and retained detector candidates.

Two figure families, one purpose each:

  1. phase1_excluded_channels_p<k>    -- the channels actually removed (clean_info.excluded_channels).
  2. phase1_severe_but_retained_p<k>  -- the detector candidates that cleaning retains (auto_detected_severe_channels minus excluded_channels).

`auto_detected_severe_channels` is the candidate list: every channel whose (sentinel-contaminated) channel_artifact_pct exceeds the severe threshold.
It is not the list of removed channels.
In auto/batch mode the policy removes only the channels flagged for abnormal kurtosis, and the low-power, isolation, redundancy, temporal-rate and powerline findings are kept as audit diagnostics.
The two sets therefore answer different questions and carry different titles, and the removed family reads `excluded_channels`.

In MATLAB, visualizeExcludedChannels.m draws the candidate traces only in interactive mode, behind the promptArtifactChannelSelection prompt; these figures draw them for batch runs.

Artifact percentages are the true temporal load reconstructed from artifact_info.time_profile, not `channel_artifact_pct`, which carries a flat 100 sentinel for every channel-level diagnostic candidate.
Channels with the sentinel are marked, because it is neither a temporal percentage nor an automatic removal decision.

Fields read:
  X_raw                                    -- pre-clean raw matrix [samples x channels]
  clean_info.excluded_channels             -- 1-based indices actually removed (may be empty)
  clean_info.auto_detected_severe_channels -- 1-based candidate indices (may be empty)
  artifact_info.time_profile               -- for the true temporal load
  artifact_info.channel_artifact_pct       -- for the 100-sentinel marking
  sidecar.flagged_channels/.reasons        -- reason string per removed channel
  sidecar.auto_detected_channels/.auto_detected_reasons
                                           -- reason string per candidate channel

Each page holds at most 8 channels and is saved as its own figure.
A family with no channels produces nothing, as in the MATLAB function.

Run:
    uv run src/visualization/python_figures/render_phase1_excluded_channels.py <preprocessed_data.mat> [out_dir]
"""

# /// script
# requires-python = ">=3.10"
# dependencies = ["h5py>=3.10", "matplotlib>=3.8", "numpy>=1.26"]
# ///
from __future__ import annotations

import json
import os
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _phase_data as D  # noqa: E402

MAX_PER_FIGURE = 8  # same pagination as visualizeExcludedChannels.m

RETAINED_COLOR = "#E08A1E"  # same amber render_phase1_artifact_quality.py uses


def _default_sidecar(mat_path):
    """Path of the run's artifact-cleaning sidecar (artifact_clean_decision.json): the copy in PhaseN/postprocess/ when that folder holds one, else the copy Phase 1 writes to PhaseN/text/."""
    mat_dir = os.path.dirname(os.path.abspath(mat_path))
    phase_dir = os.path.dirname(mat_dir)
    name = "artifact_clean_decision.json"
    corrected = os.path.join(phase_dir, "postprocess", name)
    if os.path.exists(corrected):
        return corrected
    return os.path.join(phase_dir, "text", name)


def _load_reasons(sidecar_path):
    """Map 1-based channel index -> reason string from the JSON sidecar; a missing or unreadable file gives an empty map.

    Both channel families drawn below need a reason string, so the mapping is keyed by channel rather than by position in any one list.
    The sidecar carries two reason arrays because the removed set and the candidate list are different sets: `reasons` is paired with `flagged_channels` and `auto_detected_reasons` with `auto_detected_channels`.
    A sidecar without `auto_detected_reasons` holds a single `reasons` array, which is paired with whichever channel list matches it in length.
    """
    mapping = {}
    if not sidecar_path or not os.path.exists(sidecar_path):
        return mapping
    try:
        with open(sidecar_path) as f:
            payload = json.load(f)
    except Exception:
        return mapping

    def _pair(channels_key, reasons_key):
        """Add channel -> reason pairs from two sidecar arrays when their lengths match."""
        channels = np.atleast_1d(payload.get(channels_key, [])).ravel()
        reasons = np.atleast_1d(payload.get(reasons_key, [])).ravel()
        if channels.size != reasons.size:
            return
        for ch, rs in zip(channels, reasons):
            mapping[int(ch)] = str(rs)

    if "auto_detected_reasons" in payload:
        _pair("auto_detected_channels", "auto_detected_reasons")
        _pair("flagged_channels", "reasons")
    else:
        # Single-array sidecar: pair the array with whichever list it matches in length.
        # The removed channels are a subset of the candidates, so when both lengths match the two pairings agree.
        _pair("auto_detected_channels", "reasons")
        _pair("flagged_channels", "reasons")
    return mapping


def _draw_family(channels, X_raw, fs, load_pct, is_true_load, sentinel, reasons, color, stem, suptitle, caption, out_dir):
    """Paginated red or amber raw traces for one channel set, titled with load, sentinel mark and reason; returns the saved paths."""
    if channels.size == 0:
        return []
    t = np.arange(X_raw.shape[1]) / fs
    n_pages = int(np.ceil(channels.size / MAX_PER_FIGURE))
    saved = []
    for page in range(n_pages):
        lo = page * MAX_PER_FIGURE
        hi = min(lo + MAX_PER_FIGURE, channels.size)
        n_rows = hi - lo
        fig, axes = plt.subplots(n_rows, 1, figsize=(12.0, 1.8 * n_rows + 0.9), squeeze=False, sharex=True)
        axes = axes.ravel()
        for i in range(n_rows):
            ch0 = int(channels[lo + i])  # 0-based
            ch1 = ch0 + 1  # 1-based, as MATLAB and the sidecar index
            ax = axes[i]
            ax.plot(t, X_raw[ch0], color=color, lw=0.7)
            ax.set_ylabel(f"Ch {ch1}", fontsize=9)
            title = f"Channel {ch1}"
            if ch0 < load_pct.size:
                label = "temporal artifact load" if is_true_load else "legacy channel_artifact_pct"
                title += f" — {label} {load_pct[ch0]:.1f}%"
            if ch0 in sentinel:
                title += "  [pct=100 sentinel]"
            if ch1 in reasons and reasons[ch1]:
                title += f"  |  {reasons[ch1]}"
            ax.set_title(title, fontsize=10)
            ax.margins(x=0)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.999)
        fig.text(0.01, 0.004, caption, fontsize=7, color="#555555", wrap=True)
        fig.tight_layout(rect=(0, 0.04, 1, 0.98))
        saved += D.finalize(fig, out_dir, f"{stem}_p{page + 1}")
    return saved


def render(mat_path, out_dir, sidecar_path=None):
    """Draw both trace families of one preprocessed_data.mat and return the saved paths; sidecar_path defaults to _default_sidecar(mat_path)."""
    os.makedirs(out_dir, exist_ok=True)
    D.apply_publication_style()
    if sidecar_path is None:
        sidecar_path = _default_sidecar(mat_path)
    reasons = _load_reasons(sidecar_path)

    with h5py.File(mat_path, "r") as h:
        excluded = D.channel_indices(h, "clean_info/excluded_channels")
        candidates = D.channel_indices(h, "clean_info/auto_detected_severe_channels")
        if excluded.size == 0 and candidates.size == 0:
            return []
        X_raw = D.read_matrix(h, "X_raw")
        fs = D.read_scalar(h, "fs")
        if "artifact_info/channel_artifact_pct" in h:
            load_pct, is_true_load = D.true_temporal_load_pct(h)
            sentinel = set(D.sentinel_channels(h).tolist())
        else:
            load_pct, is_true_load, sentinel = np.array([]), False, set()

    n_ch = X_raw.shape[0]
    in_range = lambda v: v[(v >= 0) & (v < n_ch)]  # noqa: E731  (MATLAB valid_mask analogue)
    excluded = in_range(excluded)
    retained_candidates = in_range(np.setdiff1d(candidates, excluded))

    load_note = (
        "Percentages are the TRUE temporal load reconstructed from artifact_info.time_profile; channels marked [pct=100 sentinel] carry the legacy diagnostic sentinel, not 100% temporal contamination or an automatic removal verdict."
        if is_true_load
        else "Percentages are the sentinel-contaminated channel_artifact_pct (this artifact predates artifact_info.time_profile and those values are not temporal percentages for sentinel-marked channels)."
    )

    saved = _draw_family(
        excluded,
        X_raw,
        fs,
        load_pct,
        is_true_load,
        sentinel,
        reasons,
        D.PALETTE["red_strong"],
        "phase1_excluded_channels",
        "Channels removed by the frozen abnormal-kurtosis-only policy",
        "Removal records the prespecified policy decision; these traces are descriptive QC, not retrospective proof that a channel is physiologically bad. No samples are deleted. " + load_note,
        out_dir,
    )

    saved += _draw_family(
        retained_candidates,
        X_raw,
        fs,
        load_pct,
        is_true_load,
        sentinel,
        reasons,
        RETAINED_COLOR,
        "phase1_severe_but_retained",
        "Detector-flagged candidates retained by the frozen policy",
        "These channels enter the analysis. Their diagnostic flags do not imply a severity verdict or sample deletion; frozen auto mode removes only abnormal-kurtosis-flagged channels. " + load_note,
        out_dir,
    )
    return saved


def main() -> None:
    """Command-line entry point: render one MAT file into out_dir (default: the MAT file's folder)."""
    if len(sys.argv) < 2:
        print("usage: render_phase1_excluded_channels.py <preprocessed_data.mat> [out_dir]")
        return
    mat_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(mat_path))
    print("saved:", render(mat_path, out_dir))


if __name__ == "__main__":
    main()
