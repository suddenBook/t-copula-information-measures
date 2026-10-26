"""Cross-recording F1: the fitted shape of every recording, side by side.

  cross_f1_shape_vs_dimension (Phase2/images)
      p_hat and nu_hat against interaction order, one line per recording, one panel per modality plus an all-recordings panel, with the envelope of the recordings' own Gaussian-surrogate intervals behind the p_hat row.
  cross_f1_overlap_vs_dimension (Phase3/images)
      How much of each recording's observed distribution of p_hat does its own surrogate realization cover, per dimension, beside the two yardsticks of that recording: two surrogate realizations, and two halves of the observed channel sets?

A recording carries its modality colour and its own dash pattern, so two recordings of one modality stay apart without a second colour meaning a second object.
The surrogate band is the one surrogate blue and every zero line is reference grey.

The surrogate band

Each recording writes the 2.5th and 97.5th percentiles of p_hat over its channel sets refitted on its own surrogate realization 1 (`sur_q025`, `sur_q975` in `Phase3/text/surrogate_comparison_by_d.csv`).
The band drawn here is the envelope of those recording-specific intervals (the minimum of the lower percentiles and the maximum of the upper ones, interpolated between dimensions), not a percentile of the recordings' surrogate fits pooled together.
The two differ: the envelope of a narrow and a wide interval is the wide one, while the pooled distribution sits between them, so the envelope is the wider, more conservative region.
The pooled distribution cannot be recovered from these summaries at all; it would need the raw draws.
Each recording's own interval is drawn as well in the modality panels, at a lower alpha, so the per-recording comparison the envelope hides stays on the figure.

The per-dimension medians come from each recording's `Phase2/text/summary_by_d.csv` when that table carries them and are computed from the unit tables when it does not, so a rollup never depends on one spelling of one statistic.
"""

from __future__ import annotations

import os
import string
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _phase_io as IO  # noqa: E402
import _style as S  # noqa: E402

plt = S.style()
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

MEDIAN_NAMES = {"p": ("median_p_hat", "p_hat_median", "median_p", "p50_p_hat"), "nu": ("median_nu_hat", "nu_hat_median", "median_nu", "p50_nu_hat")}
QUARTILE_NAMES = {("p", 25): ("q25_p_hat", "p_hat_q25", "q25_p"), ("p", 75): ("q75_p_hat", "p_hat_q75", "q75_p"), ("nu", 25): ("q25_nu_hat", "nu_hat_q25", "q25_nu"), ("nu", 75): ("q75_nu_hat", "nu_hat_q75", "q75_nu")}


def shape_summary(recording):
    """Per-dimension median and quartiles of p_hat and nu_hat for one recording, or None.

    The frame has the columns `d`, `p_median`, `p_q25`, `p_q75`, `nu_median`, `nu_q25` and `nu_q75`, read from the summary table or computed from the unit tables.
    When computed from the unit tables, the statistics cover the numerically successful units, and those of nu_hat cover its finite values only.
    """
    phase2 = IO.recording_phase(recording, 2)
    summary = IO.read_table(phase2, "summary_by_d.csv", quiet=True)
    if summary is not None and "d" in summary.columns:
        out = {"d": IO.numeric(summary["d"])}
        complete = True
        for quantity in ("p", "nu"):
            median = IO.pick(summary, *MEDIAN_NAMES[quantity])
            low = IO.pick(summary, *QUARTILE_NAMES[(quantity, 25)])
            high = IO.pick(summary, *QUARTILE_NAMES[(quantity, 75)])
            if median is None:
                complete = False
                break
            out[f"{quantity}_median"] = IO.numeric(median)
            out[f"{quantity}_q25"] = IO.numeric(low) if low is not None else np.nan
            out[f"{quantity}_q75"] = IO.numeric(high) if high is not None else np.nan
        if complete:
            return pd.DataFrame(out).sort_values("d")
    units = IO.units(phase2, quiet=True)
    if units is None:
        return None
    ok = IO.ok_units(units)
    ok = ok.assign(nu_finite=np.where(np.isfinite(IO.numeric(ok["nu_hat"])), IO.numeric(ok["nu_hat"]), np.nan))
    dims = IO.dimensions(units)
    frame = {"d": dims.astype(float)}
    for quantity, column in (("p", "p_hat"), ("nu", "nu_finite")):
        for name, q in (("median", 0.50), ("q25", 0.25), ("q75", 0.75)):
            _, values = IO.quantile_by_dimension(ok, column, q)
            frame[f"{quantity}_{name}"] = values
    return pd.DataFrame(frame)


def surrogate_band(recording):
    """(d, q025, q50, q975) of one recording's own surrogate p_hat, or None.

    These are that recording's percentiles over its own channel sets refitted on its surrogate realization 1, as its Phase 3 comparison wrote them.
    `_shape_vs_dimension` combines them across recordings into an envelope, never into a pooled distribution.
    """
    summary = IO.read_table(IO.recording_phase(recording, 3), "surrogate_comparison_by_d.csv", quiet=True)
    if summary is None or "d" not in summary.columns:
        return None
    low, middle, high = (IO.pick(summary, "sur_q025"), IO.pick(summary, "sur_q50"), IO.pick(summary, "sur_q975"))
    if low is None or high is None:
        return None
    return pd.DataFrame({"d": IO.numeric(summary["d"]), "q025": IO.numeric(low), "q50": IO.numeric(middle) if middle is not None else np.nan, "q975": IO.numeric(high)}).sort_values("d")


OVERLAP_COLUMNS = ("ovl_obs_sur", "mass_outside", "ovl_sur_sur2", "ovl_obs_halves")


def overlap_by_dimension(recording):
    """(d, ovl_obs_sur, mass_outside, ovl_sur_sur2, ovl_obs_halves) for one recording, or None when its Phase 3 comparison is absent or carries none of them."""
    summary = IO.read_table(IO.recording_phase(recording, 3), "surrogate_comparison_by_d.csv", quiet=True)
    if summary is None or "d" not in summary.columns:
        return None
    present = [c for c in OVERLAP_COLUMNS if c in summary.columns]
    if not present:
        return None
    frame = pd.DataFrame({"d": IO.numeric(summary["d"])})
    for column in present:
        frame[column] = IO.numeric(summary[column])
    return frame.sort_values("d")


def _panels(recordings):
    """One column per modality present, then the all-recordings column."""
    modalities = [m for m in S.MODALITY_ORDER if any(r.get("modality") == m for r in recordings)]
    extra = sorted({str(r.get("modality", "")) for r in recordings} - set(modalities) - {""})
    return modalities + extra + ["all recordings"]


def _members(recordings, column):
    """The recordings drawn in one panel column: all of them, or those of one modality."""
    return recordings if column == "all recordings" else [r for r in recordings if str(r.get("modality", "")) == column]


def dash_index(recordings):
    """Each recording's dash pattern index within its modality.

    The index is counted per modality rather than per panel, so one recording keeps one dash pattern in the modality panel and in the all-recordings panel, and two recordings that share a colour never share a dash.
    """
    index, seen = {}, {}
    for recording in recordings:
        modality = str(recording.get("modality", ""))
        index[recording["rel_base"]] = seen.get(modality, 0)
        seen[modality] = seen[modality] + 1 if modality in seen else 1
    return index


def _shape_vs_dimension(log, recordings, summaries, bands, label):
    """Median p_hat (top row) and nu_hat (bottom row) against dimension, one column per modality and one for all recordings."""
    columns = _panels(recordings)
    dashes_by_recording = dash_index(recordings)
    fig, axes = plt.subplots(2, len(columns), figsize=(S.WIDTH, 4.6), layout="constrained", squeeze=False, sharex=True)
    letters = list(string.ascii_uppercase)
    all_dims = np.unique(np.concatenate([IO.numeric(s["d"]) for s in summaries.values() if s is not None] or [np.array([2.0])]))
    for column_index, column in enumerate(columns):
        members = _members(recordings, column)
        band_rows = [bands[r["rel_base"]] for r in members if bands.get(r["rel_base"]) is not None]
        for row, quantity, name in ((0, "p", r"$\hat p$"), (1, "nu", r"$\hat\nu$")):
            ax = axes[row][column_index]
            if row == 0 and band_rows:
                # The envelope of the recordings' own 2.5 to 97.5 per cent intervals: the smallest lower percentile and the largest upper one at each dimension.
                # It is not a pooled quantile, and the legend does not claim one.
                grid = sorted({float(d) for frame in band_rows for d in IO.numeric(frame["d"])})
                low = [min(float(np.interp(d, IO.numeric(f["d"]), IO.numeric(f["q025"]))) for f in band_rows) for d in grid]
                high = [max(float(np.interp(d, IO.numeric(f["d"]), IO.numeric(f["q975"]))) for f in band_rows) for d in grid]
                ax.fill_between(grid, low, high, color=S.model_color("surrogate"), alpha=0.18, lw=0, zorder=0)
                if column != "all recordings":
                    # One recording at a time, so the modality panel still shows which recording the envelope's edges came from.
                    for frame in band_rows:
                        ax.fill_between(IO.numeric(frame["d"]), IO.numeric(frame["q025"]), IO.numeric(frame["q975"]), color=S.model_color("surrogate"), alpha=0.10, lw=0, zorder=0)
            for recording in members:
                frame = summaries.get(recording["rel_base"])
                if frame is None or f"{quantity}_median" not in frame:
                    continue
                colour = S.modality_color(recording.get("modality", ""))
                dashes = S.recording_linestyle(dashes_by_recording[recording["rel_base"]])
                dims = IO.numeric(frame["d"])
                ax.plot(dims, IO.numeric(frame[f"{quantity}_median"]), color=colour, lw=1.0, ls=dashes, zorder=2)
                if column != "all recordings" and f"{quantity}_q25" in frame:
                    ax.fill_between(dims, IO.numeric(frame[f"{quantity}_q25"]), IO.numeric(frame[f"{quantity}_q75"]), color=colour, alpha=0.10, lw=0, zorder=1)
            if row == 0:
                ax.axhline(0.0, color=S.GREY, lw=0.9, zorder=0)
            S.dimension_axis(ax, all_dims, label=(row == 1))
            S.panel_label(ax, letters[row * len(columns) + column_index], column if row == 0 else "")
            if column_index == 0:
                ax.set_ylabel(f"median {name}", labelpad=2)
            if log:
                S.log_scale(ax, "y", np.concatenate([IO.numeric(s[f"{quantity}_median"]) for s in summaries.values() if s is not None and f"{quantity}_median" in s]))
    handles = [Line2D([], [], color=S.modality_color(m), lw=1.4, label=m) for m in S.MODALITY_ORDER if any(r.get("modality") == m for r in recordings)]
    handles += [Line2D([], [], color=S.KEY, lw=1.0, ls=S.recording_linestyle(i), label=f"recording {i + 1} of a modality") for i in range(min(5, max(dashes_by_recording.values()) + 1))]
    handles += [
        Patch(facecolor=S.model_color("surrogate"), alpha=0.18, label="envelope of the recordings' surrogate 2.5 to 97.5 per cent intervals"),
        Patch(facecolor=S.model_color("surrogate"), alpha=0.10, label="one recording's own surrogate interval (modality panels)"),
        Patch(facecolor=S.GREY, alpha=0.10, label="interquartile range"),
    ]
    S.legend_below(fig, handles, ncols=3, reserve=0.20)
    S.source_label(fig, label)
    return fig


def _overlap_vs_dimension(recordings, overlaps, label):
    """One panel per overlap, one line per recording: the reported overlap of observed against surrogate, then each recording's two yardsticks on the same axis."""
    rows = {key: frame for key, frame in overlaps.items() if frame is not None}
    panels = [(column, title) for column, title in (("ovl_obs_sur", "observed against surrogate"), ("ovl_sur_sur2", "surrogate realization 1 against 2"), ("ovl_obs_halves", "observed half 1 against half 2")) if any(column in frame.columns for frame in rows.values())]
    if not panels:
        print("  [skip] cross_f1_overlap_vs_dimension: no recording carries an overlap")
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(S.WIDTH, 2.9), layout="constrained", squeeze=False, sharey=True)
    axes = axes.ravel()
    dashes_by_recording = dash_index(recordings)
    all_dims = np.concatenate([IO.numeric(f["d"]) for f in rows.values()])
    for index, (column, title) in enumerate(panels):
        ax = axes[index]
        for recording in recordings:
            frame = rows.get(recording["rel_base"])
            if frame is None or column not in frame.columns:
                continue
            modality = str(recording.get("modality", ""))
            ax.plot(IO.numeric(frame["d"]), IO.numeric(frame[column]), color=S.modality_color(modality), lw=1.0, ls=S.recording_linestyle(dashes_by_recording[recording["rel_base"]]), marker="o", ms=2.0, mfc="white", mew=0.6, label=S.recording_label(recording.get("dataset_key", recording["rel_base"])))
        ax.axhline(1.0, color=S.GREY, lw=0.8, zorder=0)
        S.dimension_axis(ax, all_dims)
        S.panel_label(ax, string.ascii_uppercase[index], title)
        if index == 0:
            ax.set_ylabel("overlap coefficient OVL", labelpad=2)
    handles = axes[0].get_legend_handles_labels()[0]
    handles.append(Line2D([], [], color="none", label="per recording and dimension; descriptive, no p-value"))
    S.legend_below(fig, handles, ncols=3, reserve=0.24)
    S.source_label(fig, label)
    return fig


def render(recordings, output_base, label="cross-recording"):
    """Write cross_f1_shape_vs_dimension with its log twin and cross_f1_overlap_vs_dimension; return the paths."""
    summaries = {r["rel_base"]: shape_summary(r) for r in recordings}
    bands = {r["rel_base"]: surrogate_band(r) for r in recordings}
    overlaps = {r["rel_base"]: overlap_by_dimension(r) for r in recordings}
    paths = []
    if any(frame is not None for frame in summaries.values()):
        paths += S.log_twin(_shape_vs_dimension, os.path.join(output_base, "Phase2", "images"), "cross_f1_shape_vs_dimension", recordings, summaries, bands, label)
    figure = _overlap_vs_dimension(recordings, overlaps, label)
    if figure is not None:
        paths += S.savefig_pair(figure, os.path.join(output_base, "Phase3", "images"), "cross_f1_overlap_vs_dimension")
    return paths
