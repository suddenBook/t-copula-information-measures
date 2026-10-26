"""Cross-recording F2: the information differences of every recording, side by side.

  cross_f2_differences_vs_dimension_nats
      The median difference per measure against interaction order, one line per recording.
  cross_f2_differences_vs_dimension_pct
      The same as a percentage of the Gaussian value.
  cross_f2_setwise_vs_gaussian
      Every channel set of every recording: the difference against the Gaussian value of the same measure, with constant-percentage guides and the surrogate floor strip.

Measures are separated by panel.
In the first two figures colour carries the modality of the recording; in the third, which pools the recordings, colour carries the dimension.

The per-dimension medians come from a recording's `im_summary_by_d.csv` when that table is in long form (a `d` and a `measure` column with a median column) and are otherwise computed from the per-unit tables, over the units whose difference is finite.
The pooled scatter always reads the per-unit tables, with only the columns it draws.
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
from render_cross_shape import dash_index  # noqa: E402

plt = S.style()
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SUFFIX = {"nats": "_delta_nats", "pct": "_delta_pct"}


def difference_summary(recording):
    """(d, measure, median_nats, median_pct) for one recording, or None when it has no usable Phase 4 table."""
    phase4 = IO.recording_phase(recording, 4)
    summary = IO.read_table(phase4, "im_summary_by_d.csv", quiet=True)
    if summary is not None and {"d", "measure"} <= set(summary.columns):
        nats = IO.pick(summary, "median_delta_nats", "median_nats", "delta_nats_median")
        pct = IO.pick(summary, "median_delta_pct", "median_pct", "delta_pct_median")
        if nats is not None:
            return pd.DataFrame({"d": IO.numeric(summary["d"]), "measure": IO.text(summary["measure"]), "median_nats": IO.numeric(nats), "median_pct": IO.numeric(pct) if pct is not None else np.nan})
    wanted = {"d"} | {f"{m}{s}" for m in S.MEASURES for s in SUFFIX.values()}
    frames = [f for f in (IO.read_table(phase4, "im_pairs.csv", quiet=True, columns=wanted), IO.read_table(phase4, "im_long.csv", quiet=True, columns=wanted)) if f is not None]
    if not frames:
        return None
    table = pd.concat(frames, ignore_index=True)
    rows = []
    for measure in S.MEASURES:
        nats_column = f"{measure}_delta_nats"
        if nats_column not in table.columns:
            continue
        block = table[np.isfinite(IO.numeric(table[nats_column]))]
        if block.empty:
            continue
        dims, median = IO.by_dimension(block, nats_column, "median")
        pct_column = f"{measure}_delta_pct"
        _, pct = IO.by_dimension(block, pct_column, "median") if pct_column in block.columns else (dims, np.full(dims.size, np.nan))
        rows.append(pd.DataFrame({"d": dims.astype(float), "measure": measure, "median_nats": median, "median_pct": pct}))
    return pd.concat(rows, ignore_index=True) if rows else None


def setwise_table(recording):
    """The per-unit columns the pooled scatter draws, for one recording, or None."""
    phase4 = IO.recording_phase(recording, 4)
    wanted = {"d"} | {f"{m}_G" for m in S.MEASURES} | {f"{m}_delta_nats" for m in S.MEASURES} | {f"floor95_{m}" for m in S.MEASURES}
    frames = [f for f in (IO.read_table(phase4, "im_pairs.csv", quiet=True, columns=wanted), IO.read_table(phase4, "im_long.csv", quiet=True, columns=wanted)) if f is not None]
    if not frames:
        return None
    table = pd.concat(frames, ignore_index=True)
    table["rel_base"] = recording["rel_base"]
    table["modality"] = str(recording.get("modality", ""))
    return table


def _present(summaries, column="median_nats"):
    """The measures with a finite `column` value in at least one recording's summary, in `MEASURES` order."""
    measures = set()
    for frame in summaries.values():
        if frame is not None:
            measures |= {m for m in S.MEASURES if np.isfinite(IO.numeric(frame.loc[IO.text(frame["measure"]) == m, column])).any()}
    return [m for m in S.MEASURES if m in measures]


def _differences_vs_dimension(log, recordings, summaries, kind, label):
    """One panel per measure with one line per recording, in nats or in per cent (`kind`); None when no measure has a value."""
    column = "median_nats" if kind == "nats" else "median_pct"
    measures = _present(summaries, column)
    if not measures:
        print(f"  [skip] cross_f2_differences_vs_dimension_{kind}: no recording carries a {column} for any measure")
        return None
    fig, axes = plt.subplots(1, len(measures), figsize=(S.WIDTH, 2.8), layout="constrained", squeeze=False)
    axes = axes.ravel()
    all_dims = np.unique(np.concatenate([IO.numeric(f["d"]) for f in summaries.values() if f is not None]))
    dashes_by_recording = dash_index(recordings)
    for recording in recordings:
        frame = summaries.get(recording["rel_base"])
        if frame is None:
            continue
        modality = str(recording.get("modality", ""))
        index = dashes_by_recording[recording["rel_base"]]
        for panel, measure in enumerate(measures):
            block = frame[IO.text(frame["measure"]) == measure].sort_values("d")
            axes[panel].plot(IO.numeric(block["d"]), IO.numeric(block[column]), color=S.modality_color(modality), lw=1.0, ls=S.recording_linestyle(index), marker="o", ms=1.8, mfc="white", mew=0.5)
    for panel, measure in enumerate(measures):
        ax = axes[panel]
        if kind == "pct":
            ax.axhline(0.0, color=S.GREY, lw=0.9, zorder=0)
        S.dimension_axis(ax, all_dims)
        S.panel_label(ax, string.ascii_uppercase[panel], S.MEASURE_LABEL[measure])
        if panel == 0:
            ax.set_ylabel("median difference (nats)" if kind == "nats" else "median difference (per cent of Gaussian)", labelpad=2)
        if log:
            S.log_scale(ax, "y", np.concatenate([IO.numeric(f.loc[IO.text(f["measure"]) == measure, column]) for f in summaries.values() if f is not None]))
    handles = [Line2D([], [], color=S.modality_color(m), lw=1.4, label=m) for m in S.MODALITY_ORDER if any(r.get("modality") == m for r in recordings)]
    handles += [Line2D([], [], color=S.KEY, lw=1.0, ls=S.recording_linestyle(i), label=f"recording {i + 1} of a modality") for i in range(min(5, max(dashes_by_recording.values(), default=0) + 1))]
    S.legend_below(fig, handles, ncols=4, reserve=0.18)
    S.source_label(fig, label)
    return fig


def _setwise_vs_gaussian(log, pooled, label):
    """One scatter panel per measure over the pooled channel sets, coloured by dimension; None when no measure has a value."""
    measures = [m for m in S.MEASURES if f"{m}_G" in pooled.columns and f"{m}_delta_nats" in pooled.columns and np.isfinite(IO.numeric(pooled[f"{m}_delta_nats"])).any()]
    if not measures:
        return None
    dims = IO.numeric(pooled["d"])
    fig, axes = plt.subplots(1, len(measures), figsize=(S.WIDTH, 2.8), layout="constrained", squeeze=False)
    axes = axes.ravel()
    for index, measure in enumerate(measures):
        ax = axes[index]
        gaussian = IO.numeric(pooled[f"{measure}_G"])
        difference = IO.numeric(pooled[f"{measure}_delta_nats"])
        keep = np.isfinite(gaussian) & np.isfinite(difference)
        ax.scatter(gaussian[keep], difference[keep], s=1.2, lw=0, alpha=0.25, c=dims[keep], cmap=S.cmap_d, norm=S.norm_d, rasterized=True, zorder=2)
        if log:
            S.log_scale(ax, "x", gaussian[keep])
            S.log_scale(ax, "y", difference[keep])
        positive = gaussian[keep][gaussian[keep] > 0]
        if measure != "O" and positive.size:
            S.percent_guides(ax, float(positive.min()), float(positive.max()), annotate=bool(log))
        if measure == "O":
            ax.axvline(0.0, color=S.GREY, lw=0.6, zorder=1)
        floor_column = f"floor95_{measure}"
        if floor_column in pooled.columns:
            floors = IO.numeric(pooled[floor_column])
            floors = floors[np.isfinite(floors)]
            if floors.size:
                S.floor_strip(ax, float(floors.min()), float(floors.max()))
        ax.set_xlabel("Gaussian value (nats)", labelpad=2)
        S.panel_label(ax, string.ascii_uppercase[index], S.MEASURE_LABEL[measure])
        if index == 0:
            ax.set_ylabel("difference (nats)", labelpad=2)
    handles = [Line2D([], [], color=S.KEY, marker="o", ls="none", ms=2.6, label="one channel set, every recording"), Line2D([], [], color=S.GREY, lw=0.6, ls=(0, (4, 3)), label="constant percentage"), Patch(facecolor=S.GREY, alpha=0.20, label="surrogate floor over recordings")]
    S.legend_below(fig, handles, ncols=3, reserve=0.17)
    S.dimension_colorbar(fig, list(axes))
    S.source_label(fig, label)
    return fig


def render(recordings, output_base, label="cross-recording"):
    """Write the three F2 figures and their log twins under `<output_base>/Phase4/images`; return the paths."""
    summaries = {r["rel_base"]: difference_summary(r) for r in recordings}
    images = os.path.join(output_base, "Phase4", "images")
    paths = []
    if any(frame is not None for frame in summaries.values()):
        for kind in SUFFIX:
            paths += S.log_twin(_differences_vs_dimension, images, f"cross_f2_differences_vs_dimension_{kind}", recordings, summaries, kind, label)
    blocks = [t for t in (setwise_table(r) for r in recordings) if t is not None]
    if blocks:
        paths += S.log_twin(_setwise_vs_gaussian, images, "cross_f2_setwise_vs_gaussian", pd.concat(blocks, ignore_index=True), label)
    return paths
