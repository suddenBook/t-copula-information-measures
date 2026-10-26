"""Phase 2: what the fits cost and how they ended.

  phase2_status_by_dimension  How many channel sets ended in each status, in the precedence of tFitUnit, at every interaction order?
  phase2_fit_seconds          What does one fit cost, and how does that grow with d?
  phase2_repair_shift         How far does the eigenvalue-floor repair move the correlation matrix, and how often is the move gross?

The status counts are read from the phase's own `status_counts_by_d.csv` when it carries (d, status, count) and are otherwise counted from the unit tables, so the figure draws whichever of the two a run produced.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _phase_io as IO  # noqa: E402
import _style as S  # noqa: E402

plt = S.style()
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


def _counts_by_status(source, units):
    """A (dimension x status) count table, from the summary when it exists."""
    table = IO.read_table(source, "status_counts_by_d.csv", quiet=True)
    if table is not None and {"d", "status"} <= set(table.columns):
        for name in ("count", "n", "n_units"):
            if name in table.columns:
                return table.pivot_table(index="d", columns="status", values=name, aggfunc="sum", fill_value=0)
    return units.pivot_table(index="d", columns="status", values="unit_index", aggfunc="count", fill_value=0)


def _status_by_dimension(log, source, units, label):
    counts = _counts_by_status(source, units)
    if counts is None or counts.empty:
        return None
    dims = np.asarray(counts.index, dtype=float)
    order = [s for s in S.STATUS_ORDER if s in counts.columns]
    order += [s for s in counts.columns if s not in order]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(S.WIDTH, 2.8), layout="constrained")

    bottom = np.zeros(dims.size)
    total = counts[order].to_numpy(dtype=float).sum(axis=1)
    for status in order:
        height = counts[status].to_numpy(dtype=float)
        axA.bar(dims, height, bottom=bottom, width=0.8, color=S.status_color(status), lw=0.3, edgecolor="white", label=str(status).replace("_", " "))
        bottom = bottom + height
    axA.set_ylabel("channel sets", labelpad=2)
    S.dimension_axis(axA, dims)
    S.panel_label(axA, "A", "status counts")

    with np.errstate(divide="ignore", invalid="ignore"):
        share = counts[order].to_numpy(dtype=float) / total[:, None]
    for index, status in enumerate(order):
        if status == "ok":
            continue
        axB.plot(dims, share[:, index], color=S.status_color(status), lw=1.1, marker="o", ms=2.4, mfc="white", mew=0.7, label=str(status).replace("_", " "))
    axB.set_ylabel("fraction of channel sets", labelpad=2)
    S.dimension_axis(axB, dims)
    S.panel_label(axB, "B", "everything but ok")
    if log:
        S.log_scale(axA, "y", counts[order].to_numpy(dtype=float))
        S.log_scale(axB, "y", share[:, [i for i, s in enumerate(order) if s != "ok"]])

    handles = [Patch(facecolor=S.status_color(s), label=str(s).replace("_", " ")) for s in order]
    S.legend_below(fig, handles, ncols=min(len(handles), 5), reserve=0.14)
    S.source_label(fig, label)
    return fig


def _fit_seconds(log, units, label):
    if "fit_seconds" not in units.columns:
        print("  [skip] phase2_fit_seconds: no fit_seconds column")
        return None
    dims = IO.dimensions(units)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(S.WIDTH, 2.7), layout="constrained")

    d_med, median = IO.by_dimension(units, "fit_seconds", "median")
    _, low = IO.quantile_by_dimension(units, "fit_seconds", 0.25)
    _, high = IO.quantile_by_dimension(units, "fit_seconds", 0.75)
    axA.fill_between(d_med, low, high, color=S.GREY, alpha=0.20, lw=0, label="interquartile range")
    axA.plot(d_med, median, color=S.KEY, lw=1.0, zorder=2, label="median over channel sets")
    axA.scatter(d_med, median, s=9, zorder=3, c=[S.col_d(d) for d in d_med], lw=0)
    axA.set_ylabel("seconds per fit", labelpad=2)
    S.dimension_axis(axA, dims)
    S.panel_label(axA, "A", "cost of one fit")

    totals = units.groupby("d")["fit_seconds"].sum()
    d_total = np.asarray(totals.index, dtype=float)
    axB.bar(d_total, totals.to_numpy(dtype=float) / 60.0, width=0.8, color=[S.col_d(d) for d in d_total], label="all channel sets of this dimension")
    axB.set_ylabel("minutes", labelpad=2)
    S.dimension_axis(axB, dims)
    S.panel_label(axB, "B", "cost of the sweep")
    if log:
        S.log_scale(axA, "y", IO.numeric(units["fit_seconds"]))
        S.log_scale(axB, "y", totals.to_numpy(dtype=float) / 60.0)

    handles = [Line2D([], [], color=S.KEY, lw=1.0, label="median over channel sets"), Patch(facecolor=S.GREY, alpha=0.20, label="interquartile range"), Patch(facecolor=S.col_d(11), label="one bar per dimension, coloured by $d$")]
    S.legend_below(fig, handles, ncols=3, reserve=0.15)
    S.source_label(fig, label)
    return fig


def _repair_shift(log, units, label):
    if "repair_shift_rms" not in units.columns:
        print("  [skip] phase2_repair_shift: no repair_shift_rms column")
        return None
    dims = IO.dimensions(units)
    repaired = units[IO.numeric(units["repair_shift_rms"]) > 0]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(S.WIDTH, 2.7), layout="constrained")

    groups, positions, colors = [], [], []
    for d in dims:
        values = IO.numeric(repaired.loc[repaired["d"] == d, "repair_shift_rms"])
        if values.size:
            groups.append(values)
            positions.append(d)
            colors.append(S.col_d(d))
    S.violin(axA, groups, positions, colors)
    if "repair_gross" in units.columns:
        gross = IO.numeric(units.loc[IO.flag(units["repair_gross"]), "repair_shift_rms"])
        if gross.size:
            axA.axhline(float(np.min(gross)), color=S.GREY, lw=0.9, ls=(0, (3, 2)), label="smallest shift flagged gross")
    axA.set_ylabel("repair shift, RMS", labelpad=2)
    S.dimension_axis(axA, dims)
    S.panel_label(axA, "A", "size of the projection")

    for column, dashes, name in (("repair_applied", "-", "repair applied"), ("repair_gross", (0, (3, 2)), "repair gross")):
        if column not in units.columns:
            continue
        d_frac, fraction = IO.fraction_by_dimension(units, IO.flag(units[column]))
        axB.plot(d_frac, fraction, color=S.KEY, lw=1.0, ls=dashes, marker="o", ms=2.2, mfc="white", mew=0.7, label=name)
    axB.set_ylabel("fraction of channel sets", labelpad=2)
    S.dimension_axis(axB, dims)
    S.panel_label(axB, "B", "how often it fires")
    if log:
        S.log_scale(axA, "y", IO.numeric(repaired["repair_shift_rms"]))
        S.log_scale(axB, "y", np.concatenate([IO.fraction_by_dimension(units, IO.flag(units[c]))[1] for c in ("repair_applied", "repair_gross") if c in units.columns]))

    handles = [Patch(facecolor="none", edgecolor=S.KEY, lw=0.7, label="distribution over repaired sets"), Line2D([], [], color=S.KEY, marker="_", ls="none", ms=6, mew=1.2, label="median")]
    handles += axA.get_legend_handles_labels()[0] + axB.get_legend_handles_labels()[0]
    S.legend_below(fig, handles, ncols=3, reserve=0.16)
    S.source_label(fig, label)
    return fig


def render(source, out_dir):
    units = IO.units(source)
    if units is None:
        return []
    label = IO.phase_paths(source)["label"]
    paths = []
    paths += S.log_twin(_status_by_dimension, out_dir, "phase2_status_by_dimension", source, units, label)
    paths += S.log_twin(_fit_seconds, out_dir, "phase2_fit_seconds", units, label)
    paths += S.log_twin(_repair_shift, out_dir, "phase2_repair_shift", units, label)
    return paths
