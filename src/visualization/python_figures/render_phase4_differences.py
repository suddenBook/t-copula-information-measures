"""Phase 4, how far the fitted t moves each information measure away from the Gaussian.

  phase4_differences_vs_dimension_nats  The median difference and its interquartile range against interaction order, in nats, with the surrogate floor.
  phase4_differences_vs_dimension_pct   The same as a percentage of the Gaussian value the difference is referenced to.
  phase4_setwise_vs_gaussian            One point per channel set: the difference against the Gaussian value of the same measure, with lines of constant percentage and the surrogate floor.
  phase4_pct_by_decile                  The percentage by decile of the Gaussian denominator, so a large percentage on a tiny denominator cannot pass for a large difference.

Measures are separated by panel and never by colour; inside a panel the dimension is the only colour.
Two figures read the percentage with its denominator: phase4_setwise_vs_gaussian plots each difference against its Gaussian value with lines of constant percentage, and phase4_pct_by_decile bins the percentage by decile of the Gaussian value.

The population:
Every per-unit measure is drawn over the units with `numerical_ok`, the same population Phase 4 computes its own summaries over.
A unit whose fit failed enters the measure function as p = NaN and gets NaN measures, but the population is defined by the flag, not by `isfinite`, so filtering on the flag keeps the figures in step with Phase 4 whatever the fail-closed path returns.
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


def im_units(source):
    """`im_pairs.csv` (d = 2) and `im_long.csv` (d >= 3) as one per-unit table."""
    frames = [f for f in (IO.read_table(source, "im_pairs.csv"), IO.read_table(source, "im_long.csv")) if f is not None]
    if not frames:
        return None
    table = pd.concat(frames, ignore_index=True)
    table.attrs["name"] = "im_pairs.csv + im_long.csv"
    return table


def _present_measures(table, suffix="_delta_nats"):
    """The declared measures whose column `<measure><suffix>` exists and holds a finite value."""
    return [m for m in S.MEASURES if f"{m}{suffix}" in table.columns and np.isfinite(IO.numeric(table[f"{m}{suffix}"])).any()]


def _floor_band(table, measure):
    """The (lowest, highest) joined surrogate floor floor95_<measure> over the units, or None when the column is absent or has no finite value."""
    column = f"floor95_{measure}"
    if column not in table.columns:
        return None
    values = IO.numeric(table[column])
    values = values[np.isfinite(values)]
    return (float(values.min()), float(values.max())) if values.size else None


def _differences_vs_dimension(log, table, kind, label):
    """One panel per measure: the median difference and its interquartile range against d, in nats or in per cent of the Gaussian value."""
    suffix = "_delta_nats" if kind == "nats" else "_delta_pct"
    measures = _present_measures(table, suffix)
    if not measures:
        print(f"  [skip] phase4_differences_vs_dimension_{kind}: no {suffix} column carries data")
        return None
    dims_all = IO.dimensions(table)
    fig, axes = plt.subplots(1, len(measures), figsize=(S.WIDTH, 2.7), layout="constrained", squeeze=False)
    axes = axes.ravel()
    for index, measure in enumerate(measures):
        ax = axes[index]
        column = f"{measure}{suffix}"
        block = table[np.isfinite(IO.numeric(table[column]))]
        dims, median = IO.by_dimension(block, column, "median")
        _, q25 = IO.quantile_by_dimension(block, column, 0.25)
        _, q75 = IO.quantile_by_dimension(block, column, 0.75)
        ax.fill_between(dims, q25, q75, color=S.GREY, alpha=0.20, lw=0)
        ax.plot(dims, median, color=S.KEY, lw=1.0, zorder=2)
        ax.scatter(dims, median, s=9, c=[S.col_d(d) for d in dims], lw=0, zorder=3)
        if kind == "nats":
            band = _floor_band(table, measure)
            if band:
                S.floor_strip(ax, *band)
        else:
            ax.axhline(0.0, color=S.GREY, lw=0.9, zorder=0)
        S.dimension_axis(ax, dims_all)
        S.panel_label(ax, string.ascii_uppercase[index], S.MEASURE_LABEL[measure])
        if index == 0:
            ax.set_ylabel("difference (nats)" if kind == "nats" else "difference (per cent of Gaussian)", labelpad=2)
        if log:
            S.log_scale(ax, "y", IO.numeric(block[column]))
    handles = [Line2D([], [], color=S.KEY, lw=1.0, label="median over the successfully fitted channel sets"), Patch(facecolor=S.GREY, alpha=0.20, label="interquartile range"), Line2D([], [], color=S.col_d(11), marker="o", ls="none", ms=3, label="marker coloured by $d$")]
    if kind == "nats":
        handles.append(Patch(facecolor=S.GREY, alpha=0.20, label="surrogate floor"))
    else:
        handles.append(Line2D([], [], color=S.GREY, lw=0.9, label="zero"))
    S.legend_below(fig, handles, ncols=4, reserve=0.17)
    S.source_label(fig, label)
    return fig


def _setwise_vs_gaussian(log, table, label):
    """One panel per measure: every channel set's difference against its Gaussian value, coloured by d."""
    measures = _present_measures(table)
    if not measures:
        return None
    dims = IO.numeric(table["d"])
    fig, axes = plt.subplots(1, len(measures), figsize=(S.WIDTH, 2.7), layout="constrained", squeeze=False)
    axes = axes.ravel()
    for index, measure in enumerate(measures):
        ax = axes[index]
        gaussian = IO.numeric(table[f"{measure}_G"]) if f"{measure}_G" in table.columns else None
        difference = IO.numeric(table[f"{measure}_delta_nats"])
        if gaussian is None:
            print(f"  [skip] phase4_setwise_vs_gaussian: no {measure}_G column")
            continue
        keep = np.isfinite(gaussian) & np.isfinite(difference)
        ax.scatter(gaussian[keep], difference[keep], s=1.6, lw=0, alpha=0.30, c=dims[keep], cmap=S.cmap_d, norm=S.norm_d, rasterized=True, zorder=2)
        if log:
            S.log_scale(ax, "x", gaussian[keep])
            S.log_scale(ax, "y", difference[keep])
        positive = gaussian[keep][gaussian[keep] > 0]
        if measure != "O" and positive.size:
            S.percent_guides(ax, float(positive.min()), float(positive.max()), annotate=bool(log))
        if measure == "O":
            ax.axvline(0.0, color=S.GREY, lw=0.6, zorder=1)
        band = _floor_band(table, measure)
        if band:
            S.floor_strip(ax, *band)
        ax.set_xlabel("Gaussian value (nats)", labelpad=2)
        S.panel_label(ax, string.ascii_uppercase[index], S.MEASURE_LABEL[measure])
        if index == 0:
            ax.set_ylabel("difference (nats)", labelpad=2)
    handles = [Line2D([], [], color=S.KEY, marker="o", ls="none", ms=2.6, label="one successfully fitted channel set"), Line2D([], [], color=S.GREY, lw=0.6, ls=(0, (4, 3)), label="constant percentage"), Patch(facecolor=S.GREY, alpha=0.20, label="surrogate floor")]
    S.legend_below(fig, handles, ncols=3, reserve=0.17)
    S.dimension_colorbar(fig, list(axes))
    S.source_label(fig, label)
    return fig


def _pct_by_decile(log, deciles, label):
    """One panel per measure: the median percentage by decile of the Gaussian value, one line per d, with the mean-before-ratio percentage pooled over d."""
    if deciles is None:
        return None
    IO.require(deciles, "d", "measure", "decile")
    measures = [m for m in S.MEASURES if (IO.text(deciles["measure"]) == m).any()]
    if not measures:
        return None
    fig, axes = plt.subplots(1, len(measures), figsize=(S.WIDTH, 2.7), layout="constrained", squeeze=False)
    axes = axes.ravel()
    for index, measure in enumerate(measures):
        ax = axes[index]
        block = deciles[IO.text(deciles["measure"]) == measure]
        for d, chunk in block.groupby("d"):
            chunk = chunk.sort_values("decile")
            values = IO.pick(chunk, "median_pct")
            if values is None:
                continue
            ax.plot(IO.numeric(chunk["decile"]), IO.numeric(values), color=S.col_d(d), lw=0.9)
        mean_before = IO.pick(block, "mean_before_ratio_pct")
        if mean_before is not None:
            pooled = block.assign(_value=IO.numeric(mean_before)).groupby("decile")["_value"].median()
            ax.plot(np.asarray(pooled.index, dtype=float), pooled.to_numpy(dtype=float), color=S.KEY, lw=1.1, ls=(0, (3, 2)), zorder=4)
        ax.axhline(0.0, color=S.GREY, lw=0.9, zorder=0)
        ax.set_xlabel("decile of the Gaussian value", labelpad=2)
        ax.set_xticks([1, 5, 10])
        S.panel_label(ax, string.ascii_uppercase[index], S.MEASURE_LABEL[measure])
        if index == 0:
            ax.set_ylabel("difference (per cent of Gaussian)", labelpad=2)
        if log:
            S.log_scale(ax, "y", IO.numeric(IO.pick(block, "median_pct", default=pd.Series([]))))
    handles = [Line2D([], [], color=S.col_d(11), lw=0.9, label="median per cent, one line per $d$"), Line2D([], [], color=S.KEY, lw=1.1, ls=(0, (3, 2)), label="mean before ratio, pooled over $d$"), Line2D([], [], color=S.GREY, lw=0.9, label="zero")]
    S.legend_below(fig, handles, ncols=3, reserve=0.17)
    S.dimension_colorbar(fig, list(axes))
    S.source_label(fig, label)
    return fig


def render(source, out_dir):
    """Render this module's figures from one Phase 4 output folder into out_dir and return the saved paths."""
    table = im_units(source)
    label = IO.phase_paths(source)["label"]
    paths = []
    if table is not None:
        table = IO.ok_units(table)
        for kind in ("nats", "pct"):
            paths += S.log_twin(_differences_vs_dimension, out_dir, f"phase4_differences_vs_dimension_{kind}", table, kind, label)
        paths += S.log_twin(_setwise_vs_gaussian, out_dir, "phase4_setwise_vs_gaussian", table, label)
    paths += S.log_twin(_pct_by_decile, out_dir, "phase4_pct_by_decile", IO.read_table(source, "im_by_gaussian_decile.csv"), label)
    return paths
