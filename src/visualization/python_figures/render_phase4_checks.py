"""Phase 4, the two checks that sit beside the closed-form differences.

  phase4_o_sign_flips  How often does the O-information change sign between the Gaussian and the fitted t, and at which interaction orders?
  phase4_ksg_vs_model  At the two dimensions where a model-free estimate exists, does the closed-form value agree with the k-nearest-neighbour one?

In phase4_ksg_vs_model the model-free estimate is the horizontal axis of both panels, so a point above the grey equality line is a model value larger than the model-free one.

The model values are drawn over the units with `numerical_ok`; the sign-flip rates come from the phase's own `o_sign_flips_by_d.csv`, which likewise leaves out the failed fits.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _phase_io as IO  # noqa: E402
import _style as S  # noqa: E402
from render_phase4_differences import im_units  # noqa: E402

plt = S.style()
from matplotlib.lines import Line2D  # noqa: E402


def _o_sign_flips(log, flips, label):
    """Two panels against d: the rate (A) and the count (B) of channel sets whose O-information changes sign."""
    if flips is None:
        return None
    IO.require(flips, "d")
    dims = IO.numeric(flips["d"])
    rate = IO.pick(flips, "rate")
    count = IO.pick(flips, "n_flips")
    if rate is None and count is None:
        print("  [skip] phase4_o_sign_flips: no rate or n_flips column")
        return None
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(S.WIDTH, 2.6), layout="constrained")
    if rate is not None:
        axA.plot(dims, IO.numeric(rate), color=S.model_color("t"), lw=1.2, marker="o", ms=2.6, mfc="white", mew=0.8, label="sign-flip rate")
    axA.axhline(0.0, color=S.GREY, lw=0.9, zorder=0)
    axA.set_ylabel("fraction of channel sets", labelpad=2)
    S.dimension_axis(axA, dims)
    S.panel_label(axA, "A", "rate of a sign change")

    if count is not None:
        axB.bar(dims, IO.numeric(count), width=0.8, color=[S.col_d(d) for d in dims], label="channel sets that flip")
    axB.set_ylabel("channel sets", labelpad=2)
    S.dimension_axis(axB, dims)
    S.panel_label(axB, "B", "count of sign changes")
    if log:
        if rate is not None:
            S.log_scale(axA, "y", IO.numeric(rate))
        if count is not None:
            S.log_scale(axB, "y", IO.numeric(count))

    handles = axA.get_legend_handles_labels()[0] + axB.get_legend_handles_labels()[0]
    handles.append(Line2D([], [], color=S.GREY, lw=0.9, label="zero"))
    S.legend_below(fig, handles, ncols=3, reserve=0.16)
    S.source_label(fig, label)
    return fig


def _ksg_vs_model(log, table, label):
    """One panel per KSG dimension: the closed-form t and Gaussian values against the model-free estimate."""
    panels = [("ksg_MI", "MI", 2, "mutual information"), ("ksg_TC", "TC", 3, "total correlation")]
    panels = [p for p in panels if p[0] in table.columns and np.isfinite(IO.numeric(table[p[0]])).any()]
    if not panels:
        print("  [skip] phase4_ksg_vs_model: no ksg_MI or ksg_TC column carries data")
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(S.WIDTH, 2.8), layout="constrained", squeeze=False)
    axes = axes.ravel()
    for index, (column, measure, dimension, name) in enumerate(panels):
        ax = axes[index]
        block = table[IO.numeric(table["d"]) == dimension]
        ksg = IO.numeric(block[column])
        extent = [ksg[np.isfinite(ksg)]]
        for model, marker in (("t", "o"), ("gaussian", "s")):
            source_column = f"{measure}_t" if model == "t" else f"{measure}_G"
            if source_column not in block.columns:
                continue
            values = IO.numeric(block[source_column])
            keep = np.isfinite(ksg) & np.isfinite(values)
            ax.scatter(ksg[keep], values[keep], s=3.0, lw=0, alpha=0.45, color=S.model_color(model), marker=marker, rasterized=True, label=S.MODEL_LABEL[model])
            extent.append(values[keep])
        span = np.concatenate([v for v in extent if v.size]) if extent else np.array([0.0, 1.0])
        if span.size:
            limits = (float(np.nanmin(span)), float(np.nanmax(span)))
            ax.plot(limits, limits, color=S.GREY, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.set_xlabel(f"k-nearest-neighbour {name} (nats)", labelpad=2)
        S.panel_label(ax, "AB"[index], f"$d = {dimension}$, {name}")
        if index == 0:
            ax.set_ylabel("model value (nats)", labelpad=2)
        if log:
            S.log_scale(ax, "x", span)
            S.log_scale(ax, "y", span)
    handles = [Line2D([], [], color=S.model_color("t"), marker="o", ls="none", ms=3, label="fitted $t$"), Line2D([], [], color=S.model_color("gaussian"), marker="s", ls="none", ms=3, label="Gaussian"), Line2D([], [], color=S.GREY, lw=0.8, ls=(0, (3, 2)), label="equality")]
    S.legend_below(fig, handles, ncols=3, reserve=0.16)
    S.source_label(fig, label)
    return fig


def render(source, out_dir):
    """Render this module's figures from one Phase 4 output folder into out_dir and return the saved paths."""
    label = IO.phase_paths(source)["label"]
    paths = S.log_twin(_o_sign_flips, out_dir, "phase4_o_sign_flips", IO.read_table(source, "o_sign_flips_by_d.csv"), label)
    table = im_units(source)
    if table is not None:
        paths += S.log_twin(_ksg_vs_model, out_dir, "phase4_ksg_vs_model", IO.ok_units(table), label)
    return paths
