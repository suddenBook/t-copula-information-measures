"""Phase 7, the correlation factorial: does the estimator survive a change of R?

  phase7_factorial_bias   Median and mean bias of p_hat by correlation design and dimension, with each design's condition number in the key and the spread of a single fit beside it, at the factorial's one observation window T/tau (named in the panel title).

Each design is a 20-channel parent, and dimension d fits its leading d x d block, so the condition number in the key is the median over the dimensions drawn.

The designs are a categorical object without an assigned colour, so they take the categorical ramp.
The condition number travels with the design in the legend rather than as a second colour, because one colour stands for only one object.

The bias and the spread are drawn over the replicates with `numerical_ok`, the population `simulationSummaries` reports over, because a failed fit can still leave a finite `p_hat` behind.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _phase_io as IO  # noqa: E402
import _style as S  # noqa: E402
from render_phase7_shape_recovery import SHAPE, WINDOW, window_text  # noqa: E402

plt = S.style()
from matplotlib.lines import Line2D  # noqa: E402


def _design_label(name, block):
    """The legend label of one design: its name with the median condition number of its blocks, or the name alone when the table carries none."""
    cond = IO.pick(block, "cond_R")
    if cond is None:
        return str(name)
    values = IO.numeric(cond)
    values = values[np.isfinite(values)]
    if not values.size:
        return str(name)
    return f"{name}, cond$(R) = {np.median(values):.3g}$"


def _factorial_bias(log, factorial, label):
    """`phase7_factorial_bias`: the bias and the single-fit spread of p_hat by design and dimension."""
    if factorial is None:
        return None
    factorial = IO.ok_units(factorial)
    IO.require(factorial, "design", "d")
    truth = IO.pick(factorial, "p_true")
    if truth is None or "p_hat" not in factorial.columns:
        print("  [skip] phase7_factorial_bias: r_factorial_long.csv needs p_true and p_hat")
        return None
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(S.WIDTH, 2.8), layout="constrained")
    handles = []
    for index, (design, block) in enumerate(factorial.groupby("design")):
        colour = S.categorical_color(index)
        grouped = block.groupby("d")
        dims = np.asarray(list(grouped.groups), dtype=float)
        p_true = grouped["p_true"].median().to_numpy(dtype=float)
        median = grouped["p_hat"].median().to_numpy(dtype=float)
        mean = grouped["p_hat"].mean().to_numpy(dtype=float)
        spread = grouped["p_hat"].std(ddof=1).to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            axA.plot(dims, 100.0 * (median - p_true) / p_true, color=colour, lw=1.1, marker="o", ms=2.2, mfc="white", mew=0.6)
            axA.plot(dims, 100.0 * (mean - p_true) / p_true, color=colour, lw=1.0, ls=(0, (2, 1.6)))
            axB.plot(dims, spread / p_true, color=colour, lw=1.1)
        handles.append(Line2D([], [], color=colour, lw=1.4, label=_design_label(design, block)))
    axA.axhline(0.0, color=S.GREY, lw=0.9, zorder=0)
    axA.set_ylabel(f"bias (per cent of true {SHAPE})", labelpad=2)
    S.dimension_axis(axA, IO.numeric(factorial["d"]))
    window = np.unique(IO.numeric(factorial[WINDOW])) if WINDOW in factorial.columns else np.array([])
    S.panel_label(axA, "A", "bias by design" + (f", {window_text(window[0])}" if window.size == 1 else ""))
    axB.set_ylabel(r"$\mathrm{sd}(1/\hat\nu)\,/\,(1/\nu)$", labelpad=2)
    S.dimension_axis(axB, IO.numeric(factorial["d"]))
    S.panel_label(axB, "B", "spread of one fit")
    if log:
        S.log_scale(axB, "y", IO.numeric(factorial["p_hat"]))
    handles += [Line2D([], [], color=S.KEY, lw=1.1, marker="o", ms=2.2, mfc="white", mew=0.6, label="median"), Line2D([], [], color=S.KEY, lw=1.0, ls=(0, (2, 1.6)), label="mean"), Line2D([], [], color=S.GREY, lw=0.9, label="no bias"), Line2D([], [], color="none", label="numerically successful replicates only")]
    S.legend_below(fig, handles, ncols=3, reserve=0.20)
    S.source_label(fig, label)
    return fig


def render(source, out_dir):
    """Draw the factorial figure and its log twin, and return the saved paths."""
    factorial = IO.read_table(source, "r_factorial_long.csv")
    if factorial is None:
        # The summary carries no `p_hat` column, so with only the summary present `_factorial_bias` prints a skip line and draws nothing.
        factorial = IO.read_table(source, "r_factorial_summary.csv")
    return S.log_twin(_factorial_bias, out_dir, "phase7_factorial_bias", factorial, IO.phase_paths(source)["label"])
