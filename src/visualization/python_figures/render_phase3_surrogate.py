"""Phase 3, the pooled surrogate comparison read by dimension.

  phase3_overlap_vs_dimension            How much of the observed distribution of p_hat does surrogate realization 1 cover at each dimension, against the two yardsticks of two surrogate realizations and two halves of the observed channel sets?
                                         Panel B is the mass outside the overlap, 1 - OVL.
  phase3_surrogate_floor                 How large an information difference does the surrogate reference itself produce, per measure and per dimension, that is, the floor a measured difference has to clear?
  phase3_gaussian_fraction_vs_dimension  How much of each distribution sits exactly on the Gaussian, p_hat = 0?

Everything here reads `surrogate_comparison_by_d.csv` and `surrogate_floor_by_d.csv`, which Phase 3 computes over the numerically successful fits only, so no statistic drawn here can include a failed fit.
The comparison is descriptive: OVL is an overlap coefficient between two pooled distributions, not a test, and nothing here is a p-value.

The observed recording is drawn in the fitted-t near-black, the reported surrogate realization in the surrogate blue, and the yardstick realization in a lighter shade of that blue.
The observed-halves yardstick is a dark grey distinct from the reference grey, which is kept for zero lines and bounds.
"""

from __future__ import annotations

import os
import string
import sys

import numpy as np
from matplotlib.colors import to_rgb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _phase_io as IO  # noqa: E402
import _style as S  # noqa: E402

plt = S.style()
from matplotlib.lines import Line2D  # noqa: E402

# The two surrogate realizations: the reported one in the surrogate blue and the yardstick in a lighter shade of the same blue, so both read as the surrogate object.
SURROGATE = S.model_color("surrogate")
SURROGATE_YARDSTICK = tuple(0.5 * np.asarray(to_rgb(SURROGATE)) + 0.5)
# The two observed halves: the fitted-t near-black and a dark grey that stays clear of the reference grey.
OBSERVED = S.KEY
OBSERVED_YARDSTICK = "#555555"

# (column, label, colour, dashes, marker) of the three overlaps.
OVERLAPS = (("ovl_obs_sur", "observed against surrogate realization 1", OBSERVED, "-", "o"), ("ovl_sur_sur2", "surrogate realization 1 against 2", SURROGATE, (0, (3, 2)), "s"), ("ovl_obs_halves", "observed half 1 against half 2", OBSERVED_YARDSTICK, (0, (1, 1.4)), "^"))


def _overlap_vs_dimension(log, comparison, label):
    """Two panels against d: the three overlap coefficients (A) and the mass outside each overlap (B)."""
    if comparison is None:
        return None
    present = [entry for entry in OVERLAPS if entry[0] in comparison.columns]
    if not present:
        print("  [skip] phase3_overlap_vs_dimension: surrogate_comparison_by_d.csv carries no ovl_obs_sur / ovl_sur_sur2 / ovl_obs_halves column")
        return None
    dims = IO.numeric(IO.require(comparison, "d"))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(S.WIDTH, 2.8), layout="constrained")
    outside = []
    for column, name, colour, dashes, marker in present:
        overlap = IO.numeric(comparison[column])
        for ax, values in ((axA, overlap), (axB, 1.0 - overlap)):
            ax.plot(dims, values, color=colour, lw=1.1, ls=dashes, marker=marker, ms=2.6, mfc="white", mew=0.8, label=name)
        outside.append(1.0 - overlap)
    axA.axhline(1.0, color=S.GREY, lw=0.8, zorder=0)
    axA.set_ylabel("overlap coefficient OVL", labelpad=2)
    S.dimension_axis(axA, dims)
    S.panel_label(axA, "A", "overlap of the pooled distributions")
    if not log:
        axB.axhline(0.0, color=S.GREY, lw=0.8, zorder=0)
    axB.set_ylabel("mass outside the overlap, 1 - OVL", labelpad=2)
    S.dimension_axis(axB, dims)
    S.panel_label(axB, "B", "mass outside")
    if log:
        S.log_scale(axB, "y", np.concatenate(outside))
    handles = axA.get_legend_handles_labels()[0]
    handles.append(Line2D([], [], color="none", label="numerically successful fits only; descriptive, no p-value"))
    S.legend_below(fig, handles, ncols=2, reserve=0.20)
    S.source_label(fig, label)
    return fig


def _surrogate_floor(log, floors, label):
    """One panel per measure: the percentiles of the surrogate difference against d, all four on realization 1 and the 95th on realization 2."""
    if floors is None:
        return None
    IO.require(floors, "d", "measure")
    realization = IO.numeric(floors["realization"]) if "realization" in floors.columns else np.ones(len(floors))
    measures = [m for m in S.MEASURES if (IO.text(floors["measure"]) == m).any()]
    if not measures:
        print("  [skip] phase3_surrogate_floor: no declared measure in surrogate_floor_by_d.csv")
        return None
    fig, axes = plt.subplots(1, len(measures), figsize=(S.WIDTH, 2.6), layout="constrained", squeeze=False)
    axes = axes.ravel()
    for index, measure in enumerate(measures):
        ax = axes[index]
        pooled = []
        of_measure = IO.text(floors["measure"]) == measure
        reported = floors[of_measure & (realization == 1)].sort_values("d")
        for column, dashes in (("q50", (0, (1, 1.4))), ("q90", (0, (3, 2))), ("q95", "-"), ("q99", (0, (5, 1, 1, 1)))):
            values = IO.pick(reported, column)
            if values is None:
                continue
            # Markers as well as lines, because mutual information has a floor at d = 2 only and a one-point line draws nothing.
            ax.plot(IO.numeric(reported["d"]), IO.numeric(values), color=SURROGATE, lw=1.0, ls=dashes, marker="o", ms=1.8, label=f"{column[1:]} per cent, realization 1")
            pooled.append(IO.numeric(values))
        yardstick = floors[of_measure & (realization == 2)].sort_values("d")
        if len(yardstick) and "q95" in yardstick.columns:
            ax.plot(IO.numeric(yardstick["d"]), IO.numeric(yardstick["q95"]), color=SURROGATE_YARDSTICK, lw=1.4, marker="o", ms=2.4, zorder=1, label="95 per cent, realization 2")
            pooled.append(IO.numeric(yardstick["q95"]))
        S.dimension_axis(ax, IO.numeric(floors["d"]))
        S.panel_label(ax, string.ascii_uppercase[index], S.MEASURE_LABEL[measure])
        if index == 0:
            ax.set_ylabel("difference on surrogates (nats)", labelpad=2)
        if log and pooled:
            S.log_scale(ax, "y", np.concatenate(pooled))
    handles = axes[0].get_legend_handles_labels()[0]
    S.legend_below(fig, handles, ncols=3, reserve=0.20)
    S.source_label(fig, label)
    return fig


def _gaussian_fraction(comparison, label):
    """The fraction of each sample exactly at p_hat = 0, against d."""
    if comparison is None:
        return None
    series = [(column, name, colour, dashes) for column, name, colour, dashes in (("obs_fraction_p0", "observed", OBSERVED, "-"), ("sur_fraction_p0", "surrogate realization 1", SURROGATE, "-"), ("sur2_fraction_p0", "surrogate realization 2", SURROGATE_YARDSTICK, (0, (3, 2)))) if column in comparison.columns]
    if not series:
        print("  [skip] phase3_gaussian_fraction_vs_dimension: no *_fraction_p0 column")
        return None
    dims = IO.numeric(IO.require(comparison, "d"))
    fig, ax = plt.subplots(1, 1, figsize=(S.WIDTH, 2.6), layout="constrained")
    for column, name, colour, dashes in series:
        ax.plot(dims, IO.numeric(comparison[column]), color=colour, lw=1.2, ls=dashes, marker="o", ms=2.6, mfc="white", mew=0.8, label=name)
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel(r"fraction exactly Gaussian, $\hat p = 0$", labelpad=2)
    S.dimension_axis(ax, dims)
    S.panel_label(ax, "A", "mass on the Gaussian")
    handles = ax.get_legend_handles_labels()[0]
    handles.append(Line2D([], [], color="none", label="numerically successful fits only"))
    S.legend_below(fig, handles, ncols=4, reserve=0.16)
    S.source_label(fig, label)
    return fig


def render(source, out_dir):
    """Render this module's figures from one Phase 3 output folder into out_dir and return the saved paths."""
    comparison = IO.read_table(source, "surrogate_comparison_by_d.csv")
    floors = IO.read_table(source, "surrogate_floor_by_d.csv")
    label = IO.phase_paths(source)["label"]
    paths = []
    paths += S.log_twin(_overlap_vs_dimension, out_dir, "phase3_overlap_vs_dimension", comparison, label)
    paths += S.log_twin(_surrogate_floor, out_dir, "phase3_surrogate_floor", floors, label)
    figure = _gaussian_fraction(comparison, label)
    if figure is not None:
        paths += S.savefig_pair(figure, out_dir, "phase3_gaussian_fraction_vs_dimension")
    return paths
