"""Phase 3, the pooled distributions the surrogate comparison is made of.

  phase3_distributions_by_dimension  One panel per dimension: the distribution of p_hat = 1/nu_hat over the channel sets, for two disjoint halves of the observed sets and for the same sets refitted on two surrogate realizations.
                                     Do the observed curves lie on the surrogate ones, the way the two halves lie on each other and the two realizations lie on each other?

How a panel reads:
p_hat has an atom, because a fit that lands on the Gaussian copula is exactly 0, which a logarithmic axis cannot show.
Each panel therefore has two parts on one axis in one unit, the share of the sample.
The curves are the positive estimates on a logarithmic axis, as the share of the sample in each bin; the bins are equal on the log axis and shared by every sample and every panel.
The stems at the left, above the tick "0", are the share of each sample exactly at p_hat = 0, and the stems at the right, above "3", the share exactly at the heavy end of the domain, drawn when any sample has a value there.
A curve's bins and its stems sum to one, so a stem and a curve compare directly, which a density per decade beside a probability would not.
Each panel prints the overlap coefficients Phase 3 computed: OVL of the observed sets against realization 1, then the two yardsticks, realization 1 against 2 and the observed halves against each other.

The population:
Every sample is its numerically successful fits, because a failed fit can keep a finite p_hat that is not an estimate of anything.
The rows come from `surrogate_units.csv`, one row per channel set, filtered on `numerical_ok_obs` and `numerical_ok_s1` / `numerical_ok_s2`; the overlaps come from `surrogate_comparison_by_d.csv`, which Phase 3 computes over the same population.
"""

from __future__ import annotations

import os
import string
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _phase_io as IO  # noqa: E402
import _style as S  # noqa: E402
from render_phase3_surrogate import OBSERVED, OBSERVED_YARDSTICK, SURROGATE, SURROGATE_YARDSTICK  # noqa: E402

plt = S.style()
from matplotlib.lines import Line2D  # noqa: E402

HEAVY_END = 3.0  # the upper edge of the domain [0, 3] of p = 1/nu, an atom of p_hat
BINS_PER_DECADE = 8
MIN_BINS, MAX_BINS = 10, 60
# (sample, label, colour, dashes), in drawing order.
SAMPLES = (("half1", "observed, half 1", OBSERVED, "-"), ("half2", "observed, half 2", OBSERVED_YARDSTICK, (0, (3, 2))), ("sur1", "surrogate realization 1", SURROGATE, "-"), ("sur2", "surrogate realization 2", SURROGATE_YARDSTICK, (0, (3, 2))))


def samples_by_dimension(units, d):
    """The four samples of one dimension, each over its numerically successful fits."""
    block = units[IO.numeric(units["d"]) == d]
    observed = IO.numeric(block["p_hat_obs"])
    ok = IO.flag(block["numerical_ok_obs"])
    half = IO.numeric(block["half"])
    out = {"half1": observed[ok & (half == 1)], "half2": observed[ok & (half == 2)]}
    for b in (1, 2):
        values = IO.numeric(block[f"p_hat_s{b}"])
        out[f"sur{b}"] = values[IO.flag(block[f"numerical_ok_s{b}"])]
    return out


def _bins(all_samples):
    """Log10 bin edges shared by every panel, over the pooled estimates strictly between 0 and the heavy end.

    There are about BINS_PER_DECADE bins per decade of the range they span, clipped to MIN_BINS to MAX_BINS, so a distribution far from the others keeps its shape instead of collapsing into one bin.
    Returns None when there is no such estimate.
    """
    positive = np.concatenate([v[(v > 0) & (v < HEAVY_END)] for per_d in all_samples.values() for v in per_d.values()] or [np.array([])])
    if positive.size == 0:
        return None
    low = float(np.log10(positive.min()))
    high = float(np.log10(max(positive.max(), positive.min() * 1.001)))
    count = int(np.clip(np.ceil(BINS_PER_DECADE * (high - low)), MIN_BINS, MAX_BINS))
    return np.linspace(low, high, count + 1)


def _stems(ax, x, share, width, colour, index):
    """One sample's share at an atom, as a stem beside the other samples' stems."""
    position = x + (index - 1.5) * width
    ax.vlines(position, 0.0, share, color=colour, lw=1.3, zorder=3)
    ax.plot(position, share, marker="o", ms=2.4, color=colour, mec=colour, zorder=4)


def _panel(ax, per_d, edges, heavy):
    """Draw one dimension: the log-binned curves of the positive estimates and the stems at the atoms."""
    span = edges[-1] - edges[0] if edges[-1] > edges[0] else 1.0
    gaussian_x = edges[0] - 0.22 * span
    heavy_x = edges[-1] + 0.22 * span
    width = 0.035 * span
    for index, (key, _, colour, dashes) in enumerate(SAMPLES):
        values = per_d.get(key, np.array([]))
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        inside = values[(values > 0) & (values < HEAVY_END)]
        counts, _ = np.histogram(np.log10(inside), bins=edges)
        ax.stairs(counts / values.size, edges, baseline=None, color=colour, ls=dashes, lw=1.0)
        _stems(ax, gaussian_x, float(np.mean(values == 0.0)), width, colour, index)
        if heavy:
            _stems(ax, heavy_x, float(np.mean(values == HEAVY_END)), width, colour, index)
    ax.axvline(edges[0] - 0.11 * span, color=S.GREY, lw=0.6, ls=(0, (2, 2)), zorder=0)
    # At most three decades are labelled, counted down from the largest, so the labels of a narrow panel never run into each other.
    decades = list(range(int(np.ceil(edges[0])), int(np.floor(edges[-1])) + 1))
    decades = decades[::-1][:: max(1, int(np.ceil(len(decades) / 3)))][::-1]
    ticks = [gaussian_x] + [float(t) for t in decades]
    labels = ["0"] + [f"$10^{{{int(t)}}}$" for t in decades]
    if heavy:
        ax.axvline(edges[-1] + 0.11 * span, color=S.GREY, lw=0.6, ls=(0, (2, 2)), zorder=0)
        ticks.append(heavy_x)
        labels.append("3")
    ax.set_xticks(ticks, labels)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_xlim(gaussian_x - 3 * width, (heavy_x if heavy else edges[-1] + 0.04 * span) + (3 * width if heavy else 0.0))
    # A fifth of the height is left free above the tallest curve or stem for the overlap annotation, so no curve can run under it.
    ax.set_ylim(0.0, 1.25 * max(ax.get_ylim()[1], 1e-6))


def _annotation(ax, comparison, d):
    """The three overlaps of this dimension, as Phase 3 computed them."""
    if comparison is None:
        return
    row = comparison[IO.numeric(comparison["d"]) == d]
    if row.empty:
        return
    value = {c: float(IO.numeric(row[c])[0]) if c in row.columns else np.nan for c in ("ovl_obs_sur", "ovl_sur_sur2", "ovl_obs_halves")}
    ax.text(0.98, 0.98, f"{value['ovl_obs_sur']:.2f} ({value['ovl_sur_sur2']:.2f}, {value['ovl_obs_halves']:.2f})", transform=ax.transAxes, ha="right", va="top", fontsize=6.0, color=OBSERVED, zorder=5, bbox=dict(fc="white", ec="none", pad=0.6))


def _distributions_by_dimension(units, comparison, label):
    """The grid of per-dimension panels, or None when the units table is missing, lacks a required column, or holds no estimate strictly between the two atoms."""
    if units is None:
        return None
    required = ("d", "half", "p_hat_obs", "numerical_ok_obs", "p_hat_s1", "numerical_ok_s1", "p_hat_s2", "numerical_ok_s2")
    if not IO.has(units, *required):
        missing = [c for c in required if c not in units.columns]
        print(f"  [skip] phase3_distributions_by_dimension: surrogate_units.csv carries no {', '.join(missing)}")
        return None
    dims = IO.dimensions(units)
    all_samples = {int(d): samples_by_dimension(units, d) for d in dims}
    edges = _bins(all_samples)
    if edges is None:
        print("  [skip] phase3_distributions_by_dimension: no positive estimate to draw")
        return None
    heavy = any(np.any(v == HEAVY_END) for per_d in all_samples.values() for v in per_d.values())

    columns = min(5, len(dims))
    lines = int(np.ceil(len(dims) / columns))
    fig, axes = plt.subplots(lines, columns, figsize=(S.WIDTH, 1.55 * lines + 1.1), layout="constrained", squeeze=False, sharey=False)
    flat = axes.ravel()
    for axis in flat[len(dims) :]:
        axis.set_axis_off()
    for index, d in enumerate(dims):
        ax = flat[index]
        _panel(ax, all_samples[int(d)], edges, heavy)
        _annotation(ax, comparison, d)
        S.panel_label(ax, string.ascii_uppercase[index % 26], f"$d = {int(d)}$")
        if index % columns == 0:
            ax.set_ylabel("share of the fits", labelpad=2)
        if index >= len(dims) - columns:
            ax.set_xlabel(r"$\hat p = 1/\hat\nu$", labelpad=2)
    handles = [Line2D([], [], color=colour, ls=dashes, lw=1.0, label=name) for _, name, colour, dashes in SAMPLES]
    handles += [
        Line2D([], [], color=S.KEY, lw=1.3, marker="o", ms=2.4, label=r"stem: share exactly at $\hat p = 0$ (Gaussian)" + (" or 3" if heavy else "")),
        Line2D([], [], color="none", label="top right: OVL, observed vs realization 1 (realization 1 vs 2, observed halves)"),
        Line2D([], [], color="none", label="numerically successful fits only"),
    ]
    S.legend_below(fig, handles, ncols=2, reserve=0.06 + 0.16 / max(lines, 1))
    S.source_label(fig, label)
    return fig


def render(source, out_dir):
    """Render phase3_distributions_by_dimension from one Phase 3 output folder into out_dir and return the saved paths."""
    units = IO.read_table(source, "surrogate_units.csv")
    comparison = IO.read_table(source, "surrogate_comparison_by_d.csv")
    label = IO.phase_paths(source)["label"]
    paths = []
    figure = _distributions_by_dimension(units, comparison, label)
    if figure is not None:
        paths += S.savefig_pair(figure, out_dir, "phase3_distributions_by_dimension")
    return paths
