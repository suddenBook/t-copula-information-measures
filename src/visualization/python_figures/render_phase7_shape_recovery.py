"""Phase 7, how well the estimator recovers a known shape from an autocorrelated path.

  phase7_recovery_spread_sXX          Does the relative spread of p_hat fall like (T/tau)^-1/2, and what is the spread-law constant at each dimension?
  phase7_recovery_bias_sXX            Where in p does the estimate carry bias, by median and by mean, at each dimension, at the longest window?
  phase7_recovery_gaussian_truth_sXX  At a Gaussian truth, how much of the mass lands exactly on p = 0 and how large is the rest, by (d, T/tau)?
  phase7_recovery_heavy_sXX           How often does a fit land at the heavy end of the domain?
  phase7_recovery_sampling            Does denser sampling of the same window change the estimate?
                                      For a few representative truths (rows) and dimensions (columns), it draws the median error of p_hat with its interquartile band and the sd of p_hat against the samples per tau on a log axis, one line per window T/tau.

The axes are the observation window and the sampling rate.
The truth is a path of the Student-t diffusion in units of its time scale tau, so the amount of data is T/tau (with n = T/tau x samples per tau fitted rows), and every axis of data size is a T/tau axis on a log scale.
The samples per tau are an axis of their own, and the rates are nested: every rate reads the same paths.
So each figure above that is drawn at one rate is written once per rate the table carries, with the suffix `_sXX` (the rate, zero-padded to two digits: `_s05`, `_s10`, ..., `_s80`, `_s160`) and the rate named under the source stamp, and `phase7_recovery_sampling` reads across the rates.

The code carries the shape as p = 1/nu, but figure text never does: a truth reads `1/nu = 0.1 (nu = 10)`, an estimate 1/nu_hat, and its error and spread `error of 1/nu_hat` and `sd(1/nu_hat)`.
`truth_title` and `rate_grid` carry that notation to every Phase 7 figure and to the cross-recording time axis.

The three quantitative panels (spread, bias, and the Gaussian-truth distribution of p_hat) are drawn over the replicates with `numerical_ok`, which is the population `simulationSummaries` reports over.
A finite estimate survives several fit failures, so filtering on `isfinite` is not the same thing.
The heavy-end panel is a status rate and keeps every replicate, which its key says.

Dimension is the viridis ramp.
The window T/tau takes the Reds ramp of the amount of data (`_style.col_T`, read by value on log10 T/tau over the declared windows, so one window is one shade in every Phase 7 figure).
The (T/tau)^-1/2 guide and the zero line are reference grey.
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
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

WINDOW = "T_over_tau"
WINDOW_AXIS = r"observation window $T/\tau$"
RATE = "samples_per_tau"
RATE_AXIS = r"samples per $\tau$"

# Representative cells of the recovery grid: the Gaussian truth, a mild and a heavier shape; a low, a middle and the top dimension.
# A run without one of them is drawn at the nearest value it has (nearest on log p for the shapes, on d for the dimensions).
# The cross-recording time axis reads the same cells.
TRUTHS = (0.0, 0.1, 0.5)
DIMENSIONS = (2, 5, 20)

# The shape in figure text: 1/nu, never p.
SHAPE = r"$1/\nu$"
ESTIMATE = r"$1/\hat\nu$"
ERROR_LABEL = r"error of $1/\hat\nu$"
SD_LABEL = r"$\mathrm{sd}(1/\hat\nu)$"


def window_text(window):
    """T/tau in a math label, with a thin space as the thousands separator."""
    return r"$T/\tau = " + f"{float(window):,.0f}".replace(",", r"\,") + "$"


def windows(table):
    """The finite observation windows T/tau a table carries, ascending."""
    values = np.unique(IO.numeric(table[WINDOW]))
    return values[np.isfinite(values)]


def window_keys(table):
    """One legend key per window of the table, in its Reds shade."""
    return [Line2D([], [], color=S.col_T(w), lw=1.2, label=window_text(w)) for w in windows(table)]


def rates(table):
    """The sampling rates (samples per tau) a table carries, ascending."""
    values = np.unique(IO.numeric(IO.require(table, RATE)))
    return values[np.isfinite(values)]


def rate_suffix(rate):
    """The stem suffix of one rate: `_s05`, `_s10`, ... (zero-padded to at least two digits)."""
    return f"_s{float(rate):02g}"


def rate_label(label, rate):
    """The source stamp of a figure drawn at one rate: the run, and the rate below it."""
    return f"{label}  {float(rate):g} samples per " + r"$\tau$"


def by_rate(table):
    """(rate, rows at that rate) over every rate the table carries, ascending and one at a time, so a long table is never held once more in full as its rate blocks."""
    column = IO.numeric(IO.require(table, RATE))
    for rate in rates(table):
        yield rate, table[column == rate]


def at_rate(table, rate):
    """The rows of a table at one rate, or None when the table or the rate is absent."""
    if table is None:
        return None
    block = table[IO.numeric(IO.require(table, RATE)) == rate]
    return None if block.empty else block


def nearest(preferred, available, log):
    """The available value nearest each preferred one, without repeats, in that order."""
    chosen = []
    for want in preferred:
        pool = [v for v in available if v not in chosen and (v > 0) == (want > 0)]
        if not pool:
            continue
        if want > 0 and log:
            chosen.append(min(pool, key=lambda v: (abs(np.log(v / want)), v)))
        else:
            chosen.append(min(pool, key=lambda v: (abs(v - want), v)))
    return chosen


def nu_text(nu):
    """nu for a label: 1/3 as a fraction, infinity as the symbol."""
    nu = float(nu)
    if not np.isfinite(nu):
        return r"$\infty$"
    if nu < 1 and abs(1.0 / nu - round(1.0 / nu)) < 1e-9:
        return f"1/{round(1.0 / nu):d}"
    return f"{nu:g}"


def truth_title(p, d):
    """A representative cell as a panel title: the true 1/nu = p, its nu and the dimension."""
    nu = np.inf if p == 0 else 1.0 / p
    return rf"$1/\nu = {p:g}$ ($\nu = ${nu_text(nu)}), $d = {int(d)}$"


def rate_axis(ax, values):
    """A log axis of samples per tau, ticked at the rates drawn, in plain numbers."""
    ax.set_xscale("log")
    ax.set_xticks(values)
    ax.set_xticklabels([f"{v:g}" for v in values])
    ax.minorticks_off()


def plain_log_axis(ax, axis, values):
    """A log axis ticked in plain numbers over the values drawn.

    The ticks are whole decades over a span wider than a decade and a half, then 1-2-5, 1-2-3-5 and finer steps as the span narrows, thinned to at most six.
    When the span holds fewer than two, the nearest step on either side is added, so a narrow range still carries two labelled ticks (the default locator labels the minor ticks there in powers of ten, or nothing).
    A value at or below zero has no logarithm, and the axis is then `_style.log_scale`'s.
    """
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0 or (v <= 0).any():
        return S.log_scale(ax, axis, v)
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    lo, hi = float(v.min()), float(v.max())
    span = np.log10(hi / lo)
    steps = (1.0,) if span > 1.5 else (1.0, 2.0, 5.0) if span > 0.8 else (1.0, 2.0, 3.0, 5.0) if span > 0.4 else (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
    decades = np.arange(np.floor(np.log10(lo)) - 1, np.ceil(np.log10(hi)) + 1)
    candidates = sorted(m * 10.0**e for e in decades for m in steps)
    ticks = [t for t in candidates if lo / 1.02 <= t <= hi * 1.02]
    ticks = ticks[:: int(np.ceil(len(ticks) / 6))] if ticks else ticks
    if len(ticks) < 2:
        ticks = sorted(set(ticks + [t for t in candidates if t < lo][-1:] + [t for t in candidates if t > hi][:1]))
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(FixedLocator(ticks))
    target.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    target.set_minor_formatter(NullFormatter())
    (ax.set_xlim if axis == "x" else ax.set_ylim)(min(ticks[0], lo) / 1.15, max(ticks[-1], hi) * 1.15)
    return "log"


def key_below(fig, handles, ncols, clear_stamp=False):
    """The house key under the panels, on a strip added to the figure's height.

    A key that grows with its entries then does not squeeze the panels.
    `clear_stamp` lifts the key above the two lines of the source stamp, for a key wide enough to reach the stamp in the lower right corner.
    """
    width, height = fig.get_size_inches()
    lift = 0.26 if clear_stamp else 0.0
    strip = 0.12 + 0.155 * int(np.ceil(len(handles) / ncols)) + lift
    fig.set_size_inches(width, height + strip)
    S.legend_below(fig, handles, ncols=ncols, y=lift / (height + strip), reserve=strip / (height + strip))


def rate_grid(cells, dims, label, notes=()):
    """Median error with its interquartile band, and sd, of p_hat against the rate.

    `cells` is [(truth p, rows)] with one entry per truth; each truth takes two rows of panels (the error, then the sd) and each dimension of `dims` a column.
    The rows carry `d`, `T_over_tau`, `samples_per_tau`, `median`, `q25`, `q75` and `sd` of p_hat, one row per (d, window, rate), and each window T/tau is one line in its Reds shade.
    The labels name the estimate 1/nu_hat, never p.
    """
    all_rates = np.unique(np.concatenate([rates(block) for _, block in cells]))
    fig, axes = plt.subplots(2 * len(cells), len(dims), figsize=(S.WIDTH, 1.12 * 2 * len(cells) + 0.95), layout="constrained", sharex=True, sharey="row", squeeze=False)
    letters = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    drawn_windows = set()
    for r, (p, block) in enumerate(cells):
        error_row, sd_row = axes[2 * r], axes[2 * r + 1]
        row_sds = []
        for c, d in enumerate(dims):
            at_d = block[IO.numeric(block["d"]) == d]
            for window, chunk in at_d.groupby(WINDOW):
                chunk = chunk.sort_values(RATE)
                x = IO.numeric(chunk[RATE])
                colour = S.col_T(window)
                drawn_windows.add(float(window))
                error_row[c].fill_between(x, IO.numeric(chunk["q25"]) - p, IO.numeric(chunk["q75"]) - p, color=colour, alpha=0.16, lw=0, zorder=1)
                error_row[c].plot(x, IO.numeric(chunk["median"]) - p, color=colour, lw=1.1, marker="o", ms=2.2, mfc="white", mew=0.6, zorder=3)
                sd_row[c].plot(x, IO.numeric(chunk["sd"]), color=colour, lw=1.1, marker="o", ms=2.2, mfc="white", mew=0.6, zorder=3)
                row_sds.append(IO.numeric(chunk["sd"]))
            error_row[c].axhline(0.0, color=S.GREY, lw=0.9, zorder=0)
        # Letters run along each row of panels, the error row before the sd row.
        for c, d in enumerate(dims):
            S.panel_label(error_row[c], next(letters), truth_title(p, d))
        for c in range(len(dims)):
            S.panel_label(sd_row[c], next(letters), "")
        if row_sds:
            plain_log_axis(sd_row[0], "y", np.concatenate(row_sds))
        error_row[0].set_ylabel(ERROR_LABEL, labelpad=2)
        sd_row[0].set_ylabel(SD_LABEL, labelpad=2)
    for ax in axes[-1]:
        rate_axis(ax, all_rates)
        ax.set_xlabel(RATE_AXIS, labelpad=2)
    handles = [Line2D([], [], color=S.col_T(w), lw=1.2, label=window_text(w)) for w in sorted(drawn_windows)]
    handles += [Line2D([], [], color=S.KEY, lw=1.1, marker="o", ms=2.2, mfc="white", mew=0.6, label="median error (upper row), sd (lower row)"), Patch(facecolor=S.GREY, alpha=0.3, label="interquartile range of the error"), Line2D([], [], color=S.GREY, lw=0.9, label="no error")]
    handles += [Line2D([], [], color="none", label=note) for note in notes]
    key_below(fig, handles, ncols=4, clear_stamp=True)
    S.source_label(fig, label)
    return fig


def heavy_flag(table):
    """The heavy-end indicator, from the flag column when present, else from the status."""
    if "heavy_boundary" in table.columns:
        return IO.flag(table["heavy_boundary"])
    if "status" in table.columns:
        return IO.text(table["status"]) == "heavy_boundary"
    return np.zeros(len(table), dtype=bool)


def _spread(recovery, spread_law, label):
    """`phase7_recovery_spread`: the relative spread of p_hat against T/tau, and the spread-law constant per dimension."""
    recovery = IO.ok_units(recovery)
    block = recovery[IO.numeric(recovery["p_true"]) > 0]
    if block.empty:
        return None
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(S.WIDTH, 2.8), layout="constrained")
    grouped = block.groupby(["d", WINDOW, "p_true"])["p_hat"].std(ddof=1).reset_index()
    grouped["relative"] = grouped["p_hat"] / IO.numeric(grouped["p_true"])
    curve = grouped.groupby(["d", WINDOW])["relative"].median().reset_index()
    for d, chunk in curve.groupby("d"):
        chunk = chunk.sort_values(WINDOW)
        axA.plot(IO.numeric(chunk[WINDOW]), IO.numeric(chunk["relative"]), color=S.col_d(d), lw=1.0)
    values = windows(block)
    if values.size >= 2:
        anchor = float(curve["relative"].median()) * np.sqrt(float(values.min()))
        guide = np.array([values.min(), values.max()])
        axA.plot(guide, anchor / np.sqrt(guide), color=S.GREY, lw=0.9, ls=(0, (3, 2)), zorder=0)
        axA.annotate(r"$(T/\tau)^{-1/2}$", (guide[-1], anchor / np.sqrt(guide[-1])), textcoords="offset points", xytext=(3, 2), fontsize=7, color=S.GREY)
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlabel(WINDOW_AXIS, labelpad=2)
    axA.set_ylabel(r"$\mathrm{sd}(1/\hat\nu)\,/\,(1/\nu)$", labelpad=2)
    S.panel_label(axA, "A", "relative spread")

    if spread_law is not None and "d" in spread_law.columns:
        constant = IO.pick(spread_law, "spread_constant", "constant", "spread_law", "c")
        low, high = IO.pick(spread_law, "q25"), IO.pick(spread_law, "q75")
        if constant is not None:
            dims = IO.numeric(spread_law["d"])
            if low is not None and high is not None:
                axB.fill_between(dims, IO.numeric(low), IO.numeric(high), color=S.GREY, alpha=0.20, lw=0, label="interquartile range over cells")
            axB.plot(dims, IO.numeric(constant), color=S.KEY, lw=1.0, zorder=2)
            axB.scatter(dims, IO.numeric(constant), s=9, c=[S.col_d(d) for d in dims], lw=0, zorder=3)
            axB.set_ylabel(r"spread law, $\mathrm{iqr}(1/\hat\nu)\,\sqrt{T/\tau}\,/\,(1/\nu)$", labelpad=2)
            S.dimension_axis(axB, dims)
        else:
            print("  [skip] phase7_recovery_spread panel B: spread_law_by_d_s.csv carries no spread_constant column")
    S.panel_label(axB, "B", "spread law per dimension")

    handles = [
        Line2D([], [], color=S.col_d(11), lw=1.0, label="one line per dimension $d$"),
        Line2D([], [], color=S.GREY, lw=0.9, ls=(0, (3, 2)), label=r"$(T/\tau)^{-1/2}$"),
        Patch(facecolor=S.GREY, alpha=0.20, label="interquartile range over cells"),
        Line2D([], [], color="none", label="numerically successful replicates only"),
    ]
    S.legend_below(fig, handles, ncols=4, reserve=0.17)
    S.dimension_colorbar(fig, [axA, axB])
    S.source_label(fig, label)
    return fig


def _bias(log, recovery, label):
    """`phase7_recovery_bias`: the median and mean bias of p_hat against the true shape at the longest window, one line per dimension."""
    recovery = IO.ok_units(recovery)
    block = recovery[IO.numeric(recovery["p_true"]) > 0]
    if block.empty:
        return None
    longest = float(windows(block).max())
    block = block[IO.numeric(block[WINDOW]) == longest]
    fig, axes = plt.subplots(1, 2, figsize=(S.WIDTH, 2.8), layout="constrained")
    for ax, statistic, name, letter in ((axes[0], "median", "median", "A"), (axes[1], "mean", "mean", "B")):
        grouped = getattr(block.groupby(["d", "p_true"])["p_hat"], statistic)().reset_index()
        grouped["bias"] = 100.0 * (IO.numeric(grouped["p_hat"]) - IO.numeric(grouped["p_true"])) / IO.numeric(grouped["p_true"])
        for d, chunk in grouped.groupby("d"):
            chunk = chunk.sort_values("p_true")
            ax.plot(IO.numeric(chunk["p_true"]), IO.numeric(chunk["bias"]), color=S.col_d(d), lw=1.0, marker="o", ms=2.0, mfc="white", mew=0.6)
        ax.axhline(0.0, color=S.GREY, lw=0.9, zorder=0)
        ax.set_xlabel(f"true {SHAPE}", labelpad=2)
        S.panel_label(ax, letter, f"{name} bias, {window_text(longest)}")
        if log:
            ax.set_xscale("log")
    axes[0].set_ylabel(f"bias (per cent of true {SHAPE})", labelpad=2)
    handles = [Line2D([], [], color=S.col_d(11), lw=1.0, marker="o", ms=2.0, mfc="white", mew=0.6, label="one line per dimension $d$"), Line2D([], [], color=S.GREY, lw=0.9, label="no bias"), Line2D([], [], color="none", label="numerically successful replicates only")]
    S.legend_below(fig, handles, ncols=3, reserve=0.16)
    S.dimension_colorbar(fig, list(axes))
    S.source_label(fig, label)
    return fig


def _gaussian_truth(log, recovery, label):
    """`phase7_recovery_gaussian_truth`: at a Gaussian truth, the size of the estimate and the fraction exactly at zero, by dimension and window."""
    recovery = IO.ok_units(recovery)
    block = recovery[IO.numeric(recovery["p_true"]) == 0]
    if block.empty:
        print("  [skip] phase7_recovery_gaussian_truth: no p_true = 0 cell")
        return None
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(S.WIDTH, 2.8), layout="constrained")
    for window, chunk in block.groupby(WINDOW):
        by_d = chunk.groupby("d")["p_hat"]
        dims = np.asarray(list(by_d.groups), dtype=float)
        axA.plot(dims, by_d.quantile(0.95).to_numpy(dtype=float), color=S.col_T(window), lw=1.1)
        axA.plot(dims, by_d.median().to_numpy(dtype=float), color=S.col_T(window), lw=1.1, ls=(0, (2, 1.6)))
        zero = chunk.assign(_flag=(IO.numeric(chunk["p_hat"]) == 0).astype(float))
        axB.plot(dims, zero.groupby("d")["_flag"].mean().to_numpy(dtype=float), color=S.col_T(window), lw=1.1, marker="o", ms=2.2, mfc="white", mew=0.6)
    axA.set_ylabel(f"{ESTIMATE} at a Gaussian truth", labelpad=2)
    S.dimension_axis(axA, IO.numeric(block["d"]))
    S.panel_label(axA, "A", "size of the estimate")
    axB.set_ylabel(r"fraction at $1/\hat\nu = 0$", labelpad=2)
    S.dimension_axis(axB, IO.numeric(block["d"]))
    S.panel_label(axB, "B", "mass on the truth")
    if log:
        S.log_scale(axA, "y", IO.numeric(block["p_hat"]))
    handles = window_keys(block) + [Line2D([], [], color=S.KEY, lw=1.1, label="95th percentile"), Line2D([], [], color=S.KEY, lw=1.1, ls=(0, (2, 1.6)), label="median"), Line2D([], [], color="none", label="numerically successful replicates only")]
    S.legend_below(fig, handles, ncols=min(len(handles), 5), reserve=0.19)
    S.source_label(fig, label)
    return fig


def _heavy(log, recovery, label):
    """`phase7_recovery_heavy`: the fraction of fits at the heavy end against dimension and, at the longest window, against the true shape."""
    flag = heavy_flag(recovery)
    if not flag.any():
        print("  [skip] phase7_recovery_heavy: no replicate reached the heavy end")
        return None
    work = recovery.assign(_heavy=flag.astype(float))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(S.WIDTH, 2.8), layout="constrained")
    for window, chunk in work.groupby(WINDOW):
        by_d = chunk.groupby("d")["_heavy"].mean()
        axA.plot(np.asarray(list(by_d.index), dtype=float), by_d.to_numpy(dtype=float), color=S.col_T(window), lw=1.1, marker="o", ms=2.2, mfc="white", mew=0.6)
    axA.set_ylabel("fraction at the heavy end", labelpad=2)
    S.dimension_axis(axA, IO.numeric(recovery["d"]))
    S.panel_label(axA, "A", "against dimension")

    longest = float(windows(work).max())
    block = work[IO.numeric(work[WINDOW]) == longest]
    for d, chunk in block.groupby("d"):
        by_p = chunk.groupby("p_true")["_heavy"].mean()
        axB.plot(np.asarray(list(by_p.index), dtype=float), by_p.to_numpy(dtype=float), color=S.col_d(d), lw=1.0)
    axB.set_xlabel(f"true {SHAPE}", labelpad=2)
    axB.set_ylabel("fraction at the heavy end", labelpad=2)
    S.panel_label(axB, "B", f"against the truth, {window_text(longest)}")
    if log:
        S.log_scale(axA, "y", work["_heavy"].to_numpy(dtype=float))
        positive = IO.numeric(block["p_true"])[IO.numeric(block["p_true"]) > 0]
        axB.set_xscale("symlog", linthresh=float(np.min(positive)) if positive.size else 0.01)
    handles = window_keys(work) + [Line2D([], [], color=S.col_d(11), lw=1.0, label="one line per dimension $d$"), Line2D([], [], color="none", label="every replicate, numerical failures included")]
    S.legend_below(fig, handles, ncols=min(len(handles), 5), reserve=0.19)
    S.source_label(fig, label)
    return fig


def _sampling(summary, label):
    """`phase7_recovery_sampling`: the representative cells of the recovery summary across the rates.

    The summary is formed over the numerically successful replicates of each cell (`simulationSummaries`), so it is read as it stands.
    """
    if summary is None or not IO.has(summary, "p_true", "d", WINDOW, RATE, "median_bias", "q25", "q75", "sd"):
        print(f"  [skip] phase7_recovery_sampling: no recovery_summary.csv with p_true, d, {WINDOW}, {RATE}, median_bias, q25, q75 and sd")
        return None
    if rates(summary).size < 2:
        print("  [skip] phase7_recovery_sampling: the recovery grid carries one rate")
        return None
    p_all, d_all = IO.numeric(summary["p_true"]), IO.numeric(summary["d"])
    truths = nearest(TRUTHS, sorted(set(p_all[np.isfinite(p_all)])), log=True)
    dims = nearest(DIMENSIONS, sorted(set(d_all[np.isfinite(d_all)])), log=False)
    work = summary.assign(median=IO.numeric(summary["median_bias"]) + p_all)
    cells = [(p, work[p_all == p]) for p in truths]
    notes = ["numerically successful replicates only", "nested rates: every rate reads the same paths"]
    if "reps" in summary.columns and np.isfinite(IO.numeric(summary["reps"])).any():
        notes.append(f"{int(np.nanmax(IO.numeric(summary['reps'])))} replicates per cell")
    return rate_grid(cells, dims, label, notes)


def render(source, out_dir):
    """Draw every recovery figure, the per-rate ones once per rate, and return the saved paths."""
    recovery = IO.read_table(source, "recovery_long.csv")
    if recovery is None:
        return []
    IO.require(recovery, "d", WINDOW, RATE, "p_true", "p_hat")
    label = IO.phase_paths(source)["label"]
    spread_law = IO.read_table(source, "spread_law_by_d_s.csv")
    # Each quantitative panel filters to the reporting population itself, so a panel called directly draws the same thing this driver draws.
    # The heavy-end panel is a rate over every attempted fit and keeps the failures.
    # Every figure but the last is drawn at one rate, once per rate.
    paths = []
    for rate, block in by_rate(recovery):
        suffix, stamp = rate_suffix(rate), rate_label(label, rate)
        figure = _spread(block, at_rate(spread_law, rate), stamp)
        if figure is not None:
            paths += S.savefig_pair(figure, out_dir, "phase7_recovery_spread" + suffix)
        paths += S.log_twin(_bias, out_dir, "phase7_recovery_bias" + suffix, block, stamp)
        paths += S.log_twin(_gaussian_truth, out_dir, "phase7_recovery_gaussian_truth" + suffix, block, stamp)
        paths += S.log_twin(_heavy, out_dir, "phase7_recovery_heavy" + suffix, block, stamp)
    figure = _sampling(IO.read_table(source, "recovery_summary.csv"), label)
    if figure is not None:
        paths += S.savefig_pair(figure, out_dir, "phase7_recovery_sampling")
    return paths
