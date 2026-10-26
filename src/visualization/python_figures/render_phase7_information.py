"""Phase 7, how the estimator's error reaches the information differences.

  phase7_information_estimated_vs_true_sXX  Per measure, the estimated correction against the true one, one point per replicate, coloured by dimension, with the equality line.
  phase7_information_relative_error_sXX     Per measure, the signed median relative error against the observation window T/tau on a symmetric-log axis, and its absolute value on a log axis with the (T/tau)^-1/2 guide, for the correction and the full measure.
  phase7_information_components_sXX         Per measure, how much of the full-measure estimation error at the longest window comes from the shape alone, from the correlation alone, and from their interaction.
  phase7_information_floor_sXX              Per measure, the 95th and 99th percentiles of the fitted correction at a Gaussian truth (the floor a measured difference has to clear, one line per window T/tau) and, below them, the same percentiles of the full-measure estimation error (`floor_by_d_T_s.csv`).

Every figure is drawn at one sampling rate and written once per rate the tables carry, with the suffix `_sXX` of `rate_suffix` (`_s05`, ..., `_s160`) and the rate under the source stamp.
The rates are nested, so each one reads the same paths less or more often.

Two quantities are never mixed inside one panel.
`<M>_delta_*` is the t-minus-Gaussian correction at the fitted shape, a function of (d, p) alone and the quantity the resolution floor is defined on.
`<M>_full_*` is the whole measure, that correction plus the Gaussian value of the fitted correlation, and `<M>_err*` is its estimation error, which carries the error in the correlation as well.
A panel says which of the two it draws, in its axis label and in its key.

At a Gaussian truth the true correction is exactly zero, so the estimated-versus-true panel puts both axes on one symmetric-log scale (the same linear window on each, which is what keeps the equality line straight) instead of dropping those replicates.

Only replicates with `numerical_ok` are drawn, because these are quantitative panels and the scientific summaries are computed over that same population.

The error components are one object each and are separated by line style, keyed in near-black, because colour carries dimension in the estimated-versus-true and relative-error panels.
The window T/tau is the amount of data of this study (n = T/tau x samples per tau rows) and takes the Reds ramp `_style.col_T`.
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
from render_phase7_shape_recovery import WINDOW, WINDOW_AXIS, at_rate, rate_label, rate_suffix, rates, window_text  # noqa: E402

plt = S.style()
from matplotlib.lines import Line2D  # noqa: E402

# The additive decomposition of the full-measure error, as Phase 7 writes it.
COMPONENTS = [("err", "-", "total"), ("err_p_only", (0, (3, 2)), "shape only"), ("err_R_only", (0, (1, 1.4)), "correlation only"), ("err_interaction", (0, (5, 1, 1, 1)), "interaction")]

CORRECTION_LABEL = "correction, $t$ minus Gaussian"
FULL_LABEL = "full measure"


def present_measures(recovery):
    """The measures whose true and estimated corrections the table carries, with at least one finite truth."""
    return [m for m in S.MEASURES if f"{m}_delta_true" in recovery.columns and f"{m}_delta_hat" in recovery.columns and np.isfinite(IO.numeric(recovery[f"{m}_delta_true"])).any()]


def _estimated_vs_true(recovery, label):
    """`phase7_information_estimated_vs_true`: the estimated against the true correction, per measure."""
    recovery = IO.ok_units(recovery)
    measures = present_measures(recovery)
    if not measures:
        print("  [skip] phase7_information_estimated_vs_true: recovery_long.csv carries no *_delta_true / *_delta_hat pair")
        return None
    dims = IO.numeric(recovery["d"])
    fig, axes = plt.subplots(1, len(measures), figsize=(S.WIDTH, 2.7), layout="constrained", squeeze=False)
    axes = axes.ravel()
    for index, measure in enumerate(measures):
        ax = axes[index]
        truth = IO.numeric(recovery[f"{measure}_delta_true"])
        estimate = IO.numeric(recovery[f"{measure}_delta_hat"])
        keep = np.isfinite(truth) & np.isfinite(estimate)
        ax.scatter(truth[keep], estimate[keep], s=1.4, lw=0, alpha=0.25, c=dims[keep], cmap=S.cmap_d, norm=S.norm_d, rasterized=True, zorder=2)
        # One scale for both axes: the same linear window on each is what keeps the equality line straight, and the Gaussian-truth cells sit at an exact zero.
        # The window ends at the smallest positive true correction of the design, so those cells get a place of their own instead of being pressed onto the axis.
        # The spread of the estimate inside that window is what the floor figure reports.
        span = np.concatenate([truth[keep], estimate[keep]])
        design = truth[keep][truth[keep] > 0]
        window = float(design.min()) if design.size else None
        S.log_scale(ax, "x", span, linthresh=window)
        S.log_scale(ax, "y", span, linthresh=window)
        if span.size:
            line = [float(span.min()), float(span.max())]
            ax.plot(line, line, color=S.GREY, lw=0.8, ls=(0, (3, 2)), zorder=3)
        ax.set_xlabel("true correction (nats)", labelpad=2)
        S.panel_label(ax, string.ascii_uppercase[index], S.MEASURE_LABEL[measure])
        if index == 0:
            ax.set_ylabel("estimated correction (nats)", labelpad=2)
    handles = [Line2D([], [], color=S.KEY, marker="o", ls="none", ms=2.6, label="one replicate, $t$ minus Gaussian correction"), Line2D([], [], color=S.GREY, lw=0.8, ls=(0, (3, 2)), label="equality")]
    S.legend_below(fig, handles, ncols=2, reserve=0.17)
    S.dimension_colorbar(fig, list(axes))
    S.source_label(fig, label)
    return fig


def _relative_error(summary, label):
    """The two relative errors of `recovery_summary.csv`: signed on top, magnitude below.

    The summary's own columns are used rather than a recomputation from the long table, because they are the reported numbers and are already formed over the numerically successful replicates of each cell.
    Each is a signed median error over the magnitude of its own truth (`localRelative` in `simulationSummaries.m`), so the top row is a symmetric-log axis that keeps an underestimate on the figure, where a log axis would silently drop every negative value.
    The bottom row draws the absolute value of the same per-cell statistic on a log axis, which is where the (T/tau)^-1/2 rate is read.
    Cells are pooled over the true p by their median, in both rows.
    """
    if summary is None:
        return None
    measures = [m for m in S.MEASURES if f"{m}_delta_relative_error" in summary.columns]
    if not measures or WINDOW not in summary.columns or "d" not in summary.columns:
        print("  [skip] phase7_information_relative_error: recovery_summary.csv carries no *_delta_relative_error column beside (d, T_over_tau)")
        return None
    block = summary[IO.numeric(summary["p_true"]) > 0] if "p_true" in summary.columns else summary
    if block.empty:
        return None
    fig, axes = plt.subplots(2, len(measures), figsize=(S.WIDTH, 4.9), layout="constrained", squeeze=False, sharex="col")
    letters = list(string.ascii_uppercase)
    for index, measure in enumerate(measures):
        signed_ax, magnitude_ax = axes[0][index], axes[1][index]
        signed, magnitude = [], []
        for column, dashes in ((f"{measure}_delta_relative_error", "-"), (f"{measure}_full_relative_error", (0, (3, 2)))):
            if column not in block.columns:
                continue
            value = IO.numeric(block[column])
            work = block.assign(_signed=value, _magnitude=np.abs(value))
            for d, chunk in work.groupby("d"):
                style = dict(color=S.col_d(d), lw=1.0, ls=dashes, marker="o", ms=1.8, mfc="white", mew=0.5)
                for ax, statistic, drawn in ((signed_ax, "_signed", signed), (magnitude_ax, "_magnitude", magnitude)):
                    curve = chunk.groupby(WINDOW)[statistic].median()
                    ax.plot(np.asarray(list(curve.index), dtype=float), curve.to_numpy(dtype=float), **style)
                    drawn.append(curve.to_numpy(dtype=float))
        signed = np.concatenate(signed) if signed else np.array([])
        magnitude = np.concatenate(magnitude) if magnitude else np.array([])
        # The signed row is always symmetric-log, because zero, the no-error line, is part of its meaning.
        # The linear window ends at the smallest nonzero error drawn, so every other point sits on the logarithmic part and only an exact zero is compressed.
        nonzero = np.abs(signed[np.isfinite(signed) & (signed != 0)])
        S.log_scale(signed_ax, "y", np.append(signed, 0.0), linthresh=float(nonzero.min()) if nonzero.size else None)
        signed_ax.axhline(0.0, color=S.GREY, lw=0.9, zorder=0)
        S.log_scale(magnitude_ax, "y", magnitude)
        window_values = np.unique(IO.numeric(block[WINDOW]))
        finite = magnitude[np.isfinite(magnitude) & (magnitude > 0)]
        if window_values.size >= 2 and finite.size:
            anchor = float(np.median(finite)) * np.sqrt(float(window_values.min()))
            guide = np.array([window_values.min(), window_values.max()])
            magnitude_ax.plot(guide, anchor / np.sqrt(guide), color=S.GREY, lw=0.9, ls=(0, (3, 2)), zorder=0)
        for row, ax in enumerate((signed_ax, magnitude_ax)):
            ax.set_xscale("log")
            S.panel_label(ax, letters[row * len(measures) + index], S.MEASURE_LABEL[measure] if row == 0 else "magnitude")
        magnitude_ax.set_xlabel(WINDOW_AXIS, labelpad=2)
        if index == 0:
            signed_ax.set_ylabel("median relative error\n(signed)", labelpad=2)
            magnitude_ax.set_ylabel("absolute median\nrelative error", labelpad=2)
    handles = [
        Line2D([], [], color=S.col_d(11), lw=1.0, label="one line per dimension $d$"),
        Line2D([], [], color=S.KEY, lw=1.0, label=CORRECTION_LABEL),
        Line2D([], [], color=S.KEY, lw=1.0, ls=(0, (3, 2)), label=FULL_LABEL),
        Line2D([], [], color=S.GREY, lw=0.9, label="no error (signed row)"),
        Line2D([], [], color=S.GREY, lw=0.9, ls=(0, (3, 2)), label=r"$(T/\tau)^{-1/2}$ (magnitude row)"),
    ]
    S.legend_below(fig, handles, ncols=3, reserve=0.12)
    S.dimension_colorbar(fig, list(axes.ravel()))
    S.source_label(fig, label)
    return fig


def _components(log, recovery, label):
    """`phase7_information_components`: the median absolute full-measure error and its components against dimension, at the longest window."""
    recovery = IO.ok_units(recovery)
    measures = [m for m in present_measures(recovery) if all(f"{m}_{column}" in recovery.columns for column, _, _ in COMPONENTS)]
    if not measures:
        print("  [skip] phase7_information_components: recovery_long.csv carries no *_err / *_err_p_only / *_err_R_only / *_err_interaction columns")
        return None
    longest = float(np.nanmax(IO.numeric(recovery[WINDOW])))
    block = recovery[(IO.numeric(recovery[WINDOW]) == longest) & (IO.numeric(recovery["p_true"]) > 0)]
    if block.empty:
        return None
    fig, axes = plt.subplots(1, len(measures), figsize=(S.WIDTH, 2.7), layout="constrained", squeeze=False)
    axes = axes.ravel()
    for index, measure in enumerate(measures):
        ax = axes[index]
        work = block.copy()
        drawn = []
        for column, dashes, name in COMPONENTS:
            work["_error"] = np.abs(IO.numeric(block[f"{measure}_{column}"]))
            curve = work.groupby("d")["_error"].median()
            ax.plot(np.asarray(list(curve.index), dtype=float), curve.to_numpy(dtype=float), color=S.KEY, lw=1.0, ls=dashes, marker="o", ms=1.8, mfc="white", mew=0.5, label=name)
            drawn.append(curve.to_numpy(dtype=float))
        S.dimension_axis(ax, IO.numeric(recovery["d"]))
        S.panel_label(ax, string.ascii_uppercase[index], S.MEASURE_LABEL[measure])
        if index == 0:
            ax.set_ylabel(f"full-measure estimation error (nats)\nmedian over replicates, {window_text(longest)}", labelpad=2)
        if log:
            # The decomposition is exactly additive, so the interaction is zero up to rounding.
            # A rounding-level median is read as the zero it is, which puts the axis on symmetric log instead of stretching it over fifteen decades.
            values = np.concatenate(drawn)
            finite = values[np.isfinite(values)]
            floor = 1e-12 * float(np.max(np.abs(finite))) if finite.size else 0.0
            S.log_scale(ax, "y", np.where(np.abs(values) <= floor, 0.0, values))
    handles = axes[0].get_legend_handles_labels()[0]
    S.legend_below(fig, handles, ncols=4, reserve=0.17)
    S.source_label(fig, label)
    return fig


def _floor(log, floors, label):
    """The Gaussian-truth floor: the correction on top, the full-measure error below."""
    if floors is None:
        return None
    IO.require(floors, "d", WINDOW, "measure")
    rows = [("correction at a Gaussian truth (nats)", CORRECTION_LABEL, (("q95", "-"), ("q99", (0, (3, 2))))), ("full-measure error (nats)", FULL_LABEL, (("q95_full_err", "-"), ("q99_full_err", (0, (3, 2)))))]
    rows = [row for row in rows if any(IO.pick(floors, column) is not None for column, _ in row[2])]
    if not rows:
        print("  [skip] phase7_information_floor: floor_by_d_T_s.csv carries no q95 / q99 or q95_full_err / q99_full_err column")
        return None
    measures = [m for m in S.MEASURES if (IO.text(floors["measure"]) == m).any()]
    if not measures:
        return None
    fig, axes = plt.subplots(len(rows), len(measures), figsize=(S.WIDTH, 2.3 * len(rows) + 0.4), layout="constrained", squeeze=False, sharex=True)
    letters = list(string.ascii_uppercase)
    for row_index, (ylabel, name, columns) in enumerate(rows):
        for index, measure in enumerate(measures):
            ax = axes[row_index][index]
            block = floors[IO.text(floors["measure"]) == measure]
            drawn = []
            for column, dashes in columns:
                values = IO.pick(block, column)
                if values is None:
                    continue
                work = block.assign(_value=IO.numeric(values))
                for window, chunk in work.groupby(WINDOW):
                    chunk = chunk.sort_values("d")
                    ax.plot(IO.numeric(chunk["d"]), IO.numeric(chunk["_value"]), color=S.col_T(window), lw=1.1, ls=dashes, marker="o", ms=2.2, mfc="white", mew=0.6)
                drawn.append(work["_value"].to_numpy(dtype=float))
            S.dimension_axis(ax, IO.numeric(floors["d"]), label=(row_index == len(rows) - 1))
            S.panel_label(ax, letters[row_index * len(measures) + index], S.MEASURE_LABEL[measure] if row_index == 0 else name)
            if index == 0:
                ax.set_ylabel(ylabel, labelpad=2)
            if log and drawn:
                S.log_scale(ax, "y", np.concatenate(drawn))
    handles = [Line2D([], [], color=S.col_T(window), lw=1.2, label=window_text(window)) for window in np.unique(IO.numeric(floors[WINDOW]))]
    handles += [Line2D([], [], color=S.KEY, lw=1.1, label="95th percentile"), Line2D([], [], color=S.KEY, lw=1.1, ls=(0, (3, 2)), label="99th percentile")]
    S.legend_below(fig, handles, ncols=min(len(handles), 5), reserve=0.16)
    S.source_label(fig, label)
    return fig


def render(source, out_dir):
    """Draw every information figure once per sampling rate, and return the saved paths."""
    recovery = IO.read_table(source, "recovery_long.csv")
    summary = IO.read_table(source, "recovery_summary.csv")
    floors = IO.read_table(source, "floor_by_d_T_s.csv")
    label = IO.phase_paths(source)["label"]
    # The rates are those of the long table, or of the summary when a run kept no long one.
    frame = recovery if recovery is not None else summary
    if frame is None:
        return []
    paths = []
    for rate in rates(frame):
        suffix, stamp = rate_suffix(rate), rate_label(label, rate)
        block = at_rate(recovery, rate)
        if block is not None:
            figure = _estimated_vs_true(block, stamp)
            if figure is not None:
                paths += S.savefig_pair(figure, out_dir, "phase7_information_estimated_vs_true" + suffix)
            paths += S.log_twin(_components, out_dir, "phase7_information_components" + suffix, block, stamp)
        figure = _relative_error(at_rate(summary, rate), stamp)
        if figure is not None:
            paths += S.savefig_pair(figure, out_dir, "phase7_information_relative_error" + suffix)
        paths += S.log_twin(_floor, out_dir, "phase7_information_floor" + suffix, at_rate(floors, rate), stamp)
    return paths
