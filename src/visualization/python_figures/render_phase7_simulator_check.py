"""Phase 7, the simulator check: does the simulated process do what the study needs?

  phase7_check_law              Is the one-time law of the simulated paths the Student law it should be?
                                Per nu, the Kolmogorov-Smirnov distance of the radial beta-PIT and of the first channel's t PIT from uniform, at time 0 and after the law horizon, against the Bonferroni critical value over the family of every Metropolis distance of the check (the reading rule) beside the single-test 95% and 99% values, and the direction moment D E[U_1^2] - 1 with its 95% interval, from the law part's rows.
  phase7_check_coarse           Can the check see the failure the Metropolis correction guards against?
                                At the forced coarse step of the heavy corner, per parent dimension D: the radial PIT histogram after the horizon of the Metropolis chain and of the frozen chain (the negative control, expected to fail), and every radial and marginal KS distance of the coarse part against the critical values.
  phase7_check_refinement       Does refining the integration step move p_hat?
                                Per nu and step pair (f against 2f), the pooled standardized difference of the mean p_hat with its 95% interval against the +-tolerance band, and the pooled sd ratio coarse / fine with its bootstrap interval (`check_refinement_pooled.csv`).
                                Below, per nu, the chain's clock error from the exact conditional mean at each time and refine factor (`check_relaxation.csv`).
  phase7_check_discretisation   How far can the clock error move p_hat?
                                Per nu, the mean shift it implies in sd of p_hat (`mean_shift_over_sd`), by dimension, per window T/tau and refine factor, against the tolerance.
                                The `_log` twin puts it on a logarithmic axis.
  phase7_check_timescale        What does the Phase 6 estimator read off the model's own paths over the lag windows the recordings span?
                                tau_env, tau_amp, its Gaussian reference tau_amp_ref and the amplitude excess tau_amp / tau_amp_ref against the lag window, one line per nu, beside their Gaussian values, with the integrated excess tau_amp_int / tau_amp_ref_int beside the fitted one when the table carries it.
  phase7_check_sampling         Does the estimate still change with more samples per tau at a fixed window?
                                Per nu (row pairs) and d (columns), the median error of p_hat with its interquartile band and its sd against the samples per tau, one line per window T/tau.

The check reports: nothing drawn here is a gate, and no flag column of the tables is read as a verdict, because the distances are drawn against the critical values instead.
Every row of `check_law.csv` carries the Bonferroni value of the family, whose size counts the Metropolis distances only.
The frozen rows are outside the family and are read against its value, so the coarse figure draws it over their columns as well.
The shape is written 1/nu, never p.
The degrees of freedom nu sit on a categorical axis where they are the abscissa (1/3, 1, 2, 4, 10, infinity), and take the categorical ramp where one line is drawn per nu.
The window T/tau takes the Reds ramp `_style.col_T`, and a dimension its viridis colour.
An interval is drawn with its arms clipped at zero, so a point outside its own interval is drawn instead of stopping the renderer.
Every reference (critical values, the tolerance band, the zero line, the Gaussian values, the uniform PIT density) is grey, and the other distinctions are marker shapes, marker fills and line styles keyed in near-black.
"""

from __future__ import annotations

import math
import os
import string
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _phase_io as IO  # noqa: E402
import _style as S  # noqa: E402
from render_phase7_shape_recovery import ESTIMATE, key_below, nu_text, plain_log_axis, rate_grid, window_text  # noqa: E402

plt = S.style()
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REFINE_STYLE = {1: "-", 2: (0, (3, 2)), 4: (0, (1, 1.4)), 8: (0, (5, 1, 1, 1))}
PAIR_MARKER = {(1, 2): "o", (2, 4): "s", (4, 8): "^"}
# The single-test critical values are context; the Bonferroni value is the reading rule.
DOTS = (0, (1, 1.4))
DASH = (0, (3, 2))
SCHEME_LABEL = {"metropolis": "Metropolis", "frozen": "frozen"}
# The refinement tolerance when no table carries it: a tenth of the spread of p_hat.
TOLERANCE_SD = 0.1


def _nu_axis(ax, values):
    """A categorical nu axis: positions 0..k-1 in ascending nu, labelled by value."""
    order = np.sort(np.unique(np.asarray(values, dtype=float)))
    ax.set_xticks(range(order.size))
    ax.set_xticklabels([nu_text(v) for v in order])
    ax.set_xlim(-0.6, order.size - 0.4)
    ax.set_xlabel(r"degrees of freedom $\nu$", labelpad=2)
    return {float(v): k for k, v in enumerate(order)}


def _time_markers(times):
    """Hollow at the first time (the exact start), filled after the horizon."""
    return {t: dict(marker="o", mfc="white" if k == 0 else S.KEY, mec=S.KEY) for k, t in enumerate(times)}


def _time_label(t):
    """The legend label of one reading time."""
    return r"$t = 0$" if t == 0 else rf"$t = {t:g}\,\tau$"


def _arms(y, low, high):
    """The (lower, upper) error arms of an interval, clipped at zero: a point estimate outside its own interval (a percentile bootstrap does not bound it) is drawn with a zero arm on that side instead of making `errorbar` raise."""
    return [np.clip(y - low, 0.0, None), np.clip(high - y, 0.0, None)]


def _median(frame, column):
    """The median of a column's finite values, NaN when the column is absent or holds none."""
    values = IO.numeric(frame[column]) if column in frame.columns else np.array([])
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else np.nan


def _rows(law, part, scheme=None):
    """The rows of one part (and scheme) of `check_law.csv`.

    A table without the `part` and `scheme` columns is read as the law part's Metropolis rows throughout.
    """
    keep = np.ones(len(law), dtype=bool)
    if "part" in law.columns:
        keep &= IO.text(law["part"]) == part
    elif part != "law":
        keep[:] = False
    if scheme is not None and "scheme" in law.columns:
        keep &= IO.text(law["scheme"]) == scheme
    return law[keep]


def _critical_keys(bonferroni, family):
    """Legend keys for the Bonferroni value, when it is finite, and for the single-test 95% and 99% values."""
    keys = []
    if np.isfinite(bonferroni):
        keys.append(Line2D([], [], color=S.GREY, lw=1.1, label="Bonferroni critical value" + (f", family of {int(family)} Metropolis distances" if np.isfinite(family) else "")))
    return keys + [Line2D([], [], color=S.GREY, lw=0.8, ls=DOTS, label="95% critical value"), Line2D([], [], color=S.GREY, lw=0.8, ls=DASH, label="99% critical value")]


def _law(law, label):
    """`phase7_check_law`: the law part's KS distances per nu at both times, and the direction moment."""
    if law is None:
        return None
    law = _rows(law, "law", "metropolis")
    if law.empty:
        print("  [skip] phase7_check_law: check_law.csv carries no law-part row")
        return None
    IO.require(law, "nu", "time_tau", "radial_ks", "ks_critical_95", "ks_critical_99", "marginal_ks", "direction_u1sq_mean", "direction_u1sq_se", "D")
    fig, axes = plt.subplots(1, 3, figsize=(S.WIDTH, 2.6), layout="constrained")
    times = np.sort(np.unique(IO.numeric(law["time_tau"])))
    markers = _time_markers(times)
    bonferroni = _median(law, "ks_critical_bonferroni")
    for index, (column, title) in enumerate((("radial_ks", "radial law"), ("marginal_ks", "first channel"))):
        ax = axes[index]
        where = _nu_axis(ax, IO.numeric(law["nu"]))
        for t in times:
            block = law[IO.numeric(law["time_tau"]) == t]
            x = np.array([where[float(v)] for v in IO.numeric(block["nu"])]) + (0.12 if t > times[0] else -0.12)
            ax.plot(x, IO.numeric(block[column]), ls="none", ms=3.2, mew=0.7, zorder=3, **markers[t])
        for column_crit, dashes in (("ks_critical_95", DOTS), ("ks_critical_99", DASH)):
            ax.axhline(_median(law, column_crit), color=S.GREY, lw=0.8, ls=dashes, zorder=0)
        if np.isfinite(bonferroni):
            ax.axhline(bonferroni, color=S.GREY, lw=1.1, zorder=0)
        ax.set_ylim(bottom=0.0)
        if index == 0:
            ax.set_ylabel("Kolmogorov-Smirnov distance", labelpad=2)
        S.panel_label(ax, string.ascii_uppercase[index], title)

    ax = axes[2]
    where = _nu_axis(ax, IO.numeric(law["nu"]))
    for t in times:
        block = law[IO.numeric(law["time_tau"]) == t]
        D = IO.numeric(block["D"])
        x = np.array([where[float(v)] for v in IO.numeric(block["nu"])]) + (0.12 if t > times[0] else -0.12)
        y = D * IO.numeric(block["direction_u1sq_mean"]) - 1.0
        err = np.abs(1.96 * D * IO.numeric(block["direction_u1sq_se"]))
        ax.errorbar(x, y, yerr=err, ls="none", ms=3.2, mew=0.7, elinewidth=0.7, ecolor=S.KEY, **markers[t])
    ax.axhline(0.0, color=S.GREY, lw=0.9, zorder=0)
    ax.set_ylabel(r"$D\,\mathrm{E}[U_1^2] - 1$", labelpad=2)
    S.panel_label(ax, "C", "direction")

    handles = [Line2D([], [], ls="none", ms=3.2, mew=0.7, label=_time_label(t), **markers[t]) for t in times]
    handles += _critical_keys(bonferroni, _median(law, "family_size"))
    handles.append(Line2D([], [], color=S.KEY, lw=0.7, label="95% interval (panel C)"))
    S.legend_below(fig, handles, ncols=3, reserve=0.20)
    S.source_label(fig, label)
    return fig


def _coarse(law, histogram, label):
    """The coarse part: Metropolis against the frozen chain at a forced coarse step."""
    if law is None:
        return None
    rows = _rows(law, "coarse")
    if rows.empty:
        print("  [skip] phase7_check_coarse: check_law.csv carries no coarse-part row")
        return None
    IO.require(rows, "scheme", "nu", "D", "time_tau", "radial_ks", "marginal_ks", "ks_critical_95", "ks_critical_99")
    scheme_of = IO.text(rows["scheme"])
    schemes = [s for s in SCHEME_LABEL if (scheme_of == s).any()] + sorted(set(scheme_of) - set(SCHEME_LABEL))
    cells = [(D, s) for D in np.unique(IO.numeric(rows["D"])) for s in schemes if ((IO.numeric(rows["D"]) == D) & (scheme_of == s)).any()]
    times = np.sort(np.unique(IO.numeric(rows["time_tau"])))
    horizon = float(times.max())
    pits = None
    if histogram is not None and IO.has(histogram, "part", "scheme", "D", "time_tau", "bin_lower", "bin_upper", "count", "expected"):
        pits = histogram[(IO.text(histogram["part"]) == "coarse") & (IO.numeric(histogram["time_tau"]) == horizon)]
        pits = pits if not pits.empty else None
    grid = math.lcm(len(cells), 2)
    mosaic = [[f"pit{k}" for k in range(len(cells)) for _ in range(grid // len(cells))]] if pits is not None else []
    mosaic.append(["radial"] * (grid // 2) + ["marginal"] * (grid // 2))
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(S.WIDTH, 2.35 * len(mosaic) + 0.75), layout="constrained")
    letters = iter(string.ascii_uppercase)

    if pits is not None:
        for k, (D, scheme) in enumerate(cells):
            ax = axes[f"pit{k}"]
            block = pits[(IO.numeric(pits["D"]) == D) & (IO.text(pits["scheme"]) == scheme)]
            lower, upper = IO.numeric(block["bin_lower"]), IO.numeric(block["bin_upper"])
            density = IO.numeric(block["count"]) / IO.numeric(block["expected"])
            ax.bar(lower, density, width=upper - lower, align="edge", color=S.model_color("empirical"), alpha=0.75, lw=0.3, edgecolor="white")
            ax.axhline(1.0, color=S.GREY, lw=0.9, zorder=3)
            ax.set_xlim(0.0, 1.0)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.set_xlabel("radial PIT", labelpad=2)
            if k == 0:
                ax.set_ylabel("count / expected", labelpad=2)
            at = rows[(IO.numeric(rows["D"]) == D) & (scheme_of == scheme) & (IO.numeric(rows["time_tau"]) == horizon)]
            nonfinite = _median(at, "nonfinite_fraction")
            top = max(float(np.nanmax(density)) if density.size else 1.0, 1.0)
            if np.isfinite(nonfinite) and nonfinite > 0:
                # A diverged state counts as a PIT of 1, so the share of non-finite states is printed in headroom above the bars.
                ax.text(0.04, 0.97, f"non-finite states {100.0 * nonfinite:.3g}%", transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=S.KEY)
                top *= 1.2
            ax.set_ylim(0.0, top * 1.08)
            S.panel_label(ax, next(letters), f"{SCHEME_LABEL.get(scheme, scheme)}, $D = {int(D)}$")

    markers = _time_markers(times)
    positions = {cell: k for k, cell in enumerate(cells)}
    bonferroni = _median(rows[scheme_of == "metropolis"], "ks_critical_bonferroni")
    for key, column, title in (("radial", "radial_ks", "radial law"), ("marginal", "marginal_ks", "first channel")):
        ax = axes[key]
        values = []
        for (D, scheme), k in positions.items():
            cell = rows[(IO.numeric(rows["D"]) == D) & (scheme_of == scheme)]
            for t in times:
                at = cell[IO.numeric(cell["time_tau"]) == t]
                y = IO.numeric(at[column])
                ax.plot(np.full(y.size, k + (0.12 if t > times[0] else -0.12)), y, ls="none", ms=3.4, mew=0.8, zorder=3, **markers[t])
                values.append(y)
            # Each cell's own Bonferroni value, over its own column: the family is the Metropolis distances, and a frozen row carries the family's value when the table writes one there.
            # A table that leaves it empty draws none.
            critical = _median(cell, "ks_critical_bonferroni")
            if np.isfinite(critical):
                ax.hlines(critical, k - 0.4, k + 0.4, color=S.GREY, lw=1.1, zorder=0)
        for column_crit, dashes in (("ks_critical_95", DOTS), ("ks_critical_99", DASH)):
            ax.axhline(_median(rows, column_crit), color=S.GREY, lw=0.8, ls=dashes, zorder=0)
        plain_log_axis(ax, "y", np.concatenate(values + [np.array([bonferroni, _median(rows, "ks_critical_95")])]))
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels([f"{SCHEME_LABEL.get(s, s)}\n$D = {int(D)}$" for D, s in cells])
        ax.set_xlim(-0.6, len(cells) - 0.4)
        if key == "radial":
            ax.set_ylabel("Kolmogorov-Smirnov distance", labelpad=2)
        S.panel_label(ax, next(letters), title)

    nu, h = _median(rows, "nu"), _median(rows, "h")
    handles = [Line2D([], [], ls="none", ms=3.4, mew=0.8, label=_time_label(t), **markers[t]) for t in times]
    handles += _critical_keys(bonferroni, _median(rows, "family_size"))
    if pits is not None:
        handles += [Patch(facecolor=S.model_color("empirical"), alpha=0.75, label=f"radial PIT per bin, {_time_label(horizon)}"), Line2D([], [], color=S.GREY, lw=0.9, label="uniform PIT: the Student law")]
    handles.append(Line2D([], [], color="none", label="frozen: the negative control, outside the family, read against its value"))
    key_below(fig, handles, ncols=2, clear_stamp=True)
    # The coarse part's one truth and forced step are a condition of the whole figure.
    S.source_label(fig, label + (rf"  $\nu = ${nu_text(nu)}, forced step $h = {h:g}\,\tau$" if np.isfinite(nu) and np.isfinite(h) else ""))
    return fig


def _refinement(pooled, relaxation, label):
    """The pooled refinement reading on top, the relaxation clock error below."""
    if pooled is not None:
        IO.require(pooled, "nu", "refine_coarse", "refine_fine", "pooled_difference_over_sd", "ci95_low_over_sd", "ci95_high_over_sd", "tolerance_sd", "sd_ratio_coarse_over_fine", "sd_ratio_ci95_low", "sd_ratio_ci95_high")
    bottom = sorted(np.unique(IO.numeric(relaxation["nu"]))) if relaxation is not None else []
    if pooled is None and not bottom:
        return None
    grid = math.lcm(2, max(len(bottom), 1))
    mosaic = []
    if pooled is not None:
        mosaic.append(["difference"] * (grid // 2) + ["ratio"] * (grid // 2))
    if bottom:
        mosaic.append([f"clock{k}" for k in range(len(bottom)) for _ in range(grid // len(bottom))])
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(S.WIDTH, 2.4 * len(mosaic) + 0.5), layout="constrained")
    letters = iter(string.ascii_uppercase)
    handles = []

    if pooled is not None:
        tolerance = _median(pooled, "tolerance_sd")
        coarse, fine = IO.numeric(pooled["refine_coarse"]), IO.numeric(pooled["refine_fine"])
        pairs = sorted({(int(a), int(b)) for a, b in zip(coarse, fine)})
        offsets = {pair: 0.16 * (k - (len(pairs) - 1) / 2.0) for k, pair in enumerate(pairs)}
        for key, column, low, high, title in (("difference", "pooled_difference_over_sd", "ci95_low_over_sd", "ci95_high_over_sd", f"mean of {ESTIMATE}"), ("ratio", "sd_ratio_coarse_over_fine", "sd_ratio_ci95_low", "sd_ratio_ci95_high", f"spread of {ESTIMATE}")):
            ax = axes[key]
            where = _nu_axis(ax, IO.numeric(pooled["nu"]))
            for pair in pairs:
                block = pooled[(coarse == pair[0]) & (fine == pair[1])]
                x = np.array([where[float(v)] for v in IO.numeric(block["nu"])]) + offsets[pair]
                y = IO.numeric(block[column])
                ax.errorbar(x, y, yerr=_arms(y, IO.numeric(block[low]), IO.numeric(block[high])), ls="none", color=S.KEY, marker=PAIR_MARKER.get(pair, "D"), ms=3.4, mfc="white", mew=0.8, elinewidth=0.8, zorder=3)
            S.panel_label(ax, next(letters), title)
        ax = axes["difference"]
        ax.axhspan(-tolerance, tolerance, color=S.GREY, alpha=0.20, lw=0, zorder=0)
        ax.axhline(0.0, color=S.GREY, lw=0.8, zorder=0)
        ax.set_ylabel(f"pooled change of the mean\n{ESTIMATE}, in sd of the finer step", labelpad=2)
        ax = axes["ratio"]
        ax.axhline(1.0, color=S.GREY, lw=0.8, zorder=0)
        plain_log_axis(ax, "y", np.concatenate([IO.numeric(pooled[c]) for c in ("sd_ratio_coarse_over_fine", "sd_ratio_ci95_low", "sd_ratio_ci95_high")] + [np.array([1.0])]))
        ax.set_ylabel(r"pooled $\mathrm{sd}(1/\hat\nu)$ ratio," + "\ncoarse over fine", labelpad=2)
        handles += [Line2D([], [], ls="none", color=S.KEY, marker=PAIR_MARKER.get(p, "D"), ms=3.4, mfc="white", mew=0.8, label=f"refine $f = {p[0]}$ vs ${p[1]}$") for p in pairs]
        handles += [Patch(facecolor=S.GREY, alpha=0.20, label=rf"$\pm{tolerance:g}$ sd tolerance (panel A)"), Line2D([], [], ls="none", color=S.KEY, marker="|", ms=8, mew=0.8, label="95% interval (bootstrap for the sd ratio)")]

    if bottom:
        IO.require(relaxation, "refine", "t_tau", "clock_error", "clock_error_se")
        refines = sorted(np.unique(IO.numeric(relaxation["refine"])))
        for k, nu in enumerate(bottom):
            ax = axes[f"clock{k}"]
            block = relaxation[IO.numeric(relaxation["nu"]) == nu]
            ax.axhline(0.0, color=S.GREY, lw=0.9, zorder=0)
            for index, refine in enumerate(refines):
                chunk = block[IO.numeric(block["refine"]) == refine].sort_values("t_tau")
                # A small shift in t keeps the three refine factors' intervals apart.
                shift = 1.0 + 0.05 * (index - (len(refines) - 1) / 2.0)
                x = IO.numeric(chunk["t_tau"]) * shift
                # The interval's half-width is 1.96 |se|, a 95% interval whatever the sign of the stored standard error.
                ax.errorbar(x, IO.numeric(chunk["clock_error"]), yerr=np.abs(1.96 * IO.numeric(chunk["clock_error_se"])), color=S.KEY, lw=1.0, ls=REFINE_STYLE.get(int(refine), "-"), marker="o", ms=2.4, mfc="white", mew=0.6, elinewidth=0.6, capsize=0)
            ax.set_xscale("log")
            ticks = np.unique(IO.numeric(block["t_tau"]))
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:g}" for t in ticks])
            ax.minorticks_off()
            ax.set_xlabel(r"time $t/\tau$ from the fixed start", labelpad=2)
            if k == 0:
                ax.set_ylabel(r"clock error $\kappa$" + "\n" + r"(chain time runs $1-\kappa$)", labelpad=2)
            S.panel_label(ax, next(letters), rf"relaxation, $\nu = ${nu_text(nu)}")
        handles += [Line2D([], [], color=S.KEY, lw=1.0, ls=REFINE_STYLE.get(int(f), "-"), label=f"refine $f = {int(f)}$ (lower row)") for f in refines]
    # Two columns: the upper row's keys in the first, the lower row's in the second.
    key_below(fig, handles, ncols=2 if pooled is not None and bottom else 3)
    S.source_label(fig, label)
    return fig


def _discretisation(log, table, tolerance, label):
    """The clock error carried to p_hat, in sd of p_hat, against the tolerance."""
    if table is None:
        return None
    IO.require(table, "nu", "refine", "d", "T_over_tau", "mean_shift_over_sd")
    nus = sorted(np.unique(IO.numeric(table["nu"])))
    fig, axes = plt.subplots(1, len(nus), figsize=(S.WIDTH, 2.7), layout="constrained", sharey=True, squeeze=False)
    values = [np.array([tolerance])]
    for k, nu in enumerate(nus):
        ax = axes[0, k]
        block = table[IO.numeric(table["nu"]) == nu]
        for (window, refine), chunk in block.groupby(["T_over_tau", "refine"]):
            chunk = chunk.sort_values("d")
            y = IO.numeric(chunk["mean_shift_over_sd"])
            ax.plot(IO.numeric(chunk["d"]), y, color=S.col_T(window), lw=1.1, ls=REFINE_STYLE.get(int(refine), "-"), marker="o", ms=2.2, mfc="white", mew=0.6)
            values.append(y)
        ax.axhline(tolerance, color=S.GREY, lw=0.9, ls=DASH, zorder=0)
        S.dimension_axis(ax, IO.numeric(table["d"]))
        S.panel_label(ax, string.ascii_uppercase[k], rf"$\nu = ${nu_text(nu)}")
    axes[0, 0].set_ylabel(f"mean shift of {ESTIMATE} from the clock\nerror, in sd of {ESTIMATE}", labelpad=2)
    if log:
        S.log_scale(axes[0, 0], "y", np.concatenate(values))
    else:
        axes[0, 0].set_ylim(bottom=0.0)
    handles = [Line2D([], [], color=S.col_T(w), lw=1.2, label=window_text(w)) for w in np.unique(IO.numeric(table["T_over_tau"]))]
    handles += [Line2D([], [], color=S.KEY, lw=1.1, ls=REFINE_STYLE.get(int(f), "-"), label=f"refine $f = {int(f)}$") for f in np.unique(IO.numeric(table["refine"]))]
    handles.append(Line2D([], [], color=S.GREY, lw=0.9, ls=DASH, label=f"tolerance, {tolerance:g} sd"))
    S.legend_below(fig, handles, ncols=min(len(handles), 4), reserve=0.20)
    S.source_label(fig, label)
    return fig


# Each entry is (column, floor flag columns, axis label, title, Gaussian value), with every time scale in units of the drift time tau.
# At Gaussian truth (nu = infinity) the score's envelope time is tau, the square's and its reference's tau / 2, and the excess 1.
TIMESCALES = (
    ("tau_env", ("env_floor_reached",), r"$\tau_\mathrm{env}$ ($\tau$)", "signal memory", 1.0),
    ("tau_amp", ("amp_floor_reached",), r"$\tau_\mathrm{amp}$ ($\tau$)", "amplitude memory", 0.5),
    ("tau_amp_ref", ("amp_ref_floor_reached",), r"$\tau_\mathrm{amp,ref}$ ($\tau$)", "its Gaussian reference", 0.5),
    ("amp_excess", ("amp_floor_reached", "amp_ref_floor_reached"), r"$\tau_\mathrm{amp}\,/\,\tau_\mathrm{amp,ref}$", "amplitude excess", 1.0),
)
# Beside the fitted excess when the table carries it: the integrated autocorrelation time of the squared scores over the window read, over the same for its Gaussian reference, which counts a slow component's size and duration together and is 1 at Gaussian truth.
# Its points are hollow on the same flags, where an autocorrelation never falls below the floor within the window.
INTEGRATED = ("amp_excess_int", ("amp_floor_reached", "amp_ref_floor_reached"), r"$\tau_\mathrm{amp,int}\,/\,\tau_\mathrm{amp,ref,int}$", "integrated amplitude excess", 1.0)


def _timescale(timescale, label):
    """`phase7_check_timescale`: the Phase 6 time scales of the model's own paths against the lag window, one line per nu."""
    if timescale is None:
        return None
    IO.require(timescale, "nu", "window_tau", "tau_env", "tau_amp", "tau_amp_ref", "amp_excess")
    nus = sorted(np.unique(IO.numeric(timescale["nu"])))
    windows_tau = np.unique(IO.numeric(timescale["window_tau"]))
    panels = TIMESCALES + ((INTEGRATED,) if INTEGRATED[0] in timescale.columns else ())
    ncols = 2 if len(panels) <= 4 else 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(S.WIDTH, 2.25 * nrows), layout="constrained", sharex=True, squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(panels) :]:
        ax.set_axis_off()
    hollow = False
    for index, (column, flags, ylabel, title, gaussian) in enumerate(panels):
        ax = axes[index]
        drawn = [np.array([gaussian])]
        for k, nu in enumerate(nus):
            block = timescale[IO.numeric(timescale["nu"]) == nu].sort_values("window_tau")
            x, y = IO.numeric(block["window_tau"]), IO.numeric(block[column])
            reached = np.ones(len(block), dtype=bool)
            for flag in flags:
                if flag in block.columns:
                    reached &= IO.flag(block[flag])
            hollow |= bool((~reached).any())
            colour = S.categorical_color(k)
            ax.plot(x, y, color=colour, lw=1.0, zorder=2)
            ax.plot(x[reached], y[reached], ls="none", marker="o", ms=3.0, color=colour, zorder=3)
            ax.plot(x[~reached], y[~reached], ls="none", marker="o", ms=3.0, mec=colour, mfc="white", mew=0.8, zorder=3)
            drawn.append(y)
        ax.axhline(gaussian, color=S.GREY, lw=0.9, ls=DASH, zorder=0)
        plain_log_axis(ax, "y", np.concatenate(drawn))
        ax.set_ylabel(ylabel, labelpad=2)
        ax.set_xscale("log")
        ax.set_xticks(windows_tau)
        ax.set_xticklabels([f"{w:g}" for w in windows_tau])
        ax.minorticks_off()
        if index + ncols >= len(panels):
            # The lowest panel of its column, which may sit above an empty slot.
            ax.set_xlabel(r"lag window read ($\tau$)", labelpad=2)
            ax.xaxis.set_tick_params(labelbottom=True)
        S.panel_label(ax, string.ascii_uppercase[index], title)
    handles = [Line2D([], [], color=S.categorical_color(k), lw=1.0, marker="o", ms=3.0, label=rf"$\nu = ${nu_text(nu)}") for k, nu in enumerate(nus)]
    if hollow:
        handles.append(Line2D([], [], ls="none", marker="o", ms=3.0, mec=S.KEY, mfc="white", mew=0.8, label="an autocorrelation never fell below the floor within the window"))
    handles.append(Line2D([], [], color=S.GREY, lw=0.9, ls=DASH, label=r"Gaussian value ($\nu = \infty$)"))
    notes = []
    for column, template in (("D", "D = {:g}"), ("T_over_tau", r"$T/\tau$ = {:g}"), ("n_paths", "{:g} paths per " + r"$\nu$"), ("samples_per_tau", "{:g} samples per " + r"$\tau$")):
        value = _median(timescale, column)
        if np.isfinite(value):
            notes.append(template.format(value))
    if notes:
        handles.append(Line2D([], [], color="none", label="model: " + ", ".join(notes)))
    S.legend_below(fig, handles, ncols=4, reserve=0.12)
    S.source_label(fig, label)
    return fig


def _sampling(sampling, label):
    """The sampling part: p_hat against the samples per tau, per nu and d."""
    if sampling is None:
        return None
    IO.require(sampling, "nu", "T_over_tau", "samples_per_tau", "d", "median", "q25", "q75", "sd")
    nu = IO.numeric(sampling["nu"])
    truth = np.where(np.isfinite(nu), 1.0 / nu, 0.0)
    cells = [(p, sampling[truth == p]) for p in sorted(np.unique(truth))]
    dims = [int(d) for d in np.unique(IO.numeric(sampling["d"]))]
    notes = ["numerically successful replicates only"]
    if "reps" in sampling.columns:
        notes.append(f"{int(np.nanmax(IO.numeric(sampling['reps'])))} replicates per cell")
    return rate_grid(cells, dims, label, notes)


def _tolerance(*tables):
    """The refinement tolerance in sd of p_hat, from the first table that carries it."""
    for table in tables:
        if table is not None and np.isfinite(_median(table, "tolerance_sd")):
            return _median(table, "tolerance_sd")
    return TOLERANCE_SD


def render(source, out_dir):
    """Draw every simulator-check figure from the check tables, and return the saved paths."""
    label = IO.phase_paths(source)["label"]
    law = IO.read_table(source, "check_law.csv")
    pooled = IO.read_table(source, "check_refinement_pooled.csv")
    paths = []
    figure = _law(law, label)
    if figure is not None:
        paths += S.savefig_pair(figure, out_dir, "phase7_check_law")
    figure = _coarse(law, IO.read_table(source, "check_law_pit_histogram.csv"), label)
    if figure is not None:
        paths += S.savefig_pair(figure, out_dir, "phase7_check_coarse")
    figure = _refinement(pooled, IO.read_table(source, "check_relaxation.csv"), label)
    if figure is not None:
        paths += S.savefig_pair(figure, out_dir, "phase7_check_refinement")
    paths += S.log_twin(_discretisation, out_dir, "phase7_check_discretisation", IO.read_table(source, "check_discretisation.csv"), _tolerance(pooled), label)
    figure = _timescale(IO.read_table(source, "check_timescale.csv"), label)
    if figure is not None:
        paths += S.savefig_pair(figure, out_dir, "phase7_check_timescale")
    figure = _sampling(IO.read_table(source, "check_sampling.csv"), label)
    if figure is not None:
        paths += S.savefig_pair(figure, out_dir, "phase7_check_sampling")
    return paths
