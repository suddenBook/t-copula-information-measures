"""Phase 2: the fitted shape, what the t copula says about each recording's channel sets.

  phase2_shape_vs_dimension   Where does p_hat sit at each interaction order, what tail index does that mean, and how much of the mass is exactly Gaussian?
  phase2_pair_shape_vs_rho    At d = 2, does the fitted shape depend on how concordant the pair is, and what likelihood does the extra parameter buy?
  phase2_heavy_boundary       How often does a fit land against the far end of the domain, and how close does the upper tail of p_hat get?

Each figure is written twice, `<stem>` on a linear scale and `<stem>_log` on a logarithmic one.
p_hat is exactly zero at the Gaussian copula, so the logarithmic twin is symmetric-log and keeps that mass on the axis instead of dropping it.
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


def _distribution_keys():
    return [Patch(facecolor="none", edgecolor=S.KEY, lw=0.7, label="distribution over channel sets"), Line2D([], [], color=S.KEY, marker="_", ls="none", ms=6, mew=1.2, label="median"), Line2D([], [], color=S.GREY, lw=0.9, label=r"Gaussian, $p = 0$")]


def _shape_vs_dimension(log, units, label):
    ok = IO.ok_units(units)
    dims = IO.dimensions(units)
    colors = [S.col_d(d) for d in dims]
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(S.WIDTH, 2.9), layout="constrained")

    p_groups = [IO.numeric(ok.loc[ok["d"] == d, "p_hat"]) for d in dims]
    S.violin(axA, p_groups, dims, colors)
    axA.axhline(0.0, color=S.GREY, lw=0.9, zorder=0)
    axA.set_ylabel(r"$\hat p = 1/\hat\nu$", labelpad=2)
    S.dimension_axis(axA, dims)
    S.panel_label(axA, "A", "fitted shape")

    nu_groups = [v[np.isfinite(v)] for v in (IO.numeric(ok.loc[ok["d"] == d, "nu_hat"]) for d in dims)]
    S.violin(axB, nu_groups, dims, colors)
    axB.set_ylabel(r"$\hat\nu$ (finite fits)", labelpad=2)
    S.dimension_axis(axB, dims)
    S.panel_label(axB, "B", "degrees of freedom")

    d_frac, fraction = IO.fraction_by_dimension(ok, IO.numeric(ok["p_hat"]) == 0.0)
    axC.plot(d_frac, fraction, color=S.model_color("t"), lw=1.1, marker="o", ms=2.6, mfc="white", mew=0.8, label=r"fits at $\hat p = 0$")
    axC.set_ylabel("fraction of channel sets", labelpad=2)
    S.dimension_axis(axC, dims)
    S.panel_label(axC, "C", "Gaussian estimates")

    if log:
        S.log_scale(axA, "y", np.concatenate(p_groups) if p_groups else None)
        S.log_scale(axB, "y", np.concatenate(nu_groups) if nu_groups else None)
        S.log_scale(axC, "y", fraction)

    handles = _distribution_keys()
    handles.append(Line2D([], [], color=S.model_color("t"), lw=1.1, marker="o", ms=2.6, mfc="white", mew=0.8, label=r"fraction at $\hat p = 0$"))
    S.legend_below(fig, handles, ncols=4, reserve=0.14)
    S.source_label(fig, label)
    return fig


def _pair_shape_vs_rho(log, units, label):
    pairs = IO.ok_units(units[units["d"] == 2])
    if pairs.empty:
        return None
    rho = np.abs(IO.numeric(pairs["rho"]))
    p = IO.numeric(pairs["p_hat"])
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(S.WIDTH, 2.7), layout="constrained")

    axA.scatter(rho, p, s=2.5, lw=0, alpha=0.35, color=S.col_d(2), rasterized=True, label="channel pair")
    edges = np.linspace(0.0, max(np.nanmax(rho) if rho.size else 1.0, 1e-3), 9)
    centres, medians = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (rho >= low) & (rho < high)
        if inside.sum() >= 5:
            centres.append(0.5 * (low + high))
            medians.append(np.median(p[inside]))
    if centres:
        axA.plot(centres, medians, color=S.GREY, lw=1.1, ls=(0, (3, 2)), label=r"median $\hat p$ in $|\rho|$ bins")
    axA.axhline(0.0, color=S.GREY, lw=0.9, zorder=0)
    axA.set_xlabel(r"$|\rho|$ from Kendall's tau", labelpad=2)
    axA.set_ylabel(r"$\hat p$", labelpad=2)
    S.panel_label(axA, "A", "shape against concordance")

    aic = IO.numeric(pairs["aic_diff"]) if "aic_diff" in pairs.columns else np.array([])
    if aic.size:
        axB.scatter(p, aic, s=2.5, lw=0, alpha=0.35, color=S.col_d(2), rasterized=True, label="channel pair")
        axB.axhline(0.0, color=S.GREY, lw=0.9, zorder=0, label="no gain over Gaussian")
        axB.set_xlabel(r"$\hat p$", labelpad=2)
        axB.set_ylabel(r"AIC difference, $t$ over Gaussian", labelpad=2)
    S.panel_label(axB, "B", "likelihood gain")

    if log:
        S.log_scale(axA, "y", p)
        S.log_scale(axB, "x", p)
    S.legend_below(fig, axA.get_legend_handles_labels()[0] + [Line2D([], [], color=S.GREY, lw=0.9, label="no gain over Gaussian")], ncols=3, reserve=0.15)
    S.source_label(fig, label)
    return fig


def _heavy_boundary(log, units, label):
    ok = IO.ok_units(units)
    dims = IO.dimensions(units)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(S.WIDTH, 2.7), layout="constrained")

    heavy = IO.flag(units["heavy_boundary"]) if "heavy_boundary" in units.columns else (IO.text(units["status"]) == "heavy_boundary")
    d_heavy, fraction = IO.fraction_by_dimension(units, heavy)
    axA.plot(d_heavy, fraction, color=S.status_color("heavy_boundary"), lw=1.2, marker="o", ms=2.8, mfc="white", mew=0.8, label="fits at the heavy end")
    axA.set_ylabel("fraction of channel sets", labelpad=2)
    S.dimension_axis(axA, dims)
    S.panel_label(axA, "A", "heavy-end fraction")

    for quantile, dashes in ((0.90, (0, (1, 1.4))), (0.95, (0, (3, 2))), (0.99, "-")):
        d_q, values = IO.quantile_by_dimension(ok, "p_hat", quantile)
        axB.plot(d_q, values, color=S.KEY, lw=1.0, ls=dashes, label=f"{quantile:.0%} quantile of $\\hat p$".replace("%", r"\%"))
    axB.axhline(0.0, color=S.GREY, lw=0.9, zorder=0)
    axB.set_ylabel(r"$\hat p$", labelpad=2)
    S.dimension_axis(axB, dims)
    S.panel_label(axB, "B", "upper tail of the estimate")

    if log:
        S.log_scale(axA, "y", fraction)
        S.log_scale(axB, "y", IO.numeric(ok["p_hat"]))
    handles = axA.get_legend_handles_labels()[0] + axB.get_legend_handles_labels()[0]
    handles.append(Line2D([], [], color=S.GREY, lw=0.9, label=r"Gaussian, $p = 0$"))
    S.legend_below(fig, handles, ncols=5, reserve=0.15)
    S.source_label(fig, label)
    return fig


def render(source, out_dir):
    units = IO.units(source)
    if units is None:
        return []
    label = IO.phase_paths(source)["label"]
    paths = []
    paths += S.log_twin(_shape_vs_dimension, out_dir, "phase2_shape_vs_dimension", units, label)
    paths += S.log_twin(_pair_shape_vs_rho, out_dir, "phase2_pair_shape_vs_rho", units, label)
    paths += S.log_twin(_heavy_boundary, out_dir, "phase2_heavy_boundary", units, label)
    return paths
