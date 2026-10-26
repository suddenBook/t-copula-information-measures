"""Phase 5 renderer: the copula density of selected channel pairs and what the two models miss.

  phase5_density_<band>_<rank>  One selected pair: the empirical, Gaussian and fitted-t densities on one viridis scale, and the three signed departures on one symmetric RdBu_r scale, with the RMS of each departure in the panel corner.
  phase5_density_summary        One pair per concordance tertile, in the same layout, so the three regimes are read side by side.

Colour follows one convention across the figures: viridis is a copula density, and RdBu_r is a departure from one.
The three density panels of a row share one scale and the three departure panels share one symmetric scale, because the figure's claim is that the two residues can be compared by eye, and on separate scales they cannot.
The `_log` twin puts the density on a logarithmic colour scale and the departure on a symmetric-log one.
"""

from __future__ import annotations

import os
import re
import string
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _phase_io as IO  # noqa: E402
import _style as S  # noqa: E402

plt = S.style()
from matplotlib.patches import Patch  # noqa: E402

DENSITIES = [("empirical", "empirical", "emp."), ("gaussian", "Gaussian", "G"), ("t", "fitted $t$", "$t$")]
RESIDUALS = [("resid_emp_gauss", "empirical $-$ Gaussian", "emp. $-$ G", "rms_emp_gauss"), ("resid_emp_t", "empirical $-$ $t$", "emp. $-$ $t$", "rms_emp_t"), ("resid_t_gauss", "$t$ $-$ Gaussian", "$t$ $-$ G", "rms_t_gauss")]


def colour_keys():
    """The two colour maps of the figure, keyed as patches beside their colorbars.

    Public, with `draw_row`, because the cross-recording density figure draws the same row for a different selection of pairs and must not drift from this layout.
    """
    from matplotlib import cm

    return [Patch(facecolor=cm.viridis(0.45), label="copula density (viridis)"), Patch(facecolor=cm.RdBu_r(0.85), label="departure from it (RdBu_r)")]


def _pair_title(pair):
    """Source-line text of one pair: its channels, tertile band, rho and p-hat."""
    channels = IO.as_text(pair.get("channels_str", [])) if "channels_str" in pair else ""
    if not channels and "channels" in pair:
        channels = " ".join(f"{int(c)}" for c in IO.vector(pair["channels"]))
    band = IO.as_text(pair.get("band", [])) if "band" in pair else ""
    rho, p_hat = IO.scalar(pair.get("rho", np.nan)), IO.scalar(pair.get("p_hat", np.nan))
    head = f"channels {channels}" if channels else "pair"
    if band:
        head += f", {band} tertile"
    return f"{head},  $\\rho = {rho:.2f}$,  $\\hat p = {p_hat:.3f}$"


def draw_row(fig, axes, pair, log, letters, label_first=True, short=False):
    """One pair across six panels: three densities, then three departures."""
    edges_u = IO.vector(pair["u_edges"])
    edges_v = IO.vector(pair["v_edges"])
    grids = [np.asarray(pair[key], dtype=float) for key, _, _ in DENSITIES if key in pair]
    residuals = [np.asarray(pair[key], dtype=float) for key, _, _, _ in RESIDUALS if key in pair]
    if not grids or not residuals:
        return
    norm_density = S.density_norm(np.concatenate([g.ravel() for g in grids]), log=log)
    norm_residual = S.departure_norm(np.concatenate([r.ravel() for r in residuals]), log=log)

    images = []
    for index, (key, name, brief) in enumerate(DENSITIES):
        ax = axes[index]
        grid = np.asarray(pair[key], dtype=float)
        if log:
            grid = np.ma.masked_less_equal(grid, 0.0)
        images.append(ax.pcolormesh(edges_u, edges_v, grid, cmap=S.DENS_CMAP, norm=norm_density, shading="flat"))
        S.image_axes(ax)
        S.panel_label(ax, letters[index], brief if short else name)
        if index == 0 and label_first:
            ax.set_ylabel("$v$", labelpad=1)
    for index, (key, name, brief, rms_key) in enumerate(RESIDUALS):
        ax = axes[3 + index]
        images.append(ax.pcolormesh(edges_u, edges_v, np.asarray(pair[key], dtype=float), cmap=S.DIFF_CMAP, norm=norm_residual, shading="flat"))
        S.image_axes(ax)
        S.panel_label(ax, letters[3 + index], brief if short else name)
        rms = IO.scalar(pair.get(rms_key, np.nan))
        if np.isfinite(rms):
            ax.text(0.96, 0.05, f"RMS {rms:.3g}", transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color=S.KEY, bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.85))
    for ax in axes:
        ax.set_xlabel("$u$", labelpad=1)
    for image, axis_group, name in ((images[0], list(axes[:3]), "copula density"), (images[3], list(axes[3:]), "departure")):
        bar = fig.colorbar(image, ax=axis_group, fraction=0.035, pad=0.015, aspect=14)
        bar.set_label(name, fontsize=7)
        bar.ax.tick_params(labelsize=6.5)


def _pair_figure(log, pair, label):
    """The figure of one pair, its six panels in two rows of three."""
    fig, axes = plt.subplots(2, 3, figsize=(S.WIDTH, 5.0), layout="constrained")
    flat = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]
    draw_row(fig, flat, pair, log, list(string.ascii_uppercase[:6]))
    S.legend_below(fig, colour_keys(), ncols=2, reserve=0.08)
    S.source_label(fig, f"{label}   {_pair_title(pair)}")
    return fig


def _summary_figure(log, pairs, label):
    """One row of six panels per pair, labelled by its band; None when there is no pair."""
    if not pairs:
        return None
    fig, axes = plt.subplots(len(pairs), 6, figsize=(S.WIDTH, 1.30 * len(pairs) + 1.0), layout="constrained", squeeze=False)
    letters = list(string.ascii_uppercase)
    for row, pair in enumerate(pairs):
        draw_row(fig, list(axes[row]), pair, log, letters[6 * row : 6 * row + 6], label_first=(row == 0), short=True)
        band = IO.as_text(pair.get("band", [])) if "band" in pair else f"pair {row + 1}"
        axes[row][0].set_ylabel(f"{band}\n$v$", labelpad=1, fontsize=7)
    S.legend_below(fig, colour_keys(), ncols=2, reserve=0.05 + 0.05 / len(pairs))
    S.source_label(fig, label)
    return fig


def _stem(pair, index):
    """File stem phase5_density_<band>_<index>, with 'pair' when the band is unnamed."""
    band = IO.as_text(pair.get("band", [])) if "band" in pair else ""
    band = re.sub(r"[^A-Za-z0-9]+", "_", band).strip("_").lower()
    return f"phase5_density_{band or 'pair'}_{index}"


def render(source, out_dir):
    """Write one figure per selected pair and the summary of the first pair of each band; return the paths written."""
    pairs = IO.pair_density(source)
    if not pairs:
        return []
    label = IO.phase_paths(source)["label"]
    paths = []
    seen = {}
    per_band = {}
    for pair in pairs:
        band = IO.as_text(pair.get("band", [])) if "band" in pair else ""
        seen[band] = seen.get(band, 0) + 1
        paths += S.log_twin(_pair_figure, out_dir, _stem(pair, seen[band]), pair, label)
        per_band.setdefault(band, pair)
    paths += S.log_twin(_summary_figure, out_dir, "phase5_density_summary", list(per_band.values()), label)
    return paths
