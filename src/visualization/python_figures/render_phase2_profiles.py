"""Phase 2: the profile likelihood behind each estimate.

  phase2_profiles_by_dimension  One panel per interaction order, a few channel sets each: does the profile fall away from an interior maximum, or climb monotonically into the Gaussian boundary at p = 0?

The 41-node profile is too wide for a CSV and lives in the phase's fit MAT files, so this is the one Phase 2 figure that reads HDF5.
Each curve is drawn relative to its own maximum, and the window ends at three times the largest fitted p in the panel (at least 0.3), because the question is the shape of the profile near the estimate and a full-domain view flattens it.

The profiles are in the per-dimension `fit_dXX.mat` files, not in the merged index; `_phase_io.fit_units` knows that and delivers them here as one `[units x 41]` block.
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

plt = S.style()
from matplotlib.lines import Line2D  # noqa: E402

MAX_CURVES = 5


def _by_dimension(units):
    """`{d: [(p grid, profile, p_hat), ...]}` from the unit columns."""
    if not all(key in units for key in ("d", "profile_p_grid", "profile_loglik_grid")):
        return {}
    grids = np.atleast_2d(np.asarray(units["profile_p_grid"], dtype=float))
    logliks = np.atleast_2d(np.asarray(units["profile_loglik_grid"], dtype=float))
    d = np.asarray(units["d"], dtype=float).ravel()
    p_hat = np.asarray(units.get("p_hat", np.full(d.size, np.nan)), dtype=float).ravel()
    grouped = {}
    for index in range(min(grids.shape[0], logliks.shape[0], d.size)):
        grid, loglik = grids[index], logliks[index]
        if grid.size == 0 or grid.size != loglik.size or not np.isfinite(loglik).any():
            continue
        if not np.isfinite(d[index]):
            continue
        grouped.setdefault(int(d[index]), []).append((grid, loglik, p_hat[index]))
    return dict(sorted(grouped.items()))


def _profiles(units, label):
    grouped = _by_dimension(units)
    if not grouped:
        return None
    dims = list(grouped)
    columns = min(5, len(dims))
    lines = int(np.ceil(len(dims) / columns))
    fig, axes = plt.subplots(lines, columns, figsize=(S.WIDTH, 1.55 * lines + 1.0), layout="constrained", squeeze=False)
    flat = axes.ravel()
    for axis in flat[len(dims) :]:
        axis.set_axis_off()

    for index, d in enumerate(dims):
        ax = flat[index]
        curves = grouped[d][:MAX_CURVES]
        peaks = [p for _, _, p in curves if np.isfinite(p)]
        window = max(0.3, 3.0 * (max(peaks) if peaks else 0.1))
        for grid, loglik, peak in curves:
            inside = grid <= window
            if inside.sum() < 2:
                inside = np.ones_like(grid, dtype=bool)
            relative = loglik - np.nanmax(loglik)
            ax.plot(grid[inside], relative[inside], color=S.col_d(d), lw=0.9)
            best = int(np.nanargmax(loglik))
            if grid[best] <= window:
                ax.plot([grid[best]], [relative[best]], marker="o", ms=2.6, color=S.KEY, mfc="white", mew=0.8, zorder=3)
        ax.axvline(0.0, color=S.GREY, lw=0.9, zorder=0)
        ax.set_xlim(-0.02 * window, window)
        S.panel_label(ax, string.ascii_uppercase[index % 26], f"$d = {d}$")
        if index % columns == 0:
            ax.set_ylabel("profile, relative", labelpad=2)
        if index >= len(dims) - columns:
            ax.set_xlabel("$p$", labelpad=2)

    handles = [Line2D([], [], color=S.col_d(11), lw=0.9, label="profile log-likelihood, one channel set"), Line2D([], [], color=S.KEY, marker="o", ls="none", ms=2.6, mfc="white", mew=0.8, label="maximum"), Line2D([], [], color=S.GREY, lw=0.9, label=r"Gaussian, $p = 0$")]
    S.legend_below(fig, handles, ncols=3, reserve=0.06 + 0.04 / max(lines, 1))
    S.source_label(fig, label)
    return fig


def render(source, out_dir):
    units = IO.fit_units(source)
    if not units:
        print("  [skip] phase2_profiles_by_dimension: no fit mat with per-unit profiles")
        return []
    figure = _profiles(units, IO.phase_paths(source)["label"])
    if figure is None:
        return []
    return S.savefig_pair(figure, out_dir, "phase2_profiles_by_dimension")
