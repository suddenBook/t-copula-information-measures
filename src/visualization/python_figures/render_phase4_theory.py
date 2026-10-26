"""Phase 4, the closed-form theory grid the per-unit numbers are read against.

  phase4_theory_grid  The shape term and the four Gaussian-referenced differences as functions of p, one curve per interaction order.
                      The `_log` twin draws the same curves against the degrees of freedom nu = 1/p on a logarithmic axis, for a reader who thinks in nu.

Everything on this figure is deterministic: it is a function of (d, nu) alone, and a recording contributes nothing to it.
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

CURVES = [("kappa_d", r"shape term $\kappa_t(d,\nu)$"), ("MI_delta", S.MEASURE_LABEL["MI"]), ("TC_delta", S.MEASURE_LABEL["TC"]), ("DTC_delta", S.MEASURE_LABEL["DTC"]), ("O_delta", S.MEASURE_LABEL["O"])]


def _theory_grid(log, grid, label):
    """One panel per curve of CURVES, one line per dimension, against p or, on the log twin, against nu."""
    if grid is None:
        return None
    IO.require(grid, "p", "d")
    present = [(column, name) for column, name in CURVES if column in grid.columns and np.isfinite(IO.numeric(grid[column])).any()]
    if not present:
        print(f"  [skip] phase4_theory_grid: theory_grid.csv carries none of {', '.join(c for c, _ in CURVES)}")
        return None
    columns = min(3, len(present))
    lines = int(np.ceil(len(present) / columns))
    fig, axes = plt.subplots(lines, columns, figsize=(S.WIDTH, 2.3 * lines + 0.9), layout="constrained", squeeze=False)
    flat = axes.ravel()
    for axis in flat[len(present) :]:
        axis.set_axis_off()
    for index, (column, name) in enumerate(present):
        ax = flat[index]
        for d, block in grid.groupby("d"):
            block = block.sort_values("p")
            p = IO.numeric(block["p"])
            values = IO.numeric(block[column])
            if not np.isfinite(values).any():
                continue
            if log:
                nu = IO.numeric(block["nu"]) if "nu" in block.columns else np.where(p > 0, 1.0 / np.where(p > 0, p, 1.0), np.inf)
                keep = np.isfinite(nu) & (nu > 0)
                ax.plot(nu[keep], values[keep], color=S.col_d(d), lw=0.9)
            else:
                ax.plot(p, values, color=S.col_d(d), lw=0.9)
        ax.axhline(0.0, color=S.GREY, lw=0.9, zorder=0)
        ax.set_xlabel(r"degrees of freedom $\nu = 1/p$" if log else "$p = 1/\\nu$", labelpad=2)
        if log:
            ax.set_xscale("log")
        else:
            ax.axvline(0.0, color=S.GREY, lw=0.7, zorder=0)
        S.panel_label(ax, string.ascii_uppercase[index], name)
        if index % columns == 0:
            ax.set_ylabel("nats", labelpad=2)
    handles = [Line2D([], [], color=S.col_d(11), lw=0.9, label="one curve per dimension $d$"), Line2D([], [], color=S.GREY, lw=0.9, label="Gaussian value")]
    S.legend_below(fig, handles, ncols=2, reserve=0.08 + 0.04 / max(lines, 1))
    S.dimension_colorbar(fig, [ax for ax in flat if ax.has_data()])
    S.source_label(fig, label)
    return fig


def render(source, out_dir):
    """Render phase4_theory_grid from one Phase 4 output folder into out_dir and return the saved paths."""
    grid = IO.read_table(source, "theory_grid.csv")
    return S.log_twin(_theory_grid, out_dir, "phase4_theory_grid", grid, IO.phase_paths(source)["label"])
