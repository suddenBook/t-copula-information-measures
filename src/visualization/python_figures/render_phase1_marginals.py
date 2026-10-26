"""Phase 1 QC figures: descriptive marginal-shape diagnostics.

checkMarginalDistributions.m computes the per-channel kurtosis and skewness, which Phase 1 persists as `marginal_stats`.
The histogram, QQ and pooled-z panels are recomputed here from the persisted `X` as QC diagnostics, not results: 80-bin density histograms, QQ plots against norm.ppf(((1:n) - 0.5) / n), and pooled z-scores binned on the fixed edges linspace(-6, 6, 200).

  1. per-channel histograms with a moment-matched Gaussian reference;
  2. per-channel normal-reference QQ displays;
  3. kurtosis and skewness bars (from marginal_stats, or recomputed from X when it is absent);
  4. pooled z-score histogram on [-6, 6], linear and log.

The histogram and QQ figures show up to MAX_PANELS evenly spaced retained channels.

Run:
    uv run src/visualization/python_figures/render_phase1_marginals.py <preprocessed_data.mat> [out_dir]
"""

# /// script
# requires-python = ">=3.10"
# dependencies = ["h5py>=3.10", "matplotlib>=3.8", "numpy>=1.26", "scipy>=1.11"]
# ///
from __future__ import annotations

import os
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _phase_data as D  # noqa: E402

NUM_BINS = 80
MAX_PANELS = 12
MARGINAL_NOTE = "Descriptive marginal-shape QC on the artifact-cleaned, pre-band-pass signal. Margins are removed by ranks before copula analysis; Gaussian curves/lines are references, not marginal-normality, Gaussian-copula, or PE-copula goodness-of-fit tests, and no panel determines inclusion."


def _rep_channels(n_ch, k=8):
    """Up to k evenly spaced 0-based channel indices, first and last included."""
    return np.unique(np.round(np.linspace(0, n_ch - 1, min(n_ch, k))).astype(int))


def _grid(n):
    """(rows, columns) of a panel grid for n panels, with at most four columns."""
    cols = min(n, 4)
    return int(np.ceil(n / cols)), cols


def _add_scope_note(fig, bottom=0.075):
    """Add the scope caption MARGINAL_NOTE below the panels and tighten the layout."""
    fig.text(0.01, 0.005, MARGINAL_NOTE, fontsize=7, color="#555555", wrap=True)
    fig.tight_layout(rect=(0, bottom, 1, 0.96))


def _standardize(x):
    """Finite standardized values, or an empty array for a constant/invalid channel."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.array([])
    sd = float(np.std(x))
    if not np.isfinite(sd) or sd <= 0:
        return np.array([])
    return (x - np.mean(x)) / sd


def render(mat_path, out_dir):
    """Draw the four marginal-shape figures of one preprocessed_data.mat and return the saved paths."""
    os.makedirs(out_dir, exist_ok=True)
    D.apply_publication_style()
    with h5py.File(mat_path, "r") as h:
        X = D.read_matrix(h, "X")  # [n_ch, n_samples]
        stats = D.read_struct(h, "marginal_stats") if D.has(h, "marginal_stats") else None
    n_ch = X.shape[0]
    saved = []

    # 1. per-channel histograms + Gaussian
    ch1 = _rep_channels(n_ch, MAX_PANELS)
    nr, nc = _grid(len(ch1))
    fig, axes = plt.subplots(nr, nc, figsize=(3.0 * nc, 2.2 * nr), squeeze=False)
    for ax, ch in zip(axes.ravel(), ch1):
        x = X[ch][np.isfinite(X[ch])]
        if x.size == 0:
            ax.text(0.5, 0.5, "no finite samples", ha="center", va="center")
            ax.set_title(f"Retained col {ch + 1}", fontsize=9)
            ax.set_yticks([])
            continue
        ax.hist(x, bins=NUM_BINS, density=True, color=D.PALETTE["blue_secondary"], alpha=0.7)
        sd = float(np.std(x))
        if np.isfinite(sd) and sd > 0:
            xg = np.linspace(x.min(), x.max(), 200)
            ax.plot(xg, norm.pdf(xg, x.mean(), sd), color=D.PALETTE["red_strong"], lw=1.2)
        else:
            ax.text(0.5, 0.9, "zero variance; no Gaussian curve", ha="center", va="top", transform=ax.transAxes, fontsize=7)
        ax.set_title(f"Retained col {ch + 1}", fontsize=9)
        ax.set_yticks([])
    for ax in axes.ravel()[len(ch1) :]:
        ax.set_visible(False)
    fig.suptitle("Per-channel marginal shapes with Gaussian references", fontsize=12)
    _add_scope_note(fig)
    saved += D.finalize(fig, out_dir, "phase1_marginals_histograms")

    # 2. per-channel normal QQ
    fig, axes = plt.subplots(nr, nc, figsize=(3.0 * nc, 2.4 * nr), squeeze=False)
    for ax, ch in zip(axes.ravel(), ch1):
        z = np.sort(_standardize(X[ch]))
        if z.size == 0:
            ax.text(0.5, 0.5, "zero variance or no finite samples", ha="center", va="center")
            ax.set_title(f"Retained col {ch + 1}", fontsize=9)
            continue
        p = (np.arange(1, z.size + 1) - 0.5) / z.size
        tq = norm.ppf(p)
        ax.plot(tq, z, ".", ms=1.5, color=D.PALETTE["blue_main"])
        lim = [min(tq[0], z[0]), max(tq[-1], z[-1])]
        ax.plot(lim, lim, "-", color=D.PALETTE["neutral"], lw=1.0)
        ax.set_title(f"Retained col {ch + 1}", fontsize=9)
    for ax in axes.ravel()[len(ch1) :]:
        ax.set_visible(False)
    fig.suptitle("Per-channel normal-reference QQ displays", fontsize=12)
    _add_scope_note(fig)
    saved += D.finalize(fig, out_dir, "phase1_marginals_qq")

    # 3. kurtosis + skewness bars (from marginal_stats when present, else recompute)
    if stats is not None and "kurtosis" in stats:
        kurt, skew = stats["kurtosis"].ravel(), stats["skewness"].ravel()
    else:
        standardized = [_standardize(row) for row in X]
        kurt = np.array([np.mean(z**4) if z.size else np.nan for z in standardized])
        skew = np.array([np.mean(z**3) if z.size else np.nan for z in standardized])
    idx = np.arange(1, kurt.size + 1)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(max(6, 0.12 * kurt.size), 5.4), sharex=True)
    a1.bar(idx, kurt, color=D.PALETTE["blue_main"])
    a1.axhline(3, ls="--", color=D.PALETTE["red_strong"], lw=1.0)
    a1.set_ylabel("kurtosis")
    a1.set_title("Per-channel moments (Gaussian references: kurtosis 3, skewness 0)", fontsize=11)
    a2.bar(idx, skew, color=D.PALETTE["green_3"])
    a2.axhline(0, ls="--", color=D.PALETTE["red_strong"], lw=1.0)
    a2.set_ylabel("skewness")
    a2.set_xlabel("retained-column position")
    if kurt.size > 30:
        a2.set_xticks([])
    _add_scope_note(fig)
    saved += D.finalize(fig, out_dir, "phase1_marginals_moments")

    # 4. pooled z-score on the fixed [-6, 6] grid, linear + log
    standardized = [_standardize(row) for row in X]
    nonempty = [z for z in standardized if z.size]
    Z = np.concatenate(nonempty) if nonempty else np.array([])
    edges = np.linspace(-6, 6, 200)
    xr = np.linspace(-6, 6, 300)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for ax in (a1, a2):
        if Z.size:
            ax.hist(Z, bins=edges, density=True, color=D.PALETTE["blue_secondary"], alpha=0.7)
        else:
            ax.text(0.5, 0.5, "no nonconstant finite channels", ha="center", va="center", transform=ax.transAxes)
        ax.plot(xr, norm.pdf(xr), color=D.PALETTE["red_strong"], lw=1.3)
        ax.set_xlabel("pooled z-score")
        ax.set_xlim(-6, 6)
    a1.set_ylabel("pdf")
    a1.set_title("Pooled standardized marginal with Gaussian reference", fontsize=11)
    a2.set_yscale("log")
    a2.set_title("Pooled standardized marginal with Gaussian reference (log)", fontsize=11)
    _add_scope_note(fig)
    saved += D.finalize(fig, out_dir, "phase1_marginals_pooled_z")
    return saved


def main() -> None:
    """Command-line entry point: render one MAT file into out_dir (default: the MAT file's folder)."""
    if len(sys.argv) < 2:
        print("usage: render_phase1_marginals.py <preprocessed_data.mat> [out_dir]")
        return
    mat_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(mat_path))
    print("saved:", render(mat_path, out_dir))


if __name__ == "__main__":
    main()
