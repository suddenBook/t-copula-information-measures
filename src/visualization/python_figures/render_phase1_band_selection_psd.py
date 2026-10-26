"""Phase 1 QC figure: band-selection PSD evidence on a linear frequency axis.

The figure redraws the across-channel |PSD|^0.5 mean +/- 1 sd summary that Phase 1 persists as the `psd_band_selection` struct (computeBandSelectionPsd.m).
It reads the persisted arrays rather than recomputing them, so it shows exactly the spectrum behind the band decision.

Style: a linear frequency axis with two stacked panels, linear y on top and log y below, zoomed to the informative range, with the applied filter band (filter_range) shaded and the spectral peak marked.
A log-log view compresses this structure; the linear frequency axis keeps peaks and shoulders legible for judging the band choice.

The figure handles three properties of these spectra:

  * The near-DC drift term can be many times larger than anything above 1 Hz, and letting it set the linear y-limit would flatten the informative 1-60 Hz structure onto the axis.
    The linear y-limit is therefore taken over frequencies at or above a guard frequency (see _guard_hz); the log-y panel shows the full range.
  * The recordings are 1/f-dominated without a sharp global peak, so the band is a judgment call: a global maximum falls on the 1/f shoulder, or at 0 Hz when the DC bins are included.
    The peak is therefore marked inside the band of interest (the applied band, else the upstream band), which shows whether the band holds the rhythm it was chosen for; without either band it is searched above the guard frequency.
  * A recording that Phase 1 does not filter (filter_applied = false, as for the ECoG and LFP recordings, which are band-limited upstream) gets no shaded filter band.
    Its upstream `native_band` is marked with dotted lines instead.

Run:
    uv run src/visualization/python_figures/render_phase1_band_selection_psd.py <preprocessed_data.mat> [out_dir]
"""

# /// script
# requires-python = ">=3.10"
# dependencies = ["h5py>=3.10", "matplotlib>=3.8", "numpy>=1.26"]
# ///
from __future__ import annotations

import os
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _phase_data as D  # noqa: E402

MIN_GUARD_HZ = 0.5  # Hz; the linear y-limit never uses frequencies below this


def read_band_selection(h):
    """The psd_band_selection struct as a dict with freq, mean_amp, sd_amp and fs, or None when it or one of its arrays is absent.

    Without a stored fs, fs is taken as twice the highest frequency.
    """
    if "psd_band_selection" not in h:
        return None
    g = h["psd_band_selection"]
    for key in ("freq", "mean_amp", "sd_amp"):
        if key not in g:
            return None
    out = {k: np.asarray(g[k], dtype=float).ravel() for k in ("freq", "mean_amp", "sd_amp")}
    out["fs"] = float(np.asarray(g["fs"]).ravel()[0]) if "fs" in g else 2.0 * out["freq"].max()
    return out


def _guard_hz(band_lo):
    """Lowest frequency the linear y-limit may use: a quarter of the lower band edge but at least MIN_GUARD_HZ, or MIN_GUARD_HZ without a band."""
    if band_lo is not None and np.isfinite(band_lo) and band_lo > 0:
        return max(MIN_GUARD_HZ, band_lo / 4.0)
    return MIN_GUARD_HZ


def render(mat_path, out_dir):
    """Draw the band-selection PSD figure of one preprocessed_data.mat and return the saved paths ([] without psd_band_selection)."""
    os.makedirs(out_dir, exist_ok=True)
    D.apply_publication_style()
    with h5py.File(mat_path, "r") as h:
        bs = read_band_selection(h)
        if bs is None:
            print("psd_band_selection not present; nothing to render.")
            return []
        band = D.read_vector(h, "filter_range") if D.has(h, "filter_range") else None
        native = D.read_vector(h, "native_band") if D.has(h, "native_band") else None
        # Without a filter_applied field, a two-element filter_range counts as applied.
        if D.has(h, "filter_applied"):
            filter_applied = bool(D.read_scalar(h, "filter_applied"))
        else:
            filter_applied = band is not None and band.size == 2

    freq, mean_amp, sd_amp, fs = bs["freq"], bs["mean_amp"], bs["sd_amp"], bs["fs"]
    applied = band if (filter_applied and band is not None and band.size == 2 and np.all(np.isfinite(band))) else None
    upstream = native if (native is not None and native.size == 2 and np.all(np.isfinite(native))) else None

    ref = applied if applied is not None else upstream
    band_lo = float(ref[0]) if ref is not None else None
    band_hi = float(ref[1]) if ref is not None else None
    guard = _guard_hz(band_lo)

    # Linear x-axis zoomed to the informative range: cover the band with context.
    xmax = min(fs / 2.0, max(60.0, 3.0 * band_hi if band_hi else 60.0))
    v = (freq >= 0) & (freq <= xmax) & np.isfinite(mean_amp)
    f, m, s = freq[v], mean_amp[v], sd_amp[v]

    # Linear scale from f >= guard only: the DC/drift bins are not signal.
    g = f >= guard
    ytop = float(np.nanmax((m + s)[g])) * 1.05 if np.any(g) else None

    if ref is not None:
        sel = (f >= band_lo) & (f <= band_hi)
        peak_label = "in-band peak"
    else:
        sel = g
        peak_label = f"peak (search $\\geq$ {guard:g} Hz)"
    peak = f[sel][int(np.argmax(m[sel]))] if np.any(sel) else np.nan

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 8.0), sharex=True)
    for ax, logy in ((axes[0], False), (axes[1], True)):
        ax.fill_between(f, np.maximum(m - s, np.finfo(float).tiny), m + s, color="#D9E0F2", label=r"$\pm 1$ sd across channels")
        ax.plot(f, m, color=D.PALETTE["blue_main"], lw=1.5, label=r"mean $|\mathrm{PSD}|^{0.5}$")
        if applied is not None:
            ax.axvspan(applied[0], applied[1], color="#F2D9D9", alpha=0.45, label=f"filter band [{applied[0]:g}, {applied[1]:g}] Hz")
        elif upstream is not None:
            # Band-limited upstream: show where, but never imply Phase 1 filtered it.
            for k, edge in enumerate(upstream):
                ax.axvline(edge, color=D.PALETTE["teal"], ls=":", lw=1.2, label=(f"upstream band [{upstream[0]:g}, {upstream[1]:g}] Hz (not applied here)") if k == 0 else None)
        if np.isfinite(peak):
            ax.axvline(peak, color=D.PALETTE["red_strong"], ls="--", lw=1.0, label=f"{peak_label} {peak:.2f} Hz")
        if logy:
            ax.set_yscale("log")
        elif ytop is not None:
            ax.set_ylim(0.0, ytop)  # DC/drift excluded, so the band structure is legible
        ax.set_ylabel(r"$|\mathrm{PSD}|^{0.5}$" + ("  (log y)" if logy else "  (linear y)"))
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_title("Band-selection PSD evidence (across channels)", fontsize=12)
    axes[1].set_xlabel("Frequency [Hz]  (linear)")
    axes[1].set_xlim(0, xmax)
    fig.text(0.01, 0.005, f"linear-y limit taken over f $\\geq$ {guard:g} Hz (the near-DC drift term is excluded); log-y panel shows the full range.", fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return D.finalize(fig, out_dir, "phase1_band_selection_psd")


def main() -> None:
    """Command-line entry point: render one MAT file into out_dir (default: the MAT file's folder)."""
    if len(sys.argv) < 2:
        print("usage: render_phase1_band_selection_psd.py <preprocessed_data.mat> [out_dir]")
        return
    mat_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(mat_path))
    print("saved:", render(mat_path, out_dir))


if __name__ == "__main__":
    main()
