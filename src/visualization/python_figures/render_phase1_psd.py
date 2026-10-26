"""Phase 1 QC figures: power spectral density.

Two figures for the first retained channel: the PSD of the artifact-cleaned signal before the band-pass (with the applied band shaded), and its comparison with the Phase 1 band-pass-filtered PSD.
The comparison is drawn only when X_filtered has the shape of X and differs from it.
The PSD is a QC diagnostic, not a result, recomputed here with the parameters of MATLAB pwelch: a hamming(round(2*fs)) window, 50% overlap, nfft = 2**nextpow2(2*fs), one-sided density, no detrending, then 10*log10.

Run:
    uv run src/visualization/python_figures/render_phase1_psd.py <preprocessed_data.mat> [out_dir]
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
from scipy.signal import welch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _phase_data as D  # noqa: E402

BANDS = [("delta", 0.5, 4), ("theta", 4, 8), ("alpha", 8, 13), ("beta", 13, 30), ("gamma", 30, 100)]
LINE_NOISE_HZ = 50.0


def psd_db(x, fs):
    """Welch PSD of one channel with the parameters above, as (frequencies in Hz, power in dB/Hz)."""
    wl = int(round(2 * fs))
    win = np.hamming(wl)
    nfft = 1 << int(np.ceil(np.log2(2 * fs)))  # 2^nextpow2(2*fs)
    nover = int(round(wl / 2))
    f, pxx = welch(x, fs=fs, window=win, noverlap=nover, nfft=nfft, detrend=False, scaling="density")
    return f, 10 * np.log10(np.maximum(pxx, np.finfo(float).tiny))


def _markers(ax, fmax, applied_band, native_band=None):
    """Band guides: the classic EEG band edges, the line-noise frequency and the analysis band.

    Only an applied filter band is shaded.
    A recording that Phase 1 passes through unfiltered (filter_applied = false) gets no shaded band, even when its file carries a filter_range.
    Its upstream band, when known, is marked with dotted edges instead.
    """
    for _lab, lo, hi in BANDS:
        if lo <= fmax:
            ax.axvline(lo, ls="--", color="#3775BA", lw=0.8, alpha=0.5)
    if LINE_NOISE_HZ <= fmax:
        ax.axvline(LINE_NOISE_HZ, ls=":", color="#D9541A", lw=1.1, alpha=0.8, label=f"{LINE_NOISE_HZ:g} Hz line noise")
    if applied_band is not None and np.all(np.isfinite(applied_band)):
        ax.axvspan(applied_band[0], applied_band[1], color="#D9E8FA", alpha=0.35, label=f"applied band {applied_band[0]:g}-{applied_band[1]:g} Hz")
    elif native_band is not None and np.all(np.isfinite(native_band)):
        for k, edge in enumerate(native_band):
            ax.axvline(edge, color=D.PALETTE["teal"], ls=":", lw=1.2, label=(f"upstream band {native_band[0]:g}-{native_band[1]:g} Hz (not applied here)") if k == 0 else None)


def render(mat_path, out_dir):
    """Draw the PSD figures of one preprocessed_data.mat and return the saved paths."""
    os.makedirs(out_dir, exist_ok=True)
    D.apply_publication_style()
    saved = []
    with h5py.File(mat_path, "r") as h:
        fs = D.read_scalar(h, "fs")
        X = D.read_matrix(h, "X")
        band = D.read_vector(h, "filter_range") if D.has(h, "filter_range") else None
        native = D.read_vector(h, "native_band") if D.has(h, "native_band") else None
        # Without a filter_applied field, a two-element filter_range counts as applied.
        if D.has(h, "filter_applied"):
            filter_applied = bool(D.read_scalar(h, "filter_applied"))
        else:
            filter_applied = band is not None and band.size == 2
        Xf = D.read_matrix(h, "X_filtered") if D.has(h, "X_filtered") else None
    fmax = min(100.0, fs / 2)
    applied = band if (filter_applied and band is not None and band.size == 2) else None

    f, db = psd_db(X[0], fs)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.plot(f, db, color="k", lw=1.4, label="artifact-cleaned pre-band-pass")
    _markers(ax, fmax, applied, native)
    ax.set_xlim(0, fmax)
    ax.set_xlabel("Frequency  [Hz]")
    ax.set_ylabel("Power/Freq  [dB/Hz]")
    ax.set_title("Power spectral density: artifact-cleaned pre-band-pass (retained column 1)", fontsize=12)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    saved += D.finalize(fig, out_dir, "phase1_psd_single")

    if Xf is not None and Xf.shape == X.shape and not np.allclose(Xf, X):
        ff, dbf = psd_db(Xf[0], fs)
        fig, ax = plt.subplots(figsize=(9.0, 5.2))
        ax.plot(f, db, color="k", lw=1.4, label="artifact-cleaned pre-band-pass")
        ax.plot(ff, dbf, color=D.PALETTE["red_strong"], lw=1.4, label="Phase-1 band-pass filtered")
        _markers(ax, fmax, applied, native)
        ax.set_xlim(0, fmax)
        ax.set_xlabel("Frequency  [Hz]")
        ax.set_ylabel("Power/Freq  [dB/Hz]")
        ax.set_title("Power spectral density: pre-band-pass vs Phase-1 band-pass-filtered (retained column 1)", fontsize=12)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, which="both", alpha=0.25)
        saved += D.finalize(fig, out_dir, "phase1_psd_comparison")
    return saved


def main() -> None:
    """Command-line entry point: render one MAT file into out_dir (default: the MAT file's folder)."""
    if len(sys.argv) < 2:
        print("usage: render_phase1_psd.py <preprocessed_data.mat> [out_dir]")
        return
    mat_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(mat_path))
    print("saved:", render(mat_path, out_dir))


if __name__ == "__main__":
    main()
