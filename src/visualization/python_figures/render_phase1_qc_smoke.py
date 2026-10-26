"""Smoke check of the render_phase1_*.py QC renderers on synthetic fixtures.

The MATLAB test suite does not run the Python renderers, so this script checks the Phase 1 figure layer.
The check uses synthetic inputs only and reads no recordings.
It writes synthetic preprocessed_data.mat files (HDF5, the layout h5py sees for a MATLAB -v7.3 file) with the real field names, plus a JSON decision sidecar, then runs every renderer and asserts the figures and their key series, titles and band shading.

Two fixture types, because the two cases fail differently:

  A. Filtered recording (SEEG-like): filter_applied = true, filter_range set.
     It carries a 10 Hz sinusoid in channel 0 (a PSD peak to assert); a known window_var (for the stationarity panel, which plots the stored struct); an 8-channel X_raw, a 6-channel cleaned X, clean_info, artifact_info and a sidecar (for the cleaning and excluded-channel renderers); an artifact_info.time_profile (for the temporal artifact figure); a psd_band_selection whose near-DC drift is 1000 times the in-band bump (the linear panel must ignore it and report the in-band peak, not 0 Hz); and a MATLAB-empty channel_labels (a `[]` under -v7.3 is a shape-(2,) marker, not a zero-size dataset, and D.has must call it absent).
     A variant removes no channel, so only the retained-candidate family may appear.

  B. Pass-through recording (ECoG/LFP-like), band-limited upstream: filter_applied = false, filter_range MATLAB-empty, native_band = [1, 8].
     The PSD figures must draw no shaded filter band.
     A variant also stores a filter_range of [4, 14] that was never applied, so only the filter_applied flag keeps the renderers from shading it.

Run:
    uv run src/visualization/python_figures/render_phase1_qc_smoke.py
"""

# /// script
# requires-python = ">=3.10"
# dependencies = ["h5py>=3.10", "matplotlib>=3.8", "numpy>=1.26", "scipy>=1.11"]
# ///
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _phase_data as D  # noqa: E402
import render_phase1_band_selection_psd as R_bsel  # noqa: E402
import render_phase1_time_domain as R_td  # noqa: E402
import render_phase1_uniformity as R_uni  # noqa: E402
import render_phase1_statistics as R_stat  # noqa: E402
import render_phase1_stationarity as R_stn  # noqa: E402
import render_phase1_marginals as R_mrg  # noqa: E402
import render_phase1_psd as R_psd  # noqa: E402
import render_phase1_cleaning as R_clean  # noqa: E402
import render_phase1_excluded_channels as R_excl  # noqa: E402
import render_phase1_artifact_quality as R_aq  # noqa: E402

FS = 128.0
PEAK_HZ = 10.0
WINDOW_VAR = np.array([1.0, 1.3, 0.8, 1.1])
# excluded-channel ground truth (1-based indices into the 8-channel X_raw)
CANDIDATE_CHANNELS = np.array([3.0, 7.0])
EXCLUDED_CHANNELS = np.array([3.0])  # 7 is a retained detector candidate
# Both candidates carry the flat 100 sentinel that section 8b of detectSEEGArtifacts writes over channel-level diagnostic candidates.
# Their true loads here are 60% (ch 3) and 12.5% (ch 7), so a figure that plots or labels channel_artifact_pct instead of the reconstructed load fails on a 100-versus-60 mismatch.
ARTIFACT_PCT = np.array([1.0, 2.0, 100.0, 3.0, 4.0, 5.0, 100.0, 2.0])
CH3_TRUE_LOAD_PCT = 60.0  # frac[:, 2] = 0.6 across all windows
CH7_TRUE_LOAD_PCT = 100.0 * 4 / 32  # 4 dirty windows of 32
REASONS = ["abnormal kurtosis", "low power"]
BAND = (8.0, 13.0)
NATIVE_BAND = (1.0, 8.0)
N_RAW_CH, N_WINDOWS = 8, 32  # distinct, so the [ch x win] orientation is testable


def _matlab_empty(f, name):
    """Write a MATLAB `[]` as -v7.3 stores it: a shape-(2,) uint64 dimension marker with the MATLAB_empty attribute."""
    d = f.create_dataset(name, data=np.array([0, 0], dtype=np.uint64))
    d.attrs["MATLAB_class"] = np.bytes_(b"double")
    d.attrs["MATLAB_empty"] = np.uint8(1)


def _band_selection_group(f):
    """A psd_band_selection group whose near-DC drift is 1000 times the in-band bump, the shape of a drift-dominated SEEG spectrum."""
    freq = np.linspace(0.0, 64.0, 1024)
    drift = 1000.0 * np.exp(-freq / 0.05)  # the DC/drift term
    bump = 5.0 * np.exp(-(((freq - PEAK_HZ) / 0.8) ** 2))  # the in-band rhythm
    mean_amp = drift + bump + 1.0
    g = f.create_group("psd_band_selection")
    g["freq"] = freq
    g["mean_amp"] = mean_amp
    g["sd_amp"] = 0.1 * np.ones_like(freq)
    g["fs"] = np.array([[FS]])


def _write_fixture(path, sidecar_path, *, filtered: bool, stale_filter_range: bool = False, excluded_empty: bool = False):
    """Write one fixture MAT file and its sidecar, and return the X and X_raw written as ground truth.

    filtered=False models a pass-through recording.

    stale_filter_range=True additionally stores a filter_range that was never applied.
    Only this case tests the filter_applied check: with filter_range empty, D.has() alone already suppresses the shading, so a fixture without this flag cannot detect a renderer that ignores filter_applied.

    excluded_empty=True retains every candidate and removes none.
    A renderer that paginates on `auto_detected_severe_channels` instead of `excluded_channels` then draws removed-channel pages for a recording that removed nothing.
    """
    rng = np.random.default_rng(0)
    n_ch, n = 6, 4096
    t = np.arange(n) / FS
    X = 0.2 * rng.standard_normal((n_ch, n))
    X[0] += np.sin(2 * np.pi * PEAK_HZ * t)  # channel-0 spectral peak at 10 Hz
    X[5] = 0.0  # retained flatline robustness case
    Xf = 0.5 * X if filtered else X.copy()  # skip fixture: X_filtered == X
    X_raw = 0.15 * rng.standard_normal((N_RAW_CH, n))
    X_raw[2] += 3.0 * np.sin(2 * np.pi * 3.0 * t)  # channel 3: identifiable excluded trace
    X_raw[6] += 2.0  # channel 7: retained candidate

    with h5py.File(path, "w") as f:
        f["X"] = X
        f["X_filtered"] = Xf
        f["X_raw"] = X_raw
        f["fs"] = np.array([[FS]])
        f["filter_applied"] = np.array([[1.0 if filtered else 0.0]])
        if filtered:
            f["filter_range"] = np.array([[BAND[0]], [BAND[1]]])
            _matlab_empty(f, "native_band")
        else:
            if stale_filter_range:
                f["filter_range"] = np.array([[4.0], [14.0]])  # the band never applied
            else:
                _matlab_empty(f, "filter_range")
            f["native_band"] = np.array([[NATIVE_BAND[0]], [NATIVE_BAND[1]]])
        _matlab_empty(f, "channel_labels")  # SEEG ships none; D.has must say absent
        _band_selection_group(f)

        g = f.create_group("stationarity_info")
        g["window_times"] = np.array([15.0, 45.0, 75.0, 105.0])
        g["window_var"] = WINDOW_VAR
        g["window_mean"] = np.array([0.10, 0.12, 0.09, 0.11])
        g["cv_var"] = np.array([[0.13]])

        m = f.create_group("marginal_stats")
        m["kurtosis"] = np.array([3.1, 2.9, 4.0, 3.5, 2.8, 3.2])
        m["skewness"] = np.array([0.0, 0.1, -0.2, 0.05, -0.1, 0.02])

        ci = f.create_group("clean_info")
        ci["time_retention_pct"] = np.array([[100.0]])
        ci["channel_retention_pct"] = np.array([[75.0]])
        ci["cut_segments_time"] = np.zeros((2, 0))
        ci["auto_detected_severe_channels"] = CANDIDATE_CHANNELS
        if excluded_empty:
            _matlab_empty(ci, "excluded_channels")
        else:
            ci["excluded_channels"] = EXCLUDED_CHANNELS
        cp = ci.create_group("params")
        cp["severe_threshold"] = np.array([[10.0]])

        ai = f.create_group("artifact_info")
        ai["channel_artifact_pct"] = ARTIFACT_PCT
        ai["flag_abnormal_kurt"] = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=float)
        ai["flag_low_power"] = np.array([0, 0, 0, 0, 0, 0, 1, 0], dtype=float)
        ai["powerline_channels"] = np.array([2.0, 5.0])
        tp = ai.create_group("time_profile")
        # MATLAB writes [channels x windows]; h5py reads it back as (windows, channels).
        frac = np.zeros((N_WINDOWS, N_RAW_CH))
        frac[:, 2] = 0.6  # channel 3 dirty throughout
        frac[10:14, 6] = 1.0  # channel 7 dirty in a time block
        tp["fraction"] = frac
        tp["window_centers_sec"] = np.arange(N_WINDOWS, dtype=float) + 0.5
        tp["window_sec"] = np.array([[1.0]])
        tp["n_samples"] = np.array([[float(n)]])  # 32 windows x 128 samples = 4096
        tp["n_windows"] = np.array([[float(N_WINDOWS)]])

    with open(sidecar_path, "w") as fh:
        json.dump({"auto_detected_channels": CANDIDATE_CHANNELS.astype(int).tolist(), "reasons": REASONS}, fh)
    return {"X": X, "X_raw": X_raw}


def _capture(store):
    """Wrap D.finalize to record each figure's lines, vlines, titles, labels and texts in store; returns (original, wrapper)."""
    orig = D.finalize

    def _cap(fig, out_dir, name, dpi=300):
        """Record the figure's content in store[name], then save it with the original finalize."""
        lines, vlines, labels = [], [], []
        for ax in fig.axes:
            for ln in ax.get_lines():
                x = np.asarray(ln.get_xdata(), float)
                y = np.asarray(ln.get_ydata(), float)
                if x.size == 2 and x[0] == x[1]:
                    vlines.append(float(x[0]))  # axvline
                else:
                    lines.append((x, y))
            labels.extend(ax.get_legend_handles_labels()[1])
            leg = ax.get_legend()  # legends built with explicit handles=[...]
            if leg is not None:
                labels.extend(t.get_text() for t in leg.get_texts())
        store[name] = {
            "lines": lines,
            "vlines": vlines,
            "labels": labels,
            "titles": [ax.get_title() for ax in fig.axes],
            "axis_texts": [item.get_text() for ax in fig.axes for item in ax.texts],
            "figure_texts": [item.get_text() for item in fig.texts],
            "bar_heights": [p for ax in fig.axes for p in ax.patches],
            "patches": sum(len(ax.patches) for ax in fig.axes),
        }
        return orig(fig, out_dir, name, dpi)

    return orig, _cap


def _run_filtered(tmp):
    """Render every figure from the filtered fixture and check its content; returns the figure count and the two peak frequencies."""
    mat = str(tmp / "filtered" / "preprocessed_data.mat")
    os.makedirs(os.path.dirname(mat), exist_ok=True)
    sidecar = str(tmp / "filtered" / "artifact_clean_decision.json")
    out = str(tmp / "filtered" / "out")
    truth = _write_fixture(mat, sidecar, filtered=True)

    cap = {}
    orig, wrapped = _capture(cap)
    D.finalize = wrapped
    try:
        for r in (R_bsel, R_td, R_uni, R_stat, R_stn, R_mrg, R_psd, R_clean, R_aq):
            r.render(mat, out)
        R_excl.render(mat, out, sidecar_path=sidecar)
    finally:
        D.finalize = orig

    expected = {
        "phase1_band_selection_psd",
        "phase1_time_domain_raw",
        "phase1_time_domain_raw_vs_filtered",
        "phase1_uniformity_X_raw_uniform",
        "phase1_uniformity_X_filtered_uniform",
        "phase1_statistics_X",
        "phase1_statistics_X_filtered_uniform",
        "phase1_stationarity",
        "phase1_marginals_histograms",
        "phase1_marginals_qq",
        "phase1_marginals_moments",
        "phase1_marginals_pooled_z",
        "phase1_psd_single",
        "phase1_psd_comparison",
        "phase1_cleaning",
        "phase1_excluded_channels_p1",
        "phase1_severe_but_retained_p1",
        "phase1_artifact_channel_quality",
        "phase1_artifact_channel_flags",
        "phase1_artifact_time_profile",
    }
    missing = expected - set(cap)
    assert not missing, f"filtered fixture did not produce: {sorted(missing)}"
    for name in ("phase1_band_selection_psd", "phase1_psd_single", "phase1_cleaning", "phase1_artifact_time_profile"):
        assert (Path(out) / f"{name}.pdf").exists(), f"no PDF for {name}"

    # A MATLAB-empty channel_labels must read as absent, not as a shape-(2,) array.
    with h5py.File(mat, "r") as h:
        assert not D.has(h, "channel_labels"), "has() accepted a MATLAB [] as present"
        assert not D.has(h, "native_band"), "has() accepted an empty native_band"
        assert D.has(h, "filter_range"), "has() rejected a real filter_range"

    # Band selection: the near-DC drift is 1000 times the in-band bump.
    # The peak marker must land on the in-band rhythm, and the shaded band must be the applied filter band.
    bsel = cap["phase1_band_selection_psd"]
    peak_vlines = [v for v in bsel["vlines"] if BAND[0] <= v <= BAND[1]]
    assert peak_vlines, f"band-selection peak marker not in-band: {bsel['vlines']}"
    assert abs(peak_vlines[0] - PEAK_HZ) < 0.5, f"in-band peak at {peak_vlines[0]:.2f} Hz, expected ~{PEAK_HZ} (DC leaked in?)"
    assert any(l.startswith("filter band [") for l in bsel["labels"]), f"applied band not shaded: {bsel['labels']}"

    # The stationarity panel plots window_var from the stored struct.
    assert any(y.size == WINDOW_VAR.size and np.allclose(y, WINDOW_VAR) for _x, y in cap["phase1_stationarity"]["lines"]), "stationarity panel did not plot the window_var series from the struct"

    # PSD peak at the injected 10 Hz, and the applied band shaded
    f_psd, db_psd = cap["phase1_psd_single"]["lines"][0]
    peak_f = f_psd[int(np.argmax(db_psd))]
    df = f_psd[1] - f_psd[0]
    assert abs(peak_f - PEAK_HZ) <= df, f"PSD peak at {peak_f:.2f} Hz, expected ~{PEAK_HZ} Hz"
    assert any(l.startswith("applied band") for l in cap["phase1_psd_single"]["labels"]), f"filtered fixture: applied band not shaded: {cap['phase1_psd_single']['labels']}"

    # The cleaning summary plots both X_raw and X.
    clean = cap["phase1_cleaning"]
    assert any(np.array_equal(y, truth["X_raw"][0]) for _x, y in clean["lines"]), "cleaning figure did not plot the original X_raw trace"
    assert any(np.array_equal(y, truth["X"][0]) for _x, y in clean["lines"]), "cleaning figure did not plot the cleaned X trace"

    # Excluded-channels figure: it must paginate on clean_info.excluded_channels (ch 3), not on the auto_detected_severe_channels candidate list (ch 3 and ch 7).
    # The negative check carries the weight, because a renderer that reads the candidate list also plots ch 3.
    exc = cap["phase1_excluded_channels_p1"]
    assert any(np.array_equal(y, truth["X_raw"][2]) for _x, y in exc["lines"]), "excluded-channels figure did not plot the channel-3 X_raw trace"
    assert not any(np.array_equal(y, truth["X_raw"][6]) for _x, y in exc["lines"]), "excluded-channels figure plotted RETAINED channel 7: it is paginating on auto_detected_severe_channels instead of excluded_channels"
    exc_titles = " || ".join(exc["titles"])
    assert "Channel 3" in exc_titles and "abnormal kurtosis" in exc_titles, f"excluded-channels title missing channel/reason: {exc_titles!r}"
    assert "Channel 7" not in exc_titles, f"excluded-channels titled a retained channel: {exc_titles!r}"
    # The label must be the true temporal load (60%), not the 100 sentinel the channel carries.
    assert f"{CH3_TRUE_LOAD_PCT:.1f}%" in exc_titles and "sentinel" in exc_titles, f"excluded-channels labelled channel_artifact_pct instead of the true load {CH3_TRUE_LOAD_PCT:.1f}%: {exc_titles!r}"

    # Retained-candidate figure (stem phase1_severe_but_retained): candidates minus removed channels, here ch 7 only.
    sbr = cap["phase1_severe_but_retained_p1"]
    assert any(np.array_equal(y, truth["X_raw"][6]) for _x, y in sbr["lines"]), "retained-candidate figure did not plot the channel-7 X_raw trace"
    assert not any(np.array_equal(y, truth["X_raw"][2]) for _x, y in sbr["lines"]), "retained-candidate figure plotted REMOVED channel 3"
    sbr_titles = " || ".join(sbr["titles"])
    assert "Channel 7" in sbr_titles and "Channel 3" not in sbr_titles, f"retained-candidate figure has the wrong channel set: {sbr_titles!r}"
    assert f"{CH7_TRUE_LOAD_PCT:.1f}%" in sbr_titles, f"retained-candidate figure labelled the 100 sentinel instead of the true load {CH7_TRUE_LOAD_PCT:.1f}%: {sbr_titles!r}"
    assert any("Detector-flagged candidates retained" in text for text in sbr["figure_texts"]), f"retained-candidate family still carries a severity verdict: {sbr['figure_texts']}"
    assert not any("truly bad" in text or "Severe but" in text for text in exc["figure_texts"] + sbr["figure_texts"]), "excluded/candidate captions retain retrospective badness or severity wording"

    # Artifact quality: the retained candidate 7 must be distinguishable from the removed channel 3, and the bars must be the true temporal load, not the sentinel-contaminated channel_artifact_pct.
    # Both candidates report 100; their true loads are 60% and 12.5%.
    aq = cap["phase1_artifact_channel_quality"]
    aq_labels = " || ".join(aq["labels"])
    assert "detector candidate retained (1)" in aq_labels and "removed (1)" in aq_labels, f"artifact-quality legend did not separate retained candidates from removals: {aq_labels!r}"
    assert "sentinel (2)" in aq_labels, f"artifact-quality did not mark the channel_artifact_pct=100 sentinels: {aq_labels!r}"
    bars = [b.get_height() for b in aq["bar_heights"]]
    assert abs(bars[6] - CH7_TRUE_LOAD_PCT) < 1e-9, f"artifact-quality plotted channel 7 at {bars[6]:.2f}%, expected the TRUE load {CH7_TRUE_LOAD_PCT:.2f}% (it plotted the 100 sentinel instead)"
    assert abs(bars[2] - CH3_TRUE_LOAD_PCT) < 1e-9, f"artifact-quality plotted channel 3 at {bars[2]:.2f}%, expected the TRUE load {CH3_TRUE_LOAD_PCT:.2f}% (it plotted the 100 sentinel instead)"
    marginal_note = " || ".join(text for name in ("phase1_marginals_histograms", "phase1_marginals_qq", "phase1_marginals_moments", "phase1_marginals_pooled_z") for text in cap[name]["figure_texts"])
    assert "Margins are removed by ranks" in marginal_note, f"marginal figures lack the copula-scope note: {marginal_note!r}"
    marginal_axis_text = " || ".join(text for name in ("phase1_marginals_histograms", "phase1_marginals_qq") for text in cap[name]["axis_texts"])
    assert "zero variance" in marginal_axis_text, "marginal figures did not render the retained flatline empty state"
    stationarity_note = " || ".join(cap["phase1_stationarity"]["figure_texts"])
    assert "no stationarity test, verdict, or inclusion rule" in stationarity_note, f"stationarity figure lacks its non-exclusionary scope: {stationarity_note!r}"
    signal_labels = " || ".join(cap["phase1_uniformity_X_filtered_uniform"]["titles"])
    assert "Phase-1 band-pass-filtered analysis signal" in signal_labels, f"filtered uniform figure has the wrong provenance label: {signal_labels!r}"

    return len(expected), peak_vlines[0], peak_f


def _run_no_exclusions(tmp):
    """Fixture in which every detector candidate is retained and no channel is removed; returns the figure names drawn.

    A renderer paginating on the candidate list would emit removed-channel pages for a recording that removed nothing.
    Here the removed family must be empty, and only the retained-candidate family (stem phase1_severe_but_retained) may appear.
    """
    mat = str(tmp / "no_excl" / "preprocessed_data.mat")
    os.makedirs(os.path.dirname(mat), exist_ok=True)
    sidecar = str(tmp / "no_excl" / "artifact_clean_decision.json")
    out = str(tmp / "no_excl" / "out")
    truth = _write_fixture(mat, sidecar, filtered=True, excluded_empty=True)

    cap = {}
    orig, wrapped = _capture(cap)
    D.finalize = wrapped
    try:
        R_excl.render(mat, out, sidecar_path=sidecar)
    finally:
        D.finalize = orig

    drawn = sorted(cap)
    assert not any(n.startswith("phase1_excluded_channels") for n in drawn), f"a recording that excluded ZERO channels still produced excluded-channel pages: {drawn} -- the renderer is paginating on auto_detected_severe_channels"
    assert "phase1_severe_but_retained_p1" in cap, f"the retained candidates were not drawn: {drawn}"
    sbr = cap["phase1_severe_but_retained_p1"]
    plotted = sum(any(np.array_equal(y, truth["X_raw"][c]) for _x, y in sbr["lines"]) for c in (2, 6))
    assert plotted == 2, f"retained-candidate page must show BOTH candidates (3 and 7), got {plotted}"
    return drawn


def _run_skip(tmp, *, stale_filter_range: bool, tag: str):
    """Render the spectral and signal figures from a pass-through fixture, check that no filter band is shaded, and return the figure names."""
    mat = str(tmp / tag / "preprocessed_data.mat")
    os.makedirs(os.path.dirname(mat), exist_ok=True)
    sidecar = str(tmp / tag / "artifact_clean_decision.json")
    out = str(tmp / tag / "out")
    _write_fixture(mat, sidecar, filtered=False, stale_filter_range=stale_filter_range)

    cap = {}
    orig, wrapped = _capture(cap)
    D.finalize = wrapped
    try:
        R_bsel.render(mat, out)
        R_psd.render(mat, out)
        R_td.render(mat, out)
        R_uni.render(mat, out)
        R_stat.render(mat, out)
    finally:
        D.finalize = orig

    # X_filtered == X, so the comparison figures must not be drawn.
    assert "phase1_psd_comparison" not in cap, f"{tag}: drew a raw-vs-filtered PSD comparison though X_filtered == X"
    assert "phase1_time_domain_raw_vs_filtered" not in cap, f"{tag}: drew a raw-vs-filtered trace comparison though X_filtered == X"

    uniform_title = " || ".join(cap["phase1_uniformity_X_filtered_uniform"]["titles"])
    statistics_titles = " || ".join(cap["phase1_statistics_X_filtered_uniform"]["titles"])
    expected_label = "Phase-1 pass-through; upstream band-limited"
    assert expected_label in uniform_title, f"{tag}: uniform figure calls a pass-through signal filtered: {uniform_title!r}"
    assert expected_label in statistics_titles, f"{tag}: statistics figure calls a pass-through signal filtered: {statistics_titles!r}"

    # The central check of the pass-through fixtures: no filter band may be shaded on a recording Phase 1 does not filter, and the upstream band must be marked instead.
    for name in ("phase1_band_selection_psd", "phase1_psd_single"):
        labels = cap[name]["labels"]
        assert not any(l.startswith("filter band [") or l.startswith("applied band") for l in labels), f"{tag}/{name}: an unfiltered recording was shaded with a filter band: {labels}"
        assert any("upstream band" in l for l in labels), f"{tag}/{name}: upstream native_band not marked: {labels}"
        vl = cap[name]["vlines"]
        assert any(abs(v - NATIVE_BAND[0]) < 1e-9 for v in vl) and any(abs(v - NATIVE_BAND[1]) < 1e-9 for v in vl), f"{tag}/{name}: native_band edges not drawn at {NATIVE_BAND}: {vl}"

    return sorted(cap)


def main() -> None:
    """Run the four fixture checks in a temporary folder and print a one-line summary; any failed check raises AssertionError."""
    tmp = Path(tempfile.mkdtemp())
    n_expected, inband_peak, psd_peak = _run_filtered(tmp)
    skip_figs = _run_skip(tmp, stale_filter_range=False, tag="skip")
    _run_skip(tmp, stale_filter_range=True, tag="skip_legacy")
    no_excl_figs = _run_no_exclusions(tmp)

    print(
        f"PHASE1 QC SMOKE PASS: {n_expected} figures on the filtered fixture; "
        f"in-band peak={inband_peak:.2f} Hz (DC drift 1000x larger, ignored); "
        f"PSD peak={psd_peak:.2f} Hz (~{PEAK_HZ}); "
        "stationarity self-emits window_var; cleaning plots X_raw/X; "
        "removed-channel figure holds ONLY ch3 (labelled with its 60% true load, not "
        "the 100 sentinel) and retained-candidate figure holds ONLY ch7; "
        "artifact-quality separates them. "
        f"SKIP fixtures ({len(skip_figs)} figures x2, incl. a legacy mat that still "
        "carries filter_range=[4,14]): no filter band shaded, upstream band marked, "
        "no raw-vs-filtered panels; has() rejects MATLAB []. "
        f"NO-REMOVAL fixture (all detector candidates retained): "
        f"{no_excl_figs} -- zero removed-channel pages."
    )


if __name__ == "__main__":
    main()
