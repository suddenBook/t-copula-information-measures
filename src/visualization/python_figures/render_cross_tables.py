"""Cross-recording tables: the numbers a reader quotes, as CSV and as Markdown.

  Phase2/text/recording_table
      One row per recording: modality, sample rate, analysed band, channels retained, samples, duration, segments, median integrated autocorrelation time, median effective sample count and pair count.
  Phase3/text/overlap_by_recording_and_dimension
      The overlap of the observed distribution of p_hat with the surrogate one, the mass outside it, and the two yardstick overlaps, per recording and dimension, beside the unit counts, the two medians and the two fractions at p = 0.
  Phase4/text/recording_medians_at_low_and_high_d
      Per recording, at the lowest and the highest dimension the sweep reached: the median shape, the median difference of every measure in nats and in per cent, and the surrogate mass outside.

Every table is written twice from one frame, so the CSV a script reads and the Markdown a person reads cannot disagree.
A quantity a recording does not carry is left empty rather than filled in.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _phase_io as IO  # noqa: E402
import _style as S  # noqa: E402
from render_cross_shape import overlap_by_dimension, shape_summary  # noqa: E402
from render_cross_differences import difference_summary  # noqa: E402


def _band(recording):
    """The analysed band as 'low-high' in Hz, from the first of `filter_range`, `native_band` and `bandpass` the sidecar carries, or ''."""
    for key in ("filter_range", "native_band", "bandpass"):
        value = recording.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return f"{float(value[0]):g}-{float(value[1]):g}"
    return ""


def _autocorrelation(recording):
    """Median integrated autocorrelation time (samples) and median effective sample count over the channels in Phase 1's `autocorrelation.csv`, NaN where absent."""
    table = IO.read_table(IO.recording_phase(recording, 1), "autocorrelation.csv", quiet=True)
    if table is None:
        return np.nan, np.nan
    tau = IO.pick(table, "tau_int_samples", "tau_int")
    n_eff = IO.pick(table, "n_eff", "n_effective")
    return (float(np.nanmedian(IO.numeric(tau))) if tau is not None else np.nan, float(np.nanmedian(IO.numeric(n_eff))) if n_eff is not None else np.nan)


def _pair_count(recording):
    """The number of units at d = 2, from Phase 2's `summary_by_d.csv`, else the row count of `pairs.csv`, else NaN."""
    phase2 = IO.recording_phase(recording, 2)
    summary = IO.read_table(phase2, "summary_by_d.csv", quiet=True)
    if summary is not None and {"d"} <= set(summary.columns):
        count = IO.pick(summary, "n_units", "n_sets", "count")
        if count is not None:
            row = IO.numeric(count)[IO.numeric(summary["d"]) == 2]
            if row.size:
                return float(row[0])
    path = os.path.join(IO.phase_paths(phase2)["text_dir"], "pairs.csv")
    if os.path.isfile(path):
        with open(path) as handle:
            return float(max(sum(1 for _ in handle) - 1, 0))
    return np.nan


def recording_table(recordings):
    """The recording table, one row per recording, from its sidecar and its Phase 1 and Phase 2 tables."""
    rows = []
    for recording in recordings:
        fs = float(recording.get("fs", np.nan) or np.nan)
        samples = float(recording.get("n_samples", np.nan) or np.nan)
        tau, n_eff = _autocorrelation(recording)
        rows.append(
            {
                "dataset_key": recording.get("dataset_key", recording["rel_base"]),
                "modality": recording.get("modality", ""),
                "fs_hz": fs,
                "band_hz": _band(recording),
                "channels_retained": recording.get("n_channels_retained", np.nan),
                "samples": samples,
                "duration_s": samples / fs if fs else np.nan,
                "segments": recording.get("n_segments", np.nan),
                "median_tau_int_samples": tau,
                "median_n_eff": n_eff,
                "pairs": _pair_count(recording),
            }
        )
    return pd.DataFrame(rows)


def overlap_table(recordings):
    """The Phase 3 overlap columns of every recording, one row per recording and dimension."""
    rows = []
    for recording in recordings:
        summary = IO.read_table(IO.recording_phase(recording, 3), "surrogate_comparison_by_d.csv", quiet=True)
        if summary is None or "d" not in summary.columns:
            continue
        frame = pd.DataFrame({"dataset_key": recording.get("dataset_key", recording["rel_base"]), "modality": recording.get("modality", ""), "d": IO.numeric(summary["d"])})
        for column in ("n_units", "n_obs_ok", "n_sur1_ok", "ovl_obs_sur", "mass_outside", "ovl_sur_sur2", "ovl_obs_halves", "obs_q50", "sur_q50", "obs_fraction_p0", "sur_fraction_p0"):
            values = IO.pick(summary, column)
            if values is not None:
                frame[column] = IO.numeric(values)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def medians_table(recordings):
    """The medians of every recording at its lowest and its highest dimension, two rows per recording."""
    rows = []
    for recording in recordings:
        shape = shape_summary(recording)
        differences = difference_summary(recording)
        overlap = overlap_by_dimension(recording)
        if shape is None:
            continue
        dims = IO.numeric(shape["d"])
        dims = dims[np.isfinite(dims)]
        if not dims.size:
            continue
        for d in (float(dims.min()), float(dims.max())):
            entry = {"dataset_key": recording.get("dataset_key", recording["rel_base"]), "modality": recording.get("modality", ""), "d": d}
            block = shape[IO.numeric(shape["d"]) == d]
            entry["median_p_hat"] = float(block["p_median"].iloc[0]) if len(block) else np.nan
            entry["median_nu_hat"] = float(block["nu_median"].iloc[0]) if len(block) else np.nan
            if differences is not None:
                for measure in S.MEASURES:
                    chunk = differences[(IO.text(differences["measure"]) == measure) & (IO.numeric(differences["d"]) == d)]
                    entry[f"{measure}_median_delta_nats"] = float(chunk["median_nats"].iloc[0]) if len(chunk) else np.nan
                    entry[f"{measure}_median_delta_pct"] = float(chunk["median_pct"].iloc[0]) if len(chunk) else np.nan
            if overlap is not None and "mass_outside" in overlap.columns:
                chunk = overlap[IO.numeric(overlap["d"]) == d]
                entry["mass_outside"] = float(chunk["mass_outside"].iloc[0]) if len(chunk) else np.nan
            rows.append(entry)
    return pd.DataFrame(rows)


def render(recordings, output_base, label="cross-recording"):
    """Write the three tables as CSV and Markdown under `<output_base>/Phase<N>/text`; return the paths."""
    written = []
    tables = [(recording_table(recordings), "Phase2", "recording_table"), (overlap_table(recordings), "Phase3", "overlap_by_recording_and_dimension"), (medians_table(recordings), "Phase4", "recording_medians_at_low_and_high_d")]
    for frame, phase, stem in tables:
        if frame is None or frame.empty:
            print(f"  [skip] {stem}: no recording carries the columns it needs")
            continue
        written += IO.write_table(frame, os.path.join(output_base, phase, "text"), stem)
    return written
