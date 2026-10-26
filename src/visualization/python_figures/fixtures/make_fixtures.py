# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "h5py"]
# ///
"""Write a synthetic results tree with the layout, tables and MAT files of a real run.

The renderers under `src/visualization/python_figures/` read a results tree, and a real tree takes days of computing to produce.
This script writes a small one of the same shape: two recordings of two modalities with Phases 1 to 6, and the simulation tree with Phase 7.
The SEEG recording is continuous and the ECoG one has 15 trial segments, so the within-trial-centring twins of Phase 6 differ from their headline values on the ECoG recording alone.
Every table carries the columns the phase's writer puts there, in the writer's names and order.
Tables no renderer reads, such as Phase 1's marginals and stationarity and most timing tables, are not written.
Every MAT file is written the way MATLAB's `-v7.3` writes one (HDF5, column-major; unit rows as a column struct with one dataset per field, everything else as a struct array of object references), so a renderer that reads the fixture reads a real run unchanged.

The numbers are plausible, not scientific.
p_hat is drawn from a gamma with a Gaussian atom at zero and a thin tail into the upper bound, the information differences follow a quadratic in p with the right ordering in d, and the densities are a Gaussian copula with a tail term.
Nothing here may be read as a result: the fixture exists so a figure can be shown to draw, letter, key and scale itself correctly before a real run pays for it.

Run:
    uv run src/visualization/python_figures/fixtures/make_fixtures.py <output root>
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from statistics import NormalDist

import h5py
import numpy as np
import pandas as pd

DIMENSIONS = list(range(2, 21))
N_CHANNELS = 24
N_GROUPS = 30  # channel sets per dimension for d >= 3
GRID = 20  # density grid, 20 x 20 cells on [0, 1]

RECORDINGS = [
    dict(
        dataset_key="human_SEEG/001/sleep",
        family="human_SEEG",
        results_subject="001",
        results_condition="sleep",
        seed_base=1,
        modality="SEEG",
        biological_subject="human-001",
        recording="001",
        state="sleep",
        temporal_adapter="seeg_auto",
        artifact_profile="raw_voltage",
        input_relpath="human_SEEG/001/sleep.mat",
        fs=512.0,
        filter_signal=True,
        bandpass=[4.0, 14.0],
        native_band=[0.5, 200.0],
        filter_range=[4.0, 14.0],
        expected_channels=136,
        n_channels_raw=136,
        n_channels_retained=N_CHANNELS,
        n_samples=60000,
        n_segments=1,
    ),
    dict(
        dataset_key="monkey_ECoG/EC1/eyes_closed",
        family="monkey_ECoG",
        results_subject="EC1",
        results_condition="eyes_closed",
        seed_base=7,
        modality="ECoG",
        biological_subject="monkey-George",
        recording="EC1",
        state="eyes_closed",
        temporal_adapter="trial_segments",
        artifact_profile="standardized_relative",
        input_relpath="monkey_ECoG/EC1/eyes_closed.mat",
        fs=100.0,
        filter_signal=False,
        bandpass=[1.0, 8.0],
        native_band=[1.0, 8.0],
        filter_range=[1.0, 8.0],
        expected_channels=49,
        n_channels_raw=49,
        n_channels_retained=N_CHANNELS,
        n_samples=30000,
        n_segments=15,
    ),
]

MEASURES = ("MI", "TC", "DTC", "O")
_NORM = NormalDist()


# ---------------------------------------------------------------- small helpers
def _flag(values):
    """A logical column the way the phases write one, as 1 or 0: `unitRowsToColumns` and the phase tables store logicals as doubles, so every real CSV spells them numerically."""
    return np.asarray(values, dtype=bool).astype(int)


def _kappa(d, p):
    """A stand-in shape term: negative, second order in p, growing with d."""
    return -(0.20 * d + 0.02 * d * d) * np.square(p)


def _differences(d, p):
    """The four Gaussian-referenced differences in nats, from the stand-in shape term."""
    return {"MI": -_kappa(2, p) if d == 2 else np.full_like(np.asarray(p, float), np.nan), "TC": -_kappa(d, p), "DTC": d * _kappa(d - 1, p) - (d - 1) * _kappa(d, p), "O": (d - 2) * _kappa(d, p) - d * _kappa(d - 1, p)}


def _gaussian_measures(d, logdet_R):
    """Stand-in Gaussian measures of a correlation matrix, from its log-determinant.

    `TC_G = -0.5 log det R` is the exact Gaussian identity and MI is TC at d = 2.
    DTC is a fixed share of TC and O is the remainder, which keeps O = TC - DTC true.
    This is enough to exercise a figure and makes no scientific claim.
    """
    tc = -0.5 * np.asarray(logdet_R, dtype=float)
    dtc = 0.7 * tc
    return {"MI": tc if d == 2 else np.full(tc.shape, np.nan), "TC": tc, "DTC": dtc, "O": tc - dtc}


def _error_columns(d, p_true, p_hat, logdet_true, logdet_hat):
    """The eleven per-measure columns of Phase 7's long tables, per replicate.

    Two quantities are kept apart, as `simulationInformationError` keeps them: the correction `Delta(d, p)` in `<M>_delta_*`, which is a function of the shape alone, and the full measure `M_G(R) + Delta(d, p)` in `<M>_full_*`, whose error the p-only, R-only and interaction triple attributes.
    The split is additive, so the interaction is zero to floating point, exactly as the real helper produces it.
    """
    shape_true = _differences(d, np.asarray(p_true, dtype=float))
    shape_hat = _differences(d, np.asarray(p_hat, dtype=float))
    gauss_true = _gaussian_measures(d, logdet_true)
    gauss_hat = _gaussian_measures(d, logdet_hat)
    out = {}
    for measure in MEASURES:
        delta_true = np.asarray(shape_true[measure], dtype=float)
        delta_hat = np.asarray(shape_hat[measure], dtype=float)
        full_true = gauss_true[measure] + delta_true
        full_hat = gauss_hat[measure] + delta_hat
        full_p_only = gauss_true[measure] + delta_hat
        full_R_only = gauss_hat[measure] + delta_true
        out[f"{measure}_delta_true"] = delta_true
        out[f"{measure}_delta_hat"] = delta_hat
        out[f"{measure}_delta_err"] = delta_hat - delta_true
        out[f"{measure}_full_true"] = full_true
        out[f"{measure}_full_hat"] = full_hat
        out[f"{measure}_full_p_only"] = full_p_only
        out[f"{measure}_full_R_only"] = full_R_only
        out[f"{measure}_err"] = full_hat - full_true
        out[f"{measure}_err_p_only"] = full_p_only - full_true
        out[f"{measure}_err_R_only"] = full_R_only - full_true
        out[f"{measure}_err_interaction"] = (full_hat - full_true) - (full_p_only - full_true) - (full_R_only - full_true)
    return out


def _q(values, level):
    """A quantile that survives an all-NaN column (MI exists only at d = 2)."""
    values = np.asarray(values, dtype=float)
    return float(np.nanquantile(values, level)) if np.isfinite(values).any() else float("nan")


def _stat(fn, values):
    """A statistic over the finite entries, or NaN when a measure is not defined here."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(fn(values)) if values.size else float("nan")


def _relative(numerator, truth):
    """An error as a share of its own truth; NaN against a zero or undefined truth."""
    if np.isfinite(truth) and truth != 0:
        return float(numerator) / abs(float(truth))
    return float("nan")


def _draw_p(rng, d, size):
    """p_hat with an atom at the Gaussian, a gamma body and a thin boundary tail."""
    level = 0.05 + 0.004 * (d - 2)
    p = rng.gamma(2.0, level / 2.0, size=size)
    p[rng.random(size) < max(0.02, 0.28 - 0.012 * d)] = 0.0
    heavy = rng.random(size) < 0.01
    p[heavy] = rng.uniform(2.96, 3.0, size=int(heavy.sum()))
    return np.minimum(p, 3.0)


def _status(p, ambiguous, degenerate, failure):
    """The status `tFitUnit` assigns, by its precedence: degenerate pair > numerical failure > heavy end > ambiguous profile > ok.

    Each later assignment overrides the one before it, so the label of highest precedence is written last.
    """
    out = np.full(p.shape, "ok", dtype=object)
    out[ambiguous] = "ambiguous_profile"
    out[p > 2.95] = "heavy_boundary"
    out[failure] = "numerical_failure"
    out[degenerate] = "degenerate_pair"
    return out


def _channel_sets(rng, d, count):
    """Every channel pair at d = 2, else `count` random sets of d channels, as sorted 1-based channel lists."""
    if d == 2:
        return [list(c) for c in itertools.combinations(range(1, N_CHANNELS + 1), 2)]
    return [sorted(rng.choice(np.arange(1, N_CHANNELS + 1), size=d, replace=False).tolist()) for _ in range(count)]


def _channel_columns(sets):
    """The channel sets as `channels_str` strings and as the NaN-padded columns `ch01` to `ch20`."""
    padded = np.full((len(sets), 20), np.nan)
    for row, channels in enumerate(sets):
        padded[row, : len(channels)] = channels
    columns = {f"ch{i + 1:02d}": padded[:, i] for i in range(20)}
    return [" ".join(str(c) for c in s) for s in sets], columns


# ---------------------------------------------------------------- Phase 2
def phase2_units(rng, recording):
    """Phase 2's flat unit table, the columns of `pairs.csv` and `groups_long.csv`, with every dimension in one frame."""
    frames = []
    for d in DIMENSIONS:
        sets = _channel_sets(rng, d, N_GROUPS)
        count = len(sets)
        p = _draw_p(rng, d, count)
        n = int(recording["n_samples"])
        rho = np.where(np.full(count, d == 2), rng.beta(2.0, 4.0, count) * rng.choice([-1, 1], count), np.nan)
        max_abs_rho = np.abs(rho) if d == 2 else rng.uniform(0.2, 0.9, count)
        mean_abs_rho = max_abs_rho * rng.uniform(0.4, 0.9, count)
        logdet = -0.5 * d * np.square(mean_abs_rho) * rng.uniform(0.8, 1.2, count)
        loglik_g = -n * rng.uniform(0.01, 0.4, count)
        loglik_t = loglik_g + 0.5 * n * np.square(p) * rng.uniform(0.5, 1.5, count)
        repair_shift = np.abs(rng.normal(0.0, 0.004, count)) if d >= 3 else np.zeros(count)
        repair_applied = repair_shift > 1e-6
        repair_gross = repair_shift > 0.02
        degenerate = (d == 2) & (rng.random(count) < 0.004)
        ambiguous = rng.random(count) < 0.02
        failure = rng.random(count) < 0.003
        # One failure per dimension is placed inside the prefix Phase 3 takes as its surrogate subset, so the observed-validity filter of the surrogate figures is exercised by the fixture instead of depending on a 0.3 per cent draw.
        failure[min(2, count - 1)] = True
        status = _status(p, ambiguous, degenerate, failure)
        channels_str, channel_columns = _channel_columns(sets)
        frame = pd.DataFrame(
            {
                "dataset_key": recording["dataset_key"],
                "d": d,
                "unit_index": np.arange(1, count + 1),
                "channels_str": channels_str,
                **channel_columns,
                "n": n,
                "p_hat": p,
                "nu_hat": np.where(p > 0, 1.0 / np.where(p > 0, p, 1.0), np.inf),
                "loglik_t": loglik_t,
                "loglik_gaussian": loglik_g,
                "aic_diff": 2.0 * (loglik_t - loglik_g) - 2.0,
                "rho": rho,
                "tau": (2.0 / np.pi) * np.arcsin(rho) if d == 2 else np.full(count, np.nan),
                "max_abs_rho": max_abs_rho,
                "mean_abs_rho": mean_abs_rho,
                "logdet_R": logdet,
                "sum_logdet_minors": logdet * (d - 1) * rng.uniform(0.9, 1.1, count),
                "cond_R": 1.0 + 20.0 * np.square(max_abs_rho) * rng.uniform(0.8, 1.2, count),
                "min_eig_R": np.maximum(1e-8, 1.0 - max_abs_rho * rng.uniform(0.8, 1.0, count)),
                "repair_applied": _flag(repair_applied),
                "repair_shift_rms": repair_shift,
                "repair_gross": _flag(repair_gross),
                "degenerate_pair": _flag(degenerate),
                "gaussian_snapped": _flag(p == 0),
                "ambiguous_profile": _flag(ambiguous),
                "heavy_boundary": _flag(p > 2.95),
                "numerical_ok": _flag(~failure),
                "status": status,
                "reason": np.where(failure, "optimizer_failed", ""),
                "optimizer_exitflag": np.where(failure, 0, 1),
                "fit_seconds": rng.gamma(3.0, 0.004 * d, count),
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def write_phase2(base, rng, recording, units):
    """Phase 2's tables, the per-dimension fit MAT files and their merged index."""
    text_dir, mat_dir = _phase_dirs(base, 2)
    pairs = units[units["d"] == 2]
    groups = units[units["d"] > 2]
    pairs.to_csv(os.path.join(text_dir, "pairs.csv"), index=False)
    groups.to_csv(os.path.join(text_dir, "groups_long.csv"), index=False)

    rows, counts = [], []
    for d, block in units.groupby("d"):
        ok = block[block["numerical_ok"] == 1]
        p, nu = ok["p_hat"].to_numpy(), ok["nu_hat"].to_numpy()
        finite_nu = nu[np.isfinite(nu)]
        rows.append(
            {
                "d": d,
                "n_units": len(block),
                "n_ok": len(ok),
                "median_p_hat": np.median(p),
                "q25_p_hat": np.quantile(p, 0.25),
                "q75_p_hat": np.quantile(p, 0.75),
                "mean_p_hat": p.mean(),
                "sd_p_hat": p.std(ddof=1),
                "median_nu_hat": np.median(finite_nu) if finite_nu.size else np.nan,
                "q25_nu_hat": np.quantile(finite_nu, 0.25) if finite_nu.size else np.nan,
                "q75_nu_hat": np.quantile(finite_nu, 0.75) if finite_nu.size else np.nan,
                "mean_nu_hat": finite_nu.mean() if finite_nu.size else np.nan,
                "sd_nu_hat": finite_nu.std(ddof=1) if finite_nu.size else np.nan,
                "fraction_p0": float((p == 0).mean()),
                "fraction_heavy": float((block["heavy_boundary"] == 1).mean()),
                "fraction_ambiguous": float((block["ambiguous_profile"] == 1).mean()),
                "fraction_repair_gross": float((block["repair_gross"] == 1).mean()),
                "median_fit_seconds": float(block["fit_seconds"].median()),
            }
        )
        for status, count in block["status"].value_counts().items():
            counts.append({"d": d, "status": status, "count": int(count)})
    pd.DataFrame(rows).to_csv(os.path.join(text_dir, "summary_by_d.csv"), index=False)
    pd.DataFrame(counts).to_csv(os.path.join(text_dir, "status_counts_by_d.csv"), index=False)

    # The MAT files have Phase 2's layout: the per-dimension files carry the 41-node profiles and fit_all.mat is the merged index without them, so the fixture exercises the fallback `_phase_io.fit_units` takes to find a profile.
    # They hold the first six units of every dimension.
    grid = np.linspace(0.0, 3.0, 41)
    index = {"dataset_key": recording["dataset_key"], "d": [], "unit_index": [], "p_hat": [], "status": []}
    for d, block in units.groupby("d"):
        head = block.head(6)
        peak = head["p_hat"].to_numpy(dtype=float)
        columns = {
            "dataset_key": recording["dataset_key"],
            "d": np.full(len(head), float(d)),
            "unit_index": head["unit_index"].to_numpy(dtype=float),
            "p_hat": peak,
            "status": list(head["status"]),
            "profile_p_grid": np.tile(grid, (len(head), 1)),
            "profile_loglik_grid": (head["loglik_gaussian"].to_numpy(dtype=float)[:, None] - 0.5 * 4000.0 * np.square(grid[None, :] - peak[:, None])),
        }
        write_unit_columns(os.path.join(mat_dir, f"fit_d{int(d):02d}.mat"), "units", columns)
        for key in ("d", "unit_index", "p_hat", "status"):
            index[key] = list(index[key]) + list(columns[key])
    write_unit_columns(os.path.join(mat_dir, "fit_all.mat"), "units", index)
    _write_run_config(text_dir, {"phase": 2, "dimensions": DIMENSIONS, "n_groups": N_GROUPS})


# ---------------------------------------------------------------- Phase 3
PHASE3_LEVELS = (("025", 2.5), ("05", 5.0), ("10", 10.0), ("25", 25.0), ("50", 50.0), ("75", 75.0), ("90", 90.0), ("95", 95.0), ("975", 97.5))
PHASE3_STATUSES = ("ok", "heavy_boundary", "ambiguous_profile", "numerical_failure", "degenerate_pair")


def _overlap(a, b, n_bins=20, atoms=(0.0, 3.0)):
    """`distributionOverlap`: the atoms as cells of their own, and equal-count bins of the pooled remaining values, left-closed; NaN when either sample is empty."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    pooled = np.concatenate([a[~np.isin(a, atoms)], b[~np.isin(b, atoms)]])
    edges = np.unique(np.quantile(pooled, np.arange(1, n_bins) / n_bins)) if pooled.size else np.array([])

    def mass(x):
        rest = x[~np.isin(x, atoms)]
        counts = np.bincount(np.searchsorted(edges, rest, side="right"), minlength=edges.size + 1)
        return np.concatenate([[np.mean(x == atom) for atom in atoms], counts / x.size])

    return float(np.minimum(mass(a), mass(b)).sum())


def write_phase3(base, rng, recording, units):
    """The pooled surrogate comparison of one recording, as the four tables Phase 3 writes.

    Every Phase 2 channel set carries its observed fit, its seeded half, and a fit on each of two surrogate realizations, drawn lighter-tailed than the observed ones.
    One surrogate fit per dimension fails with a finite, far-off estimate, beside the failed observed fit `phase2_units` places, so the figures' filter on the numerical_ok columns is exercised.
    One surrogate fit at the largest dimension sits exactly on the heavy end p = 3.
    The writer draws from its own generator and leaves `rng` to the other phases.
    """
    draw = np.random.default_rng(300 + int(recording["seed_base"]))
    text_dir, _ = _phase_dirs(base, 3)
    frames, comparison, floors, timing = [], [], [], []
    for d in DIMENSIONS:
        block = units[units["d"] == d]
        count = len(block)
        half = np.full(count, 2)
        half[draw.permutation(count)[: (count + 1) // 2]] = 1
        p_obs = block["p_hat"].to_numpy(dtype=float)
        ok_obs = (block["numerical_ok"] == 1).to_numpy()
        heavy_obs = (block["heavy_boundary"] == 1).to_numpy()
        frame = pd.DataFrame({"d": d, "unit_index": block["unit_index"].to_numpy(), "channels_str": block["channels_str"].to_numpy(), "half": half, "p_hat_obs": p_obs, "numerical_ok_obs": _flag(ok_obs)})
        fits = {}
        for b in (1, 2):
            p = _draw_p(draw, d, count) * 0.55
            failure = draw.random(count) < 0.003
            if b == 1:
                failure[min(1, count - 1)] = True
                if d == DIMENSIONS[-1]:
                    p[0] = 3.0
            p[failure] = 2.9
            status = _status(p, np.zeros(count, bool), np.zeros(count, bool), failure)
            frame[f"p_hat_s{b}"] = p
            frame[f"numerical_ok_s{b}"] = _flag(~failure)
            frame[f"status_s{b}"] = status
            fits[b] = (p[~failure], status[~failure] == "heavy_boundary", status)
        frames.append(frame)

        observed = p_obs[ok_obs]
        row = {"d": d, "n_units": count, "n_obs_ok": observed.size, "n_sur1_ok": fits[1][0].size, "n_sur2_ok": fits[2][0].size, "ovl_obs_sur": _overlap(observed, fits[1][0])}
        row["mass_outside"] = 1.0 - row["ovl_obs_sur"]
        row["ovl_sur_sur2"] = _overlap(fits[1][0], fits[2][0])
        row["ovl_obs_halves"] = _overlap(p_obs[ok_obs & (half == 1)], p_obs[ok_obs & (half == 2)])
        samples = (("obs", observed, heavy_obs[ok_obs]), ("sur", fits[1][0], fits[1][1]), ("sur2", fits[2][0], fits[2][1]))
        for prefix, values, _ in samples:
            for name, level in PHASE3_LEVELS:
                row[f"{prefix}_q{name}"] = _q(values, level / 100.0)
        for prefix, values, _ in samples:
            row[f"{prefix}_fraction_p0"] = _stat(np.mean, values == 0.0)
        for prefix, _, heavy in samples:
            row[f"{prefix}_fraction_heavy"] = _stat(np.mean, heavy)
        for b, prefix in ((1, "sur"), (2, "sur2")):
            for status in PHASE3_STATUSES:
                row[f"{prefix}_status_{status}"] = int((fits[b][2] == status).sum())
        comparison.append(row)

        for b in (1, 2):
            differences = _differences(d, fits[b][0])
            for measure in MEASURES:
                if measure == "MI" and d != 2:
                    continue
                values = np.asarray(differences[measure], dtype=float)
                floors.append({"d": d, "realization": b, "measure": measure, "n": int(values.size), "q50": _q(values, 0.50), "q90": _q(values, 0.90), "q95": _q(values, 0.95), "q99": _q(values, 0.99)})
            timing.append({"b": b, "d": d, "chunk": 1, "units": count, "seconds": float(draw.gamma(3.0, 0.5 * d)), "workers": 8})
    pd.concat(frames, ignore_index=True).to_csv(os.path.join(text_dir, "surrogate_units.csv"), index=False)
    pd.DataFrame(comparison).to_csv(os.path.join(text_dir, "surrogate_comparison_by_d.csv"), index=False)
    pd.DataFrame(floors).to_csv(os.path.join(text_dir, "surrogate_floor_by_d.csv"), index=False)
    pd.DataFrame(timing).to_csv(os.path.join(text_dir, "timing_realizations.csv"), index=False)
    _write_run_config(text_dir, {"phase": 3, "realizations": 2, "overlap_bins": 20})


# ---------------------------------------------------------------- Phase 4
# Phase 4's percentage settings (`settings.pct_min_abs_denominator`, nats, and `settings.o_pct_min_cancellation_ratio`) and the dimensions of the KSG anchor.
PCT_MIN_DENOMINATOR = 1e-3
O_MIN_CANCELLATION = 0.1
KSG_DIMENSIONS = (2, 3)


def _cancellation(o_value, tc_value, dtc_value):
    """`cancellationRatio`: |O| / (|TC| + |DTC|), 0 where both are zero."""
    scale = np.abs(tc_value) + np.abs(dtc_value)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(scale > 0, np.abs(o_value) / scale, 0.0)


def _percentage(delta, denominator, stable, packed, fitted):
    """`percentageStatus` over a column: (reportable, status code, percentage).

    Code 1 is reportable, 2 a denominator that is not finite or below the floor, 3 an O-information whose cancellation ratio is under its bound, and 0 a measure never packed (no usable fit, or MI above d = 2), as `tInformationMeasuresFromLogdets` writes them.
    A failed unit's percentage stays finite, computed from its wrong but finite measures, beside its code 0, so the fixture can show whether a figure filters on the validity flag or only on `isfinite`.
    """
    denominator = np.asarray(denominator, dtype=float)
    usable = np.isfinite(denominator) & (np.abs(denominator) >= PCT_MIN_DENOMINATOR)
    code = np.where(packed, np.where(usable, np.where(stable, 1, 3), 2), 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 100.0 * np.asarray(delta, dtype=float) / denominator
    return code == 1, code, np.where((code == 1) | ~fitted, raw, np.nan)


def write_phase4(base, rng, recording, units, surrogate_floors):
    """The seven tables Phase 4 writes, with the columns of `Phase4_information_measures.m`.

    The per-unit tables carry the unit's key columns, the measure struct of `tInformationMeasuresFromLogdets` under the CSV spellings (`<M>_G`, `<M>_pct_status` as the numeric codes 0 to 3), the KSG anchor and the Phase 3 floor.
    The Gaussian values are drawn (TC, DTC) rather than rebuilt from the log-determinants, with the writer's identities O = TC - DTC and MI = TC at d = 2.
    """
    text_dir, _ = _phase_dirs(base, 4)
    frames = []
    for d in DIMENSIONS:
        block = units[units["d"] == d].copy()
        p = block["p_hat"].to_numpy()
        count = len(block)
        fitted = (block["numerical_ok"] == 1).to_numpy()
        gauss = {"TC": 0.05 * d * rng.uniform(0.5, 3.0, count), "DTC": 0.04 * d * rng.uniform(0.5, 3.0, count)}
        gauss["MI"] = gauss["TC"].copy() if d == 2 else np.full(count, np.nan)
        gauss["O"] = gauss["TC"] - gauss["DTC"]
        delta = _differences(d, p)
        fitted_value = {m: np.asarray(gauss[m], dtype=float) + np.asarray(delta[m], dtype=float) for m in MEASURES}
        ratio = _cancellation(gauss["O"], gauss["TC"], gauss["DTC"])
        ratio_t = _cancellation(fitted_value["O"], fitted_value["TC"], fitted_value["DTC"])
        out = {
            "dataset_key": recording["dataset_key"],
            "d": d,
            "unit_index": block["unit_index"].to_numpy(),
            "channels_str": block["channels_str"].to_numpy(),
            "n": block["n"].to_numpy(),
            "p_hat": p,
            "nu_hat": block["nu_hat"].to_numpy(),
            "rho": block["rho"].to_numpy(),
            "logdet_R": block["logdet_R"].to_numpy(),
            "sum_logdet_minors": block["sum_logdet_minors"].to_numpy(),
            "numerical_ok": block["numerical_ok"].to_numpy(),
            "status": block["status"].to_numpy(),
            "kappa": _kappa(d, p),
            "kappa_margin": _kappa(d - 1, p),
            "pct_min_abs_denominator_nats": PCT_MIN_DENOMINATOR,
            "O_pct_min_cancellation_ratio": O_MIN_CANCELLATION,
            "O_pct_cancellation_ratio": ratio,
            "O_pct_cancellation_ratio_t": ratio_t,
            "valid": _flag(fitted),
        }
        for measure in MEASURES:
            g = np.asarray(gauss[measure], dtype=float)
            dl = np.asarray(delta[measure], dtype=float)
            t_value = fitted_value[measure]
            packed = fitted & (measure != "MI" or d == 2)
            stable = ratio >= O_MIN_CANCELLATION if measure == "O" else np.ones(count, bool)
            stable_t = ratio_t >= O_MIN_CANCELLATION if measure == "O" else np.ones(count, bool)
            reportable, code, pct = _percentage(dl, g, stable, packed, fitted)
            reportable_t, code_t, pct_t = _percentage(dl, t_value, stable_t, packed, fitted)
            out.update(
                {
                    f"{measure}_G": g,
                    f"{measure}_t": t_value,
                    f"{measure}_delta_nats": dl,
                    f"{measure}_pct_denominator": g,
                    f"{measure}_pct_reportable": _flag(reportable),
                    f"{measure}_pct_status": code,
                    f"{measure}_delta_pct": pct,
                    f"{measure}_pct_denominator_t": t_value,
                    f"{measure}_pct_reportable_t": _flag(reportable_t),
                    f"{measure}_pct_status_t": code_t,
                    f"{measure}_delta_pct_t": pct_t,
                }
            )
        out = pd.DataFrame(out)
        # A failed unit keeps its wrong but finite measures here, where Phase 4 writes NaN, because a fixture whose failures are already NaN cannot show whether a figure filters on the validity flag or only on `isfinite`.
        out["ksg_MI"] = (out["MI_t"] + rng.normal(0.0, 0.01, count)) if d == 2 else np.nan
        out["ksg_TC"] = (out["TC_t"] + rng.normal(0.0, 0.02, count)) if d == 3 else np.nan
        # The Phase 3 floor joins by dimension: realization 1, 95th percentile.
        reported = surrogate_floors[(surrogate_floors["d"] == d) & (surrogate_floors["realization"] == 1)]
        for measure in MEASURES:
            value = reported.loc[reported["measure"] == measure, "q95"]
            out[f"floor95_{measure}"] = float(value.iloc[0]) if len(value) else np.nan
        frames.append(out)
    table = pd.concat(frames, ignore_index=True)
    table[table["d"] == 2].to_csv(os.path.join(text_dir, "im_pairs.csv"), index=False)
    table[table["d"] > 2].to_csv(os.path.join(text_dir, "im_long.csv"), index=False)

    summary, deciles, flips = [], [], []
    for d, block in table.groupby("d"):
        # `localSummaryByDimension`: location and spread of every difference, and the census of each denominator's status codes.
        row = {"d": d, "n_units": int(len(block)), "n_ok": int((block["valid"] == 1).sum())}
        for measure in MEASURES:
            for quantity in ("delta_nats", "delta_pct", "delta_pct_t"):
                values = block[f"{measure}_{quantity}"].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                statistics = (np.median(values), np.quantile(values, 0.25), np.quantile(values, 0.75), values.mean()) if values.size else (np.nan,) * 4
                for name, value in zip(("median", "q25", "q75", "mean"), statistics):
                    row[f"{measure}_{quantity}_{name}"] = float(value)
            for code in range(4):
                row[f"{measure}_pct_status_{code}"] = int((block[f"{measure}_pct_status"] == code).sum())
                row[f"{measure}_pct_status_t_{code}"] = int((block[f"{measure}_pct_status_t"] == code).sum())
        summary.append(row)
        for measure in MEASURES:
            nats = block[f"{measure}_delta_nats"].to_numpy()
            pct = block[f"{measure}_delta_pct"].to_numpy()
            gaussian = block[f"{measure}_G"].to_numpy()
            if not np.isfinite(nats).any():
                continue
            order = np.argsort(np.where(np.isfinite(gaussian), gaussian, np.inf))
            for decile, chunk in enumerate(np.array_split(order[np.isfinite(gaussian[order])], 10), start=1):
                if chunk.size == 0:
                    continue
                deciles.append({"d": d, "measure": measure, "decile": decile, "n": int(chunk.size), "median_delta_nats": float(np.nanmedian(nats[chunk])), "median_pct": _stat(np.median, pct[chunk]), "mean_before_ratio_pct": float(100.0 * np.nanmean(nats[chunk]) / np.nanmean(gaussian[chunk]))})
        # `localSignFlipTable`: only units with both values finite and non-zero can flip.
        gaussian, fitted = block["O_G"].to_numpy(), block["O_t"].to_numpy()
        comparable = np.isfinite(gaussian) & np.isfinite(fitted) & (gaussian != 0) & (fitted != 0)
        signs = np.sign(gaussian[comparable]) != np.sign(fitted[comparable])
        flips.append({"d": d, "n": int(comparable.sum()), "n_flips": int(signs.sum()), "rate": float(signs.mean()) if signs.size else np.nan})
    pd.DataFrame(summary).to_csv(os.path.join(text_dir, "im_summary_by_d.csv"), index=False)
    pd.DataFrame(deciles).to_csv(os.path.join(text_dir, "im_by_gaussian_decile.csv"), index=False)
    pd.DataFrame(flips).to_csv(os.path.join(text_dir, "o_sign_flips_by_d.csv"), index=False)

    grid_rows = []
    p_grid = np.round(np.arange(0.0, 3.0001, 0.05), 4)
    for d in DIMENSIONS:
        differences = _differences(d, p_grid)
        grid_rows.append(
            pd.DataFrame(
                {"p": p_grid, "nu": np.where(p_grid > 0, 1.0 / np.where(p_grid > 0, p_grid, 1.0), np.inf), "d": d, "kappa_d": _kappa(d, p_grid), "kappa_dm1": _kappa(d - 1, p_grid), "MI_delta": _differences(2, p_grid)["MI"], "TC_delta": differences["TC"], "DTC_delta": differences["DTC"], "O_delta": differences["O"]}
            )
        )
    pd.concat(grid_rows, ignore_index=True).to_csv(os.path.join(text_dir, "theory_grid.csv"), index=False)
    # `localKsgTable`: the anchor beside the two model values it is compared with, mutual information at d = 2 and total correlation at d = 3.
    ksg = table[table["d"].isin(KSG_DIMENSIONS)]
    pairs = (ksg["d"] == 2).to_numpy()
    pd.DataFrame(
        {
            "d": ksg["d"],
            "unit_index": ksg["unit_index"],
            "channels_str": ksg["channels_str"],
            "n": ksg["n"],
            "p_hat": ksg["p_hat"],
            "nu_hat": ksg["nu_hat"],
            "measure": np.where(pairs, "MI", "TC"),
            "ksg_nats": np.where(pairs, ksg["ksg_MI"], ksg["ksg_TC"]),
            "gauss_nats": np.where(pairs, ksg["MI_G"], ksg["TC_G"]),
            "t_nats": np.where(pairs, ksg["MI_t"], ksg["TC_t"]),
        }
    ).to_csv(os.path.join(text_dir, "ksg_low_d.csv"), index=False)
    _write_run_config(text_dir, {"phase": 4, "ksg_k": 6, "pct_min_abs_denominator": PCT_MIN_DENOMINATOR, "o_pct_min_cancellation_ratio": O_MIN_CANCELLATION})


# ---------------------------------------------------------------- Phase 5
def _copula_density(rho, p, centers):
    """A Gaussian copula density with a tail term, on the cell centres, as [u, v]."""
    x = np.array([_NORM.inv_cdf(float(c)) for c in centers])
    xx, yy = np.meshgrid(x, x, indexing="ij")
    r = float(np.clip(rho, -0.95, 0.95))
    quad = (np.square(xx) + np.square(yy) - 2 * r * xx * yy) / (1 - r * r)
    dens = np.exp(-0.5 * (quad - np.square(xx) - np.square(yy))) / np.sqrt(1 - r * r)
    tail = 1.0 + p * (quad - 2.0) / 6.0
    return dens * np.clip(tail, 0.2, 5.0)


def write_phase5(base, rng, recording, units):
    """Phase 5's pair selection, its summary and `pair_density.mat`, for the two pairs at the middle of each |tau| tertile of the ok pairs."""
    text_dir, mat_dir = _phase_dirs(base, 5)
    pairs = units[(units["d"] == 2) & (units["status"] == "ok")].copy()
    pairs["abs_tau"] = pairs["tau"].abs()
    pairs = pairs.sort_values("abs_tau")
    tertiles = np.array_split(pairs.index.to_numpy(), 3)
    edges = np.linspace(0.0, 1.0, GRID + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    selection, summary, rows = [], [], []
    for band, index_block in zip(("low", "mid", "high"), tertiles):
        chosen = pairs.loc[index_block]
        middle = len(chosen) // 2
        for rank, (_, unit) in enumerate(chosen.iloc[max(0, middle - 1) : middle + 1].iterrows(), start=1):
            rho, p = float(unit["rho"]), float(unit["p_hat"])
            gaussian = _copula_density(rho, 0.0, centers)
            student = _copula_density(rho, max(p, 0.02), centers)
            empirical = student * (1.0 + rng.normal(0.0, 0.05, student.shape))
            residuals = {"resid_emp_gauss": empirical - gaussian, "resid_emp_t": empirical - student, "resid_t_gauss": student - gaussian}
            rms = {name: float(np.sqrt(np.mean(np.square(value)))) for name, value in residuals.items()}
            selection.append({"band": band, "rank_in_band": rank, "channels_str": unit["channels_str"], "tau": float(unit["tau"]), "rho": rho, "p_hat": p, "nu_hat": float(unit["nu_hat"]), "n": int(unit["n"])})
            summary.append({"channels_str": unit["channels_str"], "band": band, "rms_emp_gauss": rms["resid_emp_gauss"], "rms_emp_t": rms["resid_emp_t"], "rms_t_gauss": rms["resid_t_gauss"], "ratio_gauss_over_t": rms["resid_emp_gauss"] / rms["resid_emp_t"]})
            rows.append(
                {
                    "channels": np.array([float(c) for c in unit["channels_str"].split()]),
                    "channels_str": unit["channels_str"],
                    "band": band,
                    "tau": float(unit["tau"]),
                    "rho": rho,
                    "p_hat": p,
                    "nu_hat": float(unit["nu_hat"]),
                    "n": float(unit["n"]),
                    "u_edges": edges,
                    "v_edges": edges,
                    # MATLAB writes [u x v] and h5py hands it back transposed, the orientation pcolormesh(u_edges, v_edges, C) expects.
                    "empirical": empirical.T,
                    "gaussian": gaussian.T,
                    "t": student.T,
                    **{name: value.T for name, value in residuals.items()},
                    "rms_emp_gauss": rms["resid_emp_gauss"],
                    "rms_emp_t": rms["resid_emp_t"],
                    "rms_t_gauss": rms["resid_t_gauss"],
                }
            )
    pd.DataFrame(selection).to_csv(os.path.join(text_dir, "pair_selection.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(text_dir, "pair_density_summary.csv"), index=False)
    write_struct_array(os.path.join(mat_dir, "pair_density.mat"), "pairs", rows)
    _write_run_config(text_dir, {"phase": 5, "grid": GRID})


# ---------------------------------------------------------------- Phase 6
# The suffix of the within-trial-centring twin of every Phase 6 curve and column, and the twins a trial recording writes as NaN.
# Centring each trial forces its within-trial autocovariances to sum to about zero, so no integrated time is defined there.
TRIAL = "_trialc"
WITHIN_TRIAL_UNDEFINED = ("tau_amp_int_samples", "tau_amp_int_s", "tau_amp_ref_int_samples", "tau_amp_ref_int_s", "amp_excess_int")


def _envelope_reading(score, floor):
    """`envelopeReading`: how every curve of one series is read, decided once from the score.

    'peaks' when the score dips below -floor and |score| has an interior local maximum (lag 0 and every such maximum, a flat top counted at its last sample); 'monotone', with lag 0 alone, otherwise.
    """
    magnitude = np.abs(score)
    interior = np.arange(1, score.size - 1)
    peaks = interior[(magnitude[interior] >= magnitude[interior - 1]) & (magnitude[interior] > magnitude[interior + 1])]
    if score.min() < -floor and peaks.size:
        return "peaks", np.concatenate([[0], peaks])
    return "monotone", np.array([0])


def _envelope_fit(acf, seconds, reading, floor):
    """`fitEnvelopeTimescale` on one curve, read the way the score's reading says.

    Peaks mode reads lag 0 and the score's peak lags up to the first at which |acf| falls below the floor (never fewer than two points), and monotone mode reads the curve itself up to its first value below the floor.
    The envelope joins the points read and is NaN beyond them, as Phase 6 writes it.
    tau is the time constant of the least-squares exponential exp(-t / tau), which is 1 at lag 0, found on a fine logarithmic grid (plausible, not the production optimiser).
    """
    mode, peak_index = reading
    magnitude = np.abs(acf)
    envelope = np.full(acf.size, np.nan)
    if mode == "peaks":
        low = np.flatnonzero(magnitude[peak_index] < floor)
        points = peak_index[: max(int(low[0]), 2)] if low.size else peak_index
        span = np.arange(points[-1] + 1)
        envelope[span] = np.interp(seconds[span], seconds[points], magnitude[points])
    else:
        low = np.flatnonzero(magnitude < floor)
        points = np.arange(max(int(low[0]) - 1, 1) + 1 if low.size else acf.size)
        envelope[points] = magnitude[points]
    x, y = seconds[points], magnitude[points]
    grid = np.geomspace(seconds[1] / 100.0, 100.0 * seconds[-1], 4000)
    tau = float(grid[np.argmin(((y[None, :] - np.exp(-x[None, :] / grid[:, None])) ** 2).sum(axis=1))])
    return {
        "tau_s": tau,
        "envelope": envelope,
        "fitted": np.exp(-seconds / tau),
        "last_s": float(x[-1]),
        "floor_reached": bool(low.size),
        "rmse": float(np.sqrt(np.mean((y - np.exp(-x / tau)) ** 2))),
        "mode": mode,
        "tau_ou_s": float(seconds[1] * (1.0 + 2.0 * np.sum(acf[1:] ** 2))),
        "tau_int_s": float(seconds[1] * (1.0 + 2.0 * np.sum(acf[1:]))),
        "n_points": int(np.size(points)),
    }


def write_phase6(base, recording):
    """The autocorrelation time scales of one recording, as the three tables Phase 6 writes.

    The score's autocorrelation is a cosine at the band centre under an exponential envelope, so it oscillates and every curve is read in peaks mode, at lag 0 and at the score's peaks, as the phase reads a band-passed recording.
    The square carries the recording's amplitude excess: on the score's peak lags it is exp(-lag / tau_amp) with tau_amp = excess x tau_env / 2, and its Gaussian reference is the squared score, whose time at those lags is tau_env / 2.
    The excess differs by recording, and so does tau_env, so the recordings land on different model lag windows.
    Each curve carries the envelope its fit reads (NaN beyond the fit range), the fitted exponential, an across-channel band and its Ornstein-Uhlenbeck-equivalent time.
    One channel is constant and the time scales fail on three others (the signal, the amplitude and the Gaussian reference each once), so the figures' counts of the channels without a value are drawn.
    Beside the fits, the integrated times of the square and its reference over the whole window give the integrated excess.

    Every curve, fit and time-scale column is written again for the within-trial centring (`_trialc`), read in the headline score's reading.
    Centring each of the segments of L samples on its own mean pulls a curve down by about its integrated time over L, as derived in the Phase 6 script, so on the trial recording the twin curves are the headline ones pulled down by that much and the reference is the square of the pulled score; a one-segment recording's twins equal its headline.
    The trial recording's within-trial integrated times and excess are NaN, as the phase writes them.
    The writer draws from its own generator, so it leaves every other phase's fixture unchanged.
    """
    rng = np.random.default_rng(600 + int(recording["seed_base"]))
    text_dir, _ = _phase_dirs(base, 6)
    fs = float(recording["fs"])
    n, segments = int(recording["n_samples"]), int(recording["n_segments"])
    max_lag = int(min(round(2.0 * fs), (n // segments) // 2))
    lags = np.arange(max_lag + 1)
    seconds = lags / fs
    floor = 0.05
    carrier = 0.5 * (recording["bandpass"][0] + recording["bandpass"][1])
    tau_env = 1.0 / carrier
    excess = 1.6 if recording["modality"] == "SEEG" else 1.2
    oscillation = np.cos(2.0 * np.pi * carrier * seconds)
    score = np.exp(-seconds / tau_env) * oscillation
    series = {"score": score, "square": np.exp(-seconds / (excess * tau_env / 2.0)) * (0.35 + 0.65 * oscillation**2), "square_gaussian": score**2}
    reading = _envelope_reading(score, floor)
    curves = {"lag_samples": lags, "lag_s": seconds, "pair_count": n - segments * lags}
    fits, spreads = {}, {}
    for transform, acf in series.items():
        fit = _envelope_fit(acf, seconds, reading, floor)
        spread = 0.06 + 0.04 * rng.random(lags.size)
        curves.update({f"acf_{transform}": acf, f"envelope_{transform}": fit["envelope"], f"fitted_{transform}": fit["fitted"], f"acf_{transform}_q10": acf - spread, f"acf_{transform}_q50": acf + 0.1 * spread, f"acf_{transform}_q90": acf + spread})
        fits[transform], spreads[transform] = fit, spread

    # The within-trial set: every lag pulled down by the curve's integrated time over the segment length (nothing on one segment), with the pulled score squared as the reference.
    length = (n / segments) / fs
    pull = {transform: fits[transform]["tau_int_s"] / length if segments > 1 else 0.0 for transform in ("score", "square")}
    twin_score = score - pull["score"]
    twin_series = {"score": twin_score, "square": series["square"] - pull["square"], "square_gaussian": twin_score**2}
    twin_fits = {}
    for transform, acf in twin_series.items():
        fit = _envelope_fit(acf, seconds, reading, floor)
        spread = spreads[transform]
        curves.update({f"{column}{TRIAL}": values for column, values in ((f"acf_{transform}", acf), (f"envelope_{transform}", fit["envelope"]), (f"fitted_{transform}", fit["fitted"]), (f"acf_{transform}_q10", acf - spread), (f"acf_{transform}_q50", acf + 0.1 * spread), (f"acf_{transform}_q90", acf + spread))})
        twin_fits[transform] = fit
    pd.DataFrame(curves).to_csv(os.path.join(text_dir, "acf_curves.csv"), index=False)

    count = int(recording["n_channels_retained"])
    row = _phase6_summary(recording, fits, fs, n, segments, max_lag, floor, count)
    twin = _phase6_summary(recording, twin_fits, fs, n, segments, max_lag, floor, count)
    if segments > 1:
        twin.update(dict.fromkeys(WITHIN_TRIAL_UNDEFINED, np.nan))
    headline = list(row)
    for column in headline[headline.index("tau_env_samples") :]:
        if column not in ("n_valid_channels", "env_fit_mode", "amp_fit_mode"):
            row[column + TRIAL] = twin[column]
    pd.DataFrame([row]).to_csv(os.path.join(text_dir, "timescale_summary.csv"), index=False)

    # A floor flag is 0 or 1 where the channel's fit ran and NaN where it did not, as the phase writes it.
    # A channel's within-trial time is its headline time scaled by the channel average's within-trial over headline ratio.
    table, twins = {"channel_id": np.arange(1, count + 1)}, {}
    for transform, role, failed in (("score", "env", 2), ("square", "amp", 4), ("square_gaussian", "amp_ref", 9)):
        status = np.full(count, "ok", dtype=object)
        error = np.full(count, "", dtype=object)
        status[6] = "constant_series"
        status[failed] = "fit_error"
        error[failed] = "fitEnvelopeTimescale:BoundaryFit"
        ok = status == "ok"
        tau = np.where(ok, fits[transform]["tau_s"] * np.exp(rng.normal(0.0, 0.2, count)), np.nan)
        for target, fit, scale in ((table, fits[transform], 1.0), (twins, twin_fits[transform], twin_fits[transform]["tau_s"] / fits[transform]["tau_s"])):
            target.update(
                {
                    f"tau_{role}_samples": tau * scale * fs,
                    f"tau_{role}_s": tau * scale,
                    f"{role}_fit_last_lag_s": np.where(ok, fit["last_s"], np.nan),
                    f"{role}_fit_rmse": np.where(ok, fit["rmse"], np.nan),
                    f"{role}_floor_reached": pd.Series(np.full(count, _flag(fit["floor_reached"])), dtype="Int64").mask(~ok),
                    f"{role}_status": status,
                    f"{role}_error": error,
                    f"{role}_fit_mode": np.where(ok, fit["mode"], ""),
                    f"tau_{role}_ou_s": np.where(ok, fit["tau_ou_s"], np.nan),
                }
            )
    # The integrated times need no fit: only the constant channel has none.
    spread = np.exp(rng.normal(0.0, 0.2, (2, count)))
    for target, fit in ((table, fits), (twins, twin_fits)):
        target["amp_excess"] = target["tau_amp_s"] / target["tau_amp_ref_s"]
        for k, (role, transform) in enumerate((("amp", "square"), ("amp_ref", "square_gaussian"))):
            value = fit[transform]["tau_int_s"] * spread[k]
            value[6] = np.nan
            target[f"tau_{role}_int_samples"] = value * fs
            target[f"tau_{role}_int_s"] = value
        target["amp_excess_int"] = target["tau_amp_int_s"] / target["tau_amp_ref_int_s"]
        for role, transform in (("env", "score"), ("amp", "square"), ("amp_ref", "square_gaussian")):
            target[f"{role}_fit_n_points"] = np.where(_status_ok(target, role), fit[transform]["n_points"], np.nan)
    if segments > 1:
        twins.update({column: np.full(count, np.nan) for column in WITHIN_TRIAL_UNDEFINED})
    for column in list(table)[1:]:
        if not column.endswith("_fit_mode"):
            table[column + TRIAL] = twins[column]
    pd.DataFrame(table).to_csv(os.path.join(text_dir, "timescale_by_channel.csv"), index=False)
    _write_run_config(text_dir, {"phase": 6, "max_lag_seconds": 2.0, "envelope_floor": floor, "envelope_method": "score_mode_peak_envelope", "centrings": ["recording", "segment"]})


def _status_ok(table, role):
    """True for every channel whose `<role>_status` is ok."""
    return np.asarray(table[f"{role}_status"], dtype=object) == "ok"


def _phase6_summary(recording, fits, fs, n, segments, max_lag, floor, count):
    """One row of `timescale_summary.csv` for one centring, in the phase's column order: the recording's metadata, the three fits, and the integrated times and point counts after them."""
    env, amp, ref = fits["score"], fits["square"], fits["square_gaussian"]
    return {
        "dataset_key": recording["dataset_key"],
        "modality": recording["modality"],
        "fs": fs,
        "n_samples": n,
        "duration_s": n / fs,
        "n_channels": count,
        "n_segments": segments,
        "max_lag_samples": max_lag,
        "max_lag_s": max_lag / fs,
        "envelope_floor": floor,
        "tau_env_samples": env["tau_s"] * fs,
        "tau_env_s": env["tau_s"],
        "tau_amp_samples": amp["tau_s"] * fs,
        "tau_amp_s": amp["tau_s"],
        "ratio_amp_env": amp["tau_s"] / env["tau_s"],
        "samples_per_tau": fs * env["tau_s"],
        "T_over_tau": (n / fs) / env["tau_s"],
        "env_fit_last_lag_s": env["last_s"],
        "env_fit_rmse": env["rmse"],
        "env_floor_reached": _flag(env["floor_reached"]),
        "amp_fit_last_lag_s": amp["last_s"],
        "amp_fit_rmse": amp["rmse"],
        "amp_floor_reached": _flag(amp["floor_reached"]),
        "n_valid_channels": count - 1,
        "env_fit_error": "",
        "amp_fit_error": "",
        "env_fit_mode": env["mode"],
        "amp_fit_mode": amp["mode"],
        "tau_env_ou_samples": env["tau_ou_s"] * fs,
        "tau_env_ou_s": env["tau_ou_s"],
        "tau_amp_ou_samples": amp["tau_ou_s"] * fs,
        "tau_amp_ou_s": amp["tau_ou_s"],
        "ratio_amp_env_ou": amp["tau_ou_s"] / env["tau_ou_s"],
        "tau_amp_ref_samples": ref["tau_s"] * fs,
        "tau_amp_ref_s": ref["tau_s"],
        "amp_excess": amp["tau_s"] / ref["tau_s"],
        "amp_ref_fit_last_lag_s": ref["last_s"],
        "amp_ref_fit_rmse": ref["rmse"],
        "amp_ref_floor_reached": _flag(ref["floor_reached"]),
        "amp_ref_fit_error": "",
        "tau_amp_int_samples": amp["tau_int_s"] * fs,
        "tau_amp_int_s": amp["tau_int_s"],
        "tau_amp_ref_int_samples": ref["tau_int_s"] * fs,
        "tau_amp_ref_int_s": ref["tau_int_s"],
        "amp_excess_int": amp["tau_int_s"] / ref["tau_int_s"],
        "env_fit_n_points": env["n_points"],
        "amp_fit_n_points": amp["n_points"],
        "amp_ref_fit_n_points": ref["n_points"],
    }


# ---------------------------------------------------------------- Phase 7
RATES = [5, 10, 20, 40, 80, 160]  # samples per tau, nested: every rate reads the same paths
STEP_RATE = 160  # the finest rate, whose interval carries the substep rule
REFINE = 2
PARENT_DIM = 20
# The recovery's equicorrelation (a plausible stand-in) and the base of the path seeds, which carry no rate: a replicate's path is one draw read at every rate and dimension.
EQUICORRELATION = 0.3
SEED_BASE = 1_600_000_000


def _radial_time(nu, D=PARENT_DIM):
    """`diffusionRadialTime`: the slowest radial relaxation time, in tau."""
    if not np.isfinite(nu):
        return 0.5
    m = nu + D - 2
    return 4 * m / nu**2 if nu <= 4 else m / (2 * (nu - 2))


def _phase7_summary_row(block, p_true, d, window, samples_per_tau, replicates, extra=None):
    """One `recovery_summary.csv` row, formed over the successful replicates as `simulationSummaries` forms it, keyed on the window T/tau and the rate.

    `extra` (the factorial's design and conditioning) follows `acceptance_median`, as in `r_factorial_summary.csv`.
    """
    p = block["p_hat"].to_numpy()
    iqr = float(np.quantile(p, 0.75) - np.quantile(p, 0.25))
    nu = (1.0 / p_true) if p_true > 0 else np.inf
    entry = {
        "cell": int(block["cell"].iloc[0]),
        "p_true": p_true,
        "nu_true": nu,
        "d": d,
        "T_over_tau": window,
        "samples_per_tau": samples_per_tau,
        "n": int(window * samples_per_tau),
        "refine": int(block["refine"].iloc[0]),
        "h": float(block["h"].iloc[0]),
        "reps": int(replicates),
        "n_ok": int(len(block)),
        "fraction_p0": float((p == 0).mean()),
        "fraction_heavy": float((block["heavy_boundary"] == 1).mean()),
        "median_bias": float(np.median(p) - p_true),
        "mean_bias": float(p.mean() - p_true),
        "sd": float(p.std(ddof=1)),
        "q25": float(np.quantile(p, 0.25)),
        "q75": float(np.quantile(p, 0.75)),
        "iqr": iqr,
        "relative_bias": float((np.median(p) - p_true) / p_true) if p_true > 0 else np.nan,
        "spread_law": float(iqr / p_true * np.sqrt(window)) if p_true > 0 else np.nan,
        "radial_time_tau": _radial_time(nu),
        "acceptance_median": float(block["acceptance"].median()),
        **(extra or {}),
    }
    for measure in MEASURES:
        delta_true = float(block[f"{measure}_delta_true"].iloc[0])
        full_true = float(block[f"{measure}_full_true"].iloc[0])
        delta_err = block[f"{measure}_delta_err"].to_numpy()
        full_err = block[f"{measure}_err"].to_numpy()
        delta_median = _stat(np.median, delta_err)
        full_median = _stat(np.median, full_err)
        entry[f"{measure}_delta_true"] = delta_true
        entry[f"{measure}_delta_median_error"] = delta_median
        entry[f"{measure}_delta_mean_error"] = _stat(np.mean, delta_err)
        entry[f"{measure}_delta_sd"] = _stat(lambda v: np.std(v, ddof=1), delta_err)
        entry[f"{measure}_delta_relative_error"] = _relative(delta_median, delta_true)
        entry[f"{measure}_full_median_error"] = full_median
        entry[f"{measure}_full_mean_error"] = _stat(np.mean, full_err)
        entry[f"{measure}_full_sd"] = _stat(lambda v: np.std(v, ddof=1), full_err)
        entry[f"{measure}_full_relative_error"] = _relative(full_median, full_true)
        entry[f"{measure}_p_only_median"] = _stat(np.median, block[f"{measure}_err_p_only"].to_numpy())
        entry[f"{measure}_R_only_median"] = _stat(np.median, block[f"{measure}_err_R_only"].to_numpy())
        entry[f"{measure}_R_only_max_abs"] = _stat(lambda v: np.max(np.abs(v)), block[f"{measure}_err_R_only"].to_numpy())
        entry[f"{measure}_interaction_median"] = _stat(np.median, block[f"{measure}_err_interaction"].to_numpy())
    return entry


def _phase7_path_columns(rng, p_true, window, samples_per_tau, replicates):
    """The per-replicate path facts Phase 7 writes beside every fit.

    Every path runs on the substep of the finest rate's interval, so `h` is the same at every rate and `substeps` (per observation) grows as the rate falls.
    Beside them come the Metropolis acceptance, the repeated-observation fraction and the seconds.
    """
    nu = (1.0 / p_true) if p_true > 0 else np.inf
    interval = 1.0 / STEP_RATE
    finest = 1 if not np.isfinite(nu) else int(np.ceil(interval / (0.005 * min(1.0, (nu + PARENT_DIM - 2) / (PARENT_DIM + 2), nu / (1 + nu)) / REFINE) - 1e-9))
    return {
        "refine": REFINE,
        "h": interval / finest,
        "substeps": finest * STEP_RATE // int(samples_per_tau),
        "parent_dim": PARENT_DIM,
        "acceptance": 1.0 if not np.isfinite(nu) else rng.uniform(0.95, 0.98, replicates),
        "repeated_observation_fraction": 0.0,
        "sim_seconds": 4e-7 * finest * window * STEP_RATE * rng.uniform(0.9, 1.1, replicates),
        "prep_seconds": 1e-4 * window * rng.uniform(0.9, 1.1, replicates),
    }


def _fit_columns(p_hat, failure, heavy, n, d, cond, min_eig, logdet_true, logdet_hat, repair, seconds):
    """The fit's columns of a Phase 7 long table, `p_hat` to `fit_seconds`, in the writer's order.

    The log-likelihoods are a plausible stand-in: the t nests the Gaussian, so a snapped fit ties it and the AIC difference is -2.
    """
    loglik_gaussian = np.full(p_hat.size, -0.02 * n * d)
    loglik_t = loglik_gaussian + 0.05 * n * np.square(p_hat)
    return {
        "p_hat": p_hat,
        "nu_hat": np.where(p_hat > 0, 1.0 / np.where(p_hat > 0, p_hat, 1.0), np.inf),
        "numerical_ok": _flag(~failure),
        "heavy_boundary": _flag(heavy),
        "gaussian_snapped": _flag(p_hat == 0),
        "ambiguous_profile": _flag(np.zeros(p_hat.size, bool)),
        "status": np.where(failure, "numerical_failure", np.where(heavy, "heavy_boundary", "ok")),
        "logdet_R_true": logdet_true,
        "logdet_R_hat": logdet_hat,
        "cond_R": cond,
        "min_eig_R": min_eig,
        "repair_shift_rms": repair,
        "loglik_t": loglik_t,
        "loglik_gaussian": loglik_gaussian,
        "aic_diff": 2.0 * (loglik_t - loglik_gaussian) - 2.0,
        "fit_seconds": seconds,
    }


def _check_law_tables(rng, paths):
    """`check_law.csv` and its PIT histogram.

    The law part covers every nu, and the coarse part, at nu = 1/3 and D = 2 and 20, sets the Metropolis chain, which keeps the law, beside the uncorrected chain of scheme 'frozen', the negative control, which has lost it at the later time.
    """
    edges = np.linspace(0.0, 1.0, 21)
    law, histogram = [], []

    def add(part, scheme, nu, D, time, counts, marginal_ks, h, substeps, refine, rate, acceptance, nonfinite):
        radial_ks = float(np.max(np.abs(np.cumsum(counts) / paths - edges[1:])))
        law.append(
            {
                "part": part,
                "scheme": scheme,
                "nu": nu,
                "D": D,
                "time_tau": time,
                "n_paths": paths,
                "samples_per_tau": rate,
                "refine": refine,
                "h": h,
                "substeps": substeps,
                "radial_pit_mean": float(np.sum(counts * (edges[:-1] + 0.025)) / paths),
                "radial_ks": radial_ks,
                "ks_critical_95": 1.3581 / np.sqrt(paths),
                "ks_critical_99": 1.6276 / np.sqrt(paths),
                "marginal_ks": marginal_ks,
                "direction_u1sq_mean": 1.0 / D + rng.normal(0, 0.02 / D),
                "direction_u1sq_se": 0.022 / D,
                "direction_u1sq_target": 1.0 / D,
                "acceptance_mean": acceptance,
                "nonfinite_fraction": nonfinite,
            }
        )
        histogram.extend({"part": part, "scheme": scheme, "nu": nu, "D": D, "time_tau": time, "bin_lower": edges[k], "bin_upper": edges[k + 1], "count": int(counts[k]), "expected": paths / 20} for k in range(20))

    uniform = np.full(20, 1 / 20)
    for nu in [1 / 3, 1.0, 2.0, 4.0, 10.0, np.inf]:
        for time in (0.0, 50.0):
            add("law", "metropolis", nu, 20, time, rng.multinomial(paths, uniform), abs(rng.normal(0.012, 0.005)), 0.0125 / 6, 48, REFINE, 10, 1.0 if not np.isfinite(nu) else 0.97, 0.0)
    for D, lost, nonfinite in ((2, np.r_[np.full(19, 0.002), 0.962], 0.03), (20, np.linspace(0.2, 1.8, 20) / 20, 0.0)):
        start = rng.multinomial(paths, uniform)
        for scheme in ("metropolis", "frozen"):
            for time in (0.0, 50.0):
                late = time > 0
                if not late:
                    counts = start  # both chains leave from the same exact start
                elif scheme == "metropolis":
                    counts = rng.multinomial(paths, uniform)
                else:
                    counts = rng.multinomial(paths, lost / lost.sum())
                marginal = abs(rng.normal(0.012, 0.005)) + (0.3 if late and scheme == "frozen" and D == 2 else 0.0)
                add("coarse", scheme, 1 / 3, D, time, counts, marginal, 0.5, 1, 1, 2, 1.0 if scheme == "frozen" else (0.37 if D == 2 else 0.76), nonfinite if late and scheme == "frozen" else 0.0)
    frame = pd.DataFrame(law)
    # The Bonferroni family is every KS distance of the corrected chain, two per row.
    # Every row carries the family's critical value, the 'frozen' rows outside the family included, so their `exceeds` flags compare them with it, as `simulatorCheckSummaries` writes it.
    in_family = (frame["scheme"] == "metropolis").to_numpy()
    family = 2 * int(in_family.sum())
    frame["family_size"] = family
    frame["ks_critical_bonferroni"] = np.sqrt(-0.5 * np.log(0.05 / family / 2)) / np.sqrt(frame["n_paths"])
    critical = frame["ks_critical_bonferroni"].to_numpy()
    frame["radial_exceeds_bonferroni"] = _flag(frame["radial_ks"].to_numpy() > critical)
    frame["marginal_exceeds_bonferroni"] = _flag(frame["marginal_ks"].to_numpy() > critical)
    return frame, pd.DataFrame(histogram)


def _check_fitted_row(varied, nu, window, rate, refine, d, sd, median, reps):
    """One row of `check_refinement.csv` or `check_sampling.csv`."""
    gaussian = not np.isfinite(nu)
    return {
        "varied": varied,
        "nu": nu,
        "T_over_tau": window,
        "samples_per_tau": rate,
        "n": window * rate,
        "refine": refine,
        "d": d,
        "reps": reps,
        "n_ok": reps,
        "mean": median + (0.2 * sd if gaussian else 0.0),
        "sd": sd,
        "q05": 0.0 if gaussian else median - 1.6 * sd,
        "q25": 0.0 if gaussian else median - 0.67 * sd,
        "median": median,
        "q75": median + (1.2 if gaussian else 0.67) * sd,
        "q95": median + 1.6 * sd,
        "fraction_p0": 0.45 if gaussian else 0.0,
        "fraction_heavy": 0.0,
        "acceptance_median": 1.0 if gaussian else 0.97,
        "repeated_median": 0.0,
        "substeps": 6 * refine * STEP_RATE // (REFINE * rate),
        "h": 0.0125 / (6 * refine / REFINE),
        "sim_seconds_median": 0.01 * window,
        "fit_seconds_median": 0.001 * window,
    }


def _phase7_check_tables(rng, text_dir):
    """The ten `check_*.csv` tables of the simulator check, plausible, not scientific."""
    law, histogram = _check_law_tables(rng, paths=2000)
    law.to_csv(os.path.join(text_dir, "check_law.csv"), index=False)
    histogram.to_csv(os.path.join(text_dir, "check_law_pit_histogram.csv"), index=False)

    dims = [2, 5, 10, 20]
    windows = [300, 3000]
    refinement, sampling, equivalence, pooled = [], [], [], []
    for nu in (1 / 3, 2.0, 10.0):
        for window in windows:
            for refine in (1, 2, 4):
                for d in dims:
                    sd = 0.6 / np.sqrt(window) * (1 + 1 / nu)
                    median = 1 / nu + 0.5 / np.sqrt(window) + rng.normal(0, sd / 20)
                    refinement.append(_check_fitted_row("refine", nu, window, 10, refine, d, sd, median, 200))
            for coarse, fine in ((1, 2), (2, 4)):
                for d in dims:
                    ratio, se = rng.normal(0.0, 0.03), 0.03
                    equivalence.append(
                        {
                            "nu": nu,
                            "T_over_tau": window,
                            "d": d,
                            "refine_coarse": coarse,
                            "refine_fine": fine,
                            "n_pairs": 200,
                            "mean_coarse": 1 / nu,
                            "mean_fine": 1 / nu,
                            "difference": ratio * 0.01,
                            "sd_fine": 0.01,
                            "difference_over_sd": ratio,
                            "paired_se_over_sd": se,
                            "ci95_low_over_sd": ratio - 1.96 * se,
                            "ci95_high_over_sd": ratio + 1.96 * se,
                            "tolerance_sd": 0.1,
                            "within_tolerance": _flag(abs(ratio) <= 0.1),
                            "interval_within_tolerance": _flag(abs(ratio) + 1.96 * se <= 0.1),
                        }
                    )
        for coarse, fine in ((1, 2), (2, 4)):
            difference, se = rng.normal(0.0, 0.02), 0.025
            ratio = float(np.exp(rng.normal(0.0, 0.02)))
            pooled.append(
                {
                    "nu": nu,
                    "refine_coarse": coarse,
                    "refine_fine": fine,
                    "n_windows": len(windows),
                    "n_d": len(dims),
                    "n_pairs_min": 200,
                    "pooled_difference_over_sd": difference,
                    "pooled_se_over_sd": se,
                    "ci95_low_over_sd": difference - 1.96 * se,
                    "ci95_high_over_sd": difference + 1.96 * se,
                    "tolerance_sd": 0.1,
                    "interval_within_tolerance": _flag(abs(difference) + 1.96 * se <= 0.1),
                    "sd_ratio_coarse_over_fine": ratio,
                    "sd_ratio_ci95_low": ratio * 0.95,
                    "sd_ratio_ci95_high": ratio * 1.05,
                }
            )
    for nu in (2.0, 5.0, np.inf):
        for window in windows:
            for rate in RATES:
                for d in dims:
                    sd = 0.6 / np.sqrt(window) * (1 + (1 / nu if np.isfinite(nu) else 0.0)) * np.sqrt(1 + 4.0 / rate)
                    p = 1 / nu if np.isfinite(nu) else 0.0
                    median = p + (0.3 * sd if not np.isfinite(nu) else rng.normal(0, sd / 20))
                    sampling.append(_check_fitted_row("samples_per_tau", nu, window, rate, REFINE, d, sd, median, 200))
    refinement = pd.DataFrame(refinement)
    refinement.to_csv(os.path.join(text_dir, "check_refinement.csv"), index=False)
    pd.DataFrame(sampling).to_csv(os.path.join(text_dir, "check_sampling.csv"), index=False)
    pd.DataFrame(equivalence).to_csv(os.path.join(text_dir, "check_refinement_equivalence.csv"), index=False)
    pd.DataFrame(pooled).to_csv(os.path.join(text_dir, "check_refinement_pooled.csv"), index=False)

    # The chain's clock runs slow by kappa, less so at a finer step: the conditional mean decays like exp(-(1 - kappa) t), so its ratio to the exact exp(-t) is exp(kappa t).
    relaxation = []
    for nu in (1 / 3, 2.0, 10.0):
        for refine in (1, 2, 4):
            kappa = (0.034 if nu < 1 else 0.012) / np.sqrt(refine)
            for t in (0.5, 1.0, 2.0):
                error = float(np.expm1(kappa * t) + rng.normal(0, 0.002))
                error_se = 0.02 / (4.5 * np.exp(-t))
                relaxation.append(
                    {
                        "nu": nu,
                        "D": PARENT_DIM,
                        "refine": refine,
                        "t_tau": t,
                        "n_paths": 2000,
                        "start_quantile": 0.5,
                        "y1_start": 4.5,
                        "y1_mean": 4.5 * np.exp(-t) * (1 + error),
                        "y1_se": 0.02,
                        "y1_expected": 4.5 * np.exp(-t),
                        "relative_error": error,
                        "relative_error_se": error_se,
                        "clock_error": np.log1p(error) / t,
                        "clock_error_se": error_se / ((1 + error) * t),
                        "acceptance_mean": 0.97,
                    }
                )
    relaxation = pd.DataFrame(relaxation)
    relaxation.to_csv(os.path.join(text_dir, "check_relaxation.csv"), index=False)

    # The discretisation bound, as `simulatorCheckSummaries` forms it from the two tables.
    # Its se is the phase's standard error across paths of each path's own combination of the times (`localClockErrorSe`).
    # The fixture has no paths, so it uses a plausible stand-in: 1.28 times the standard error of the mean of the per-time clock errors, taken as independent.
    discretisation = []
    for (nu, refine), clock in relaxation.groupby(["nu", "refine"]):
        kappa = float(clock["clock_error"].mean())
        kappa_se = 1.28 * float(np.sqrt(np.sum(clock["clock_error_se"] ** 2)) / len(clock))
        block = refinement[(refinement["nu"] == nu) & (refinement["refine"] == refine)]
        for d, at_d in block.groupby("d"):
            short = at_d[at_d["T_over_tau"] == min(windows)]["mean"].iloc[0]
            long = at_d[at_d["T_over_tau"] == max(windows)]["mean"].iloc[0]
            slope = (long - short) / np.log(max(windows) / min(windows))
            for window in windows:
                sd = float(at_d[at_d["T_over_tau"] == window]["sd"].iloc[0])
                discretisation.append({"nu": nu, "refine": refine, "d": d, "T_over_tau": window, "clock_error": kappa, "clock_error_se": kappa_se, "slope_mean_p_per_log_T": slope, "sd_p_hat": sd, "mean_shift": -kappa * slope, "mean_shift_over_sd": abs(kappa * slope) / sd})
    pd.DataFrame(discretisation).to_csv(os.path.join(text_dir, "check_discretisation.csv"), index=False)

    # The model's time scales by the Phase 6 estimator, per nu and lag window.
    # The model cannot oscillate, so the score reads every curve in monotone mode.
    # The amplitude memory of the heavy tails grows with the window it is read over, the Gaussian reference is about half the signal time, and the excess is 1 at Gaussian truth.
    # The integrated times (in tau) count a slow component's duration as well, so at heavy tails the integrated excess outgrows the fitted one; at Gaussian truth it is 1.
    timescale = []
    for nu, env, amp_far in ((1 / 3, 0.76, 330.0), (1.0, 0.93, 36.0), (2.0, 0.97, 6.5), (4.0, 1.04, 1.3), (10.0, 1.02, 0.62), (np.inf, 1.0, 0.5)):
        for window in (2, 5, 10, 20, 50, 100):
            tau_env = env * (1 + rng.normal(0, 0.005))
            tau_ref = 0.5 * tau_env * (0.85 if nu < 1 else 1.0)
            tau_amp = max(tau_ref, 0.5 + (amp_far - 0.5) * window / (window + amp_far / 3))
            tau_amp_int = tau_amp * (1.0 + 2.0 * (1.0 - tau_ref / tau_amp) * window / (window + 20.0))
            timescale.append(
                {
                    "nu": nu,
                    "D": PARENT_DIM,
                    "T_over_tau": 1000,
                    "n_paths": 20,
                    "samples_per_tau": 10,
                    "refine": REFINE,
                    "window_tau": window,
                    "floor": 0.05,
                    "reading_mode": "monotone",
                    "tau_env": tau_env,
                    "tau_amp": tau_amp,
                    "tau_amp_ref": tau_ref,
                    "amp_excess": tau_amp / tau_ref,
                    "amp_over_env": tau_amp / tau_env,
                    "env_floor_reached": _flag(3 * tau_env < window),
                    "amp_floor_reached": _flag(3 * tau_amp < window),
                    "amp_ref_floor_reached": _flag(3 * tau_ref < window),
                    "env_last_lag_tau": min(3 * tau_env, window),
                    "amp_last_lag_tau": min(3 * tau_amp, window),
                    "amp_ref_last_lag_tau": min(3 * tau_ref, window),
                    "env_rmse": 0.01,
                    "amp_rmse": 0.02,
                    "amp_ref_rmse": 0.01,
                    "env_message": "",
                    "amp_message": "",
                    "amp_ref_message": "",
                    "tau_env_ou": 0.9 * tau_env,
                    "tau_amp_ou": 0.9 * tau_amp,
                    "tau_amp_ref_ou": 0.9 * tau_ref,
                    "amp_over_env_ou": tau_amp / tau_env,
                    "radial_time_tau": _radial_time(nu),
                    "acceptance_mean": 1.0 if not np.isfinite(nu) else 0.97,
                    "tau_amp_int": tau_amp_int,
                    "tau_amp_ref_int": tau_ref,
                    "amp_excess_int": tau_amp_int / tau_ref,
                }
            )
    pd.DataFrame(timescale).to_csv(os.path.join(text_dir, "check_timescale.csv"), index=False)


def write_phase7(root, rng):
    """Phase 7: the diffusion study on the (p, d, T/tau, samples per tau) grid, with the simulator check's tables.

    The rates are nested as the study's are: a replicate is one path read at every rate, so its estimates share one draw across the rates and a denser rate sharpens it only a little.
    A failed fit keeps a finite estimate at the far end of the domain, so every quantitative panel has to drop it and every status panel has to keep it.
    """
    base = os.path.join(root, "simulation", "t_copula", "all", "Phase7")
    text_dir, _ = _phase_dirs(os.path.dirname(base), 7)
    p_grid = [0.0, 0.05, 0.2, 1.0]
    dims = [2, 3, 10, 20]
    windows = [10, 300, 3000]
    replicates = 20
    rows = []
    for ip, p_true in enumerate(p_grid):
        for iw, window in enumerate(windows):
            for d in dims:
                common = rng.normal(0.0, 1.0, replicates)
                atom = rng.random(replicates) < 0.6
                tail = rng.random(replicates) < (0.03 if p_true >= 0.2 else 0.0)
                for ir, rate in enumerate(RATES):
                    job = 1 + (ip * len(windows) + iw) * len(RATES) + ir
                    n = window * rate
                    spread = 3.2 / np.sqrt(10 * window) * np.sqrt(1 + 4.0 / rate) * (1.0 - 0.02 * (d - 2))
                    draw = np.maximum(0.0, p_true + spread * (0.95 * common + 0.31 * rng.normal(0.0, 1.0, replicates)))
                    if p_true == 0:
                        draw[atom] = 0.0
                    draw[tail] = rng.uniform(2.96, 3.0, int(tail.sum()))
                    draw = np.minimum(draw, 3.0)
                    failure = rng.random(replicates) < 0.02
                    draw[failure] = 3.0
                    heavy = draw > 2.95
                    logdet_true = np.full(replicates, -0.1 * d)
                    logdet_hat = logdet_true + rng.normal(0.0, 0.2 / np.sqrt(n), replicates)
                    path = _phase7_path_columns(rng, p_true, window, rate, replicates)
                    repair = np.abs(rng.normal(0.0, 0.002, replicates))
                    seconds = rng.gamma(3.0, 0.002 * d, replicates)
                    rho = EQUICORRELATION
                    block = {
                        "cell": (job - 1) * len(dims) + dims.index(d) + 1,
                        "job": job,
                        "p_true": p_true,
                        "nu_true": (1.0 / p_true) if p_true > 0 else np.inf,
                        "d": d,
                        "T_over_tau": window,
                        "samples_per_tau": rate,
                        "n": n,
                        **{k: path[k] for k in ("refine", "h", "substeps", "parent_dim")},
                        "rep": np.arange(1, replicates + 1),
                        "seed": SEED_BASE + (ip * len(windows) + iw) * replicates + np.arange(1, replicates + 1),
                    }
                    block.update(_fit_columns(draw, failure, heavy, n, d, (1 + (d - 1) * rho) / (1 - rho), 1 - rho, logdet_true, logdet_hat, repair, seconds))
                    block.update({k: path[k] for k in ("acceptance", "repeated_observation_fraction", "sim_seconds", "prep_seconds")})
                    block.update(_error_columns(d, np.full(replicates, p_true), draw, logdet_true, logdet_hat))
                    rows.append(pd.DataFrame(block))
    recovery = pd.concat(rows, ignore_index=True)
    recovery.to_csv(os.path.join(text_dir, "recovery_long.csv"), index=False)

    # Every summary statistic below is formed over the numerically successful replicates, the way `simulationSummaries` forms it, per (p, d, T/tau, rate).
    ok_recovery = recovery[recovery["numerical_ok"] == 1]
    summary, floors, spread_law = [], [], []
    for (p_true, d, window, rate), block in ok_recovery.groupby(["p_true", "d", "T_over_tau", "samples_per_tau"]):
        summary.append(_phase7_summary_row(block, p_true, d, window, rate, replicates))
        if p_true == 0:
            # The floor is the correction the estimator fits on Gaussian data; the full-measure error travels beside it, never in its place.
            for measure in MEASURES:
                fitted = block[f"{measure}_delta_hat"].to_numpy()
                full_err = block[f"{measure}_err"].to_numpy()
                if not np.isfinite(fitted).any():
                    continue
                floors.append({"d": d, "T_over_tau": window, "samples_per_tau": rate, "n": window * rate, "measure": measure, "n_ok": int(np.isfinite(fitted).sum()), "q95": _q(fitted, 0.95), "q99": _q(fitted, 0.99), "q95_full_err": _q(full_err, 0.95), "q99_full_err": _q(full_err, 0.99)})
    pd.DataFrame(summary).to_csv(os.path.join(text_dir, "recovery_summary.csv"), index=False)
    pd.DataFrame(floors).to_csv(os.path.join(text_dir, "floor_by_d_T_s.csv"), index=False)

    for (d, rate), block in ok_recovery[ok_recovery["p_true"] > 0].groupby(["d", "samples_per_tau"]):
        constants = []
        for (p_true, window), cellblock in block.groupby(["p_true", "T_over_tau"]):
            p = cellblock["p_hat"].to_numpy()
            iqr = np.quantile(p, 0.75) - np.quantile(p, 0.25)
            constants.append(iqr / p_true * np.sqrt(window))
        spread_law.append({"d": d, "samples_per_tau": rate, "spread_constant": float(np.median(constants)), "q25": float(np.quantile(constants, 0.25)), "q75": float(np.quantile(constants, 0.75)), "n_cells": len(constants)})
    pd.DataFrame(spread_law).to_csv(os.path.join(text_dir, "spread_law_by_d_s.csv"), index=False)

    designs = [("equi_0.10", 1.4, 0.9), ("equi_0.75", 12.0, 0.25), ("block_four", 30.0, 0.08)]
    factorial_rows, factorial_summary = [], []
    p_true, window, rate = 0.1, 3000, 10
    n = window * rate
    for index, (design, cond, min_eig) in enumerate(designs):
        for d in dims:
            draw = np.maximum(0.0, rng.normal(p_true * (1 - 0.01 * cond / 10), 0.02, replicates))
            failure = rng.random(replicates) < 0.02
            draw[failure] = 3.0
            heavy = draw > 2.95
            logdet_true = np.full(replicates, -0.1 * d)
            logdet_hat = logdet_true + rng.normal(0.0, 0.2 / np.sqrt(n), replicates)
            path = _phase7_path_columns(rng, p_true, window, rate, replicates)
            block = {
                "cell": index * 19 + d - 1,
                "design": design,
                "job": index + 1,
                "p_true": p_true,
                "nu_true": 1.0 / p_true,
                "d": d,
                "T_over_tau": window,
                "samples_per_tau": rate,
                "n": n,
                **{k: path[k] for k in ("refine", "h", "substeps", "parent_dim")},
                "rep": np.arange(1, replicates + 1),
                "seed": SEED_BASE + 10_000 * (index + 1) + np.arange(1, replicates + 1),
            }
            block.update(_fit_columns(draw, failure, heavy, n, d, cond, min_eig, logdet_true, logdet_hat, np.full(replicates, 0.001 * cond), np.full(replicates, 0.002 * d)))
            block.update({k: path[k] for k in ("acceptance", "repeated_observation_fraction", "sim_seconds", "prep_seconds")})
            block.update(_error_columns(d, np.full(replicates, p_true), draw, logdet_true, logdet_hat))
            block = pd.DataFrame(block)
            factorial_rows.append(block)
            # The summary is the recovery summary's row, formed over the successful replicates, with the design and its conditioning after the acceptance.
            factorial_summary.append(_phase7_summary_row(block[block["numerical_ok"] == 1], p_true, d, window, rate, replicates, extra={"design": design, "cond_R": cond, "min_eig_R": min_eig}))
    pd.concat(factorial_rows, ignore_index=True).to_csv(os.path.join(text_dir, "r_factorial_long.csv"), index=False)
    pd.DataFrame(factorial_summary).to_csv(os.path.join(text_dir, "r_factorial_summary.csv"), index=False)

    _phase7_check_tables(rng, text_dir)
    _write_run_config(text_dir, {"phase": 7, "replicates": replicates, "samples_per_tau": RATES, "refine": REFINE, "step_samples_per_tau": STEP_RATE})


# ---------------------------------------------------------------- tree plumbing
def _phase_dirs(base, phase):
    """Create `<base>/Phase<phase>/{text,mat,images}`; return the text and mat folders."""
    phase_dir = os.path.join(base, f"Phase{phase}")
    for child in ("text", "mat", "images"):
        os.makedirs(os.path.join(phase_dir, child), exist_ok=True)
    return os.path.join(phase_dir, "text"), os.path.join(phase_dir, "mat")


def _write_run_config(text_dir, settings):
    """Write `run_config.json` with the given settings and placeholder provenance."""
    payload = {"settings": settings, "env": {"TC_RESULTS_ROOT": "(fixture)"}, "provenance": {"git_sha": "unknown", "matlab_release": "(fixture)", "hostname": "(fixture)", "n_workers": 0, "started_utc": "2026-09-22T00:00:00+00:00", "finished_utc": "2026-09-22T00:00:00+00:00"}}
    with open(os.path.join(text_dir, "run_config.json"), "w") as handle:
        json.dump(payload, handle, indent=2)


def write_unit_columns(path, name, columns):
    """Write a MATLAB-style `-v7.3` column struct of unit rows, the layout `unitRowsToColumns` produces.

    Each field is one dataset in MATLAB's column-major order, so an `[n x w]` array is written with HDF5 dims `(w, n)` and an `[n x 1]` column with `(1, n)`.
    A text column becomes int8 codes beside a sorted cellstr `<name>_labels`, and `dataset_key` is one char array, stored once.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["MATLAB_class"] = np.bytes_("struct")
        store = handle.create_group("#refs#")
        group = handle.create_group(name)
        group.attrs["MATLAB_class"] = np.bytes_("struct")
        for field, value in columns.items():
            if isinstance(value, str):
                _char_dataset(group, field, value)
            elif _is_text_column(value):
                _code_column(group, store, field, list(value))
            else:
                block = np.asarray(value, dtype=float)
                data = block.reshape(1, -1) if block.ndim == 1 else block.T
                node = group.create_dataset(field, data=data)
                node.attrs["MATLAB_class"] = np.bytes_("double")


def _is_text_column(value):
    """True for a non-empty list, tuple or array whose first element is a string."""
    return isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0 and isinstance(value[0], str)


def _char_dataset(parent, key, text):
    """One MATLAB char array, the way `-v7.3` stores one."""
    data = np.array([ord(c) for c in text], dtype=np.uint16).reshape(-1, 1)
    node = parent.create_dataset(key, data=data)
    node.attrs["MATLAB_class"] = np.bytes_("char")
    node.attrs["MATLAB_int_decode"] = 2
    return node


def _code_column(group, store, field, values):
    """A text column as int8 codes into a sorted cellstr of its distinct values."""
    labels = sorted(set(values))
    codes = np.array([labels.index(v) + 1 for v in values], dtype=np.int8)
    node = group.create_dataset(field, data=codes.reshape(1, -1))
    node.attrs["MATLAB_class"] = np.bytes_("int8")
    references = [_char_dataset(store, f"{field}_label_{i}", label).ref for i, label in enumerate(labels)]
    cell = group.create_dataset(f"{field}_labels", data=np.array(references, dtype=h5py.ref_dtype).reshape(-1, 1))
    cell.attrs["MATLAB_class"] = np.bytes_("cell")


def write_struct_array(path, name, rows):
    """Write a MATLAB-style `-v7.3` struct array: one dataset per field, holding an object reference per element."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["MATLAB_class"] = np.bytes_("struct")
        store = handle.create_group("#refs#")
        group = handle.create_group(name)
        for field in rows[0].keys():
            references = []
            for index, row in enumerate(rows):
                value = row[field]
                if isinstance(value, str):
                    data = np.array([ord(c) for c in value], dtype=np.uint16).reshape(-1, 1)
                else:
                    data = np.atleast_2d(np.asarray(value, dtype=float))
                dataset = store.create_dataset(f"{field}_{index}", data=data)
                references.append(dataset.ref)
            group.create_dataset(field, data=np.array(references, dtype=h5py.ref_dtype).reshape(len(rows), 1))


def write_phase1(base, rng, recording):
    """Phase 1's channel, autocorrelation and band tables; return the path of Phase 1's MAT file, which is not written."""
    text_dir, mat_dir = _phase_dirs(base, 1)
    channels = np.arange(1, recording["n_channels_raw"] + 1)
    retained = channels <= recording["n_channels_retained"]
    pd.DataFrame(
        {
            "channel_id": channels,
            "label": [f"ch{c}" for c in channels],
            "retained": _flag(retained),
            "exclusion_reason": np.where(retained, "", "low_power"),
            "kurtosis": rng.normal(3.0, 0.4, channels.size),
            "robust_std": rng.uniform(0.8, 1.2, channels.size),
            "low_power": _flag(~retained),
            "abnormal_kurtosis": _flag(np.zeros(channels.size, bool)),
            "isolation": _flag(np.zeros(channels.size, bool)),
            "redundancy": _flag(np.zeros(channels.size, bool)),
            "temporal_rate": rng.uniform(0.0, 3.0, channels.size),
        }
    ).to_csv(os.path.join(text_dir, "channels.csv"), index=False)
    kept = channels[retained]
    tau = rng.uniform(2.0, 9.0, kept.size)
    pd.DataFrame({"channel_id": kept, "tau_int_samples": tau, "n_eff": recording["n_samples"] / tau, "n_samples": recording["n_samples"], "n_segments": recording["n_segments"]}).to_csv(os.path.join(text_dir, "autocorrelation.csv"), index=False)
    pd.DataFrame(
        [
            {
                "dataset_key": recording["dataset_key"],
                "modality": recording["modality"],
                "fs": recording["fs"],
                "filter_signal": _flag([recording["filter_signal"]])[0],
                "bandpass": f"{recording['bandpass'][0]} {recording['bandpass'][1]}",
                "native_band": f"{recording['native_band'][0]} {recording['native_band'][1]}",
                "filter_range": f"{recording['filter_range'][0]} {recording['filter_range'][1]}",
                "filter_order": 4,
                "method": "butterworth",
                "band_selection_mode": "fixed",
            }
        ]
    ).to_csv(os.path.join(text_dir, "band_provenance.csv"), index=False)
    _write_run_config(text_dir, {"phase": 1, "schema": "2.3.0"})
    return os.path.join(mat_dir, "preprocessed_data.mat")


def build(root, seed=20260922):
    """Write the whole fixture tree under `root`; return the root."""
    rng = np.random.default_rng(seed)
    os.makedirs(root, exist_ok=True)
    for recording in RECORDINGS:
        base = os.path.join(root, recording["family"], recording["results_subject"], recording["results_condition"])
        os.makedirs(base, exist_ok=True)
        mat_path = write_phase1(base, rng, recording)
        sidecar = dict(recording)
        sidecar["phase1_schema_version"] = "2.3.0"
        sidecar["phase1_mat"] = os.path.relpath(mat_path, base)
        sidecar["written_utc"] = "2026-09-22T00:00:00+00:00"
        with open(os.path.join(base, "recording.json"), "w") as handle:
            json.dump(sidecar, handle, indent=2)
        units = phase2_units(rng, recording)
        write_phase2(base, rng, recording, units)
        write_phase3(base, rng, recording, units)
        surrogate_floors = pd.read_csv(os.path.join(base, "Phase3", "text", "surrogate_floor_by_d.csv"))
        write_phase4(base, rng, recording, units, surrogate_floors)
        write_phase5(base, rng, recording, units)
        write_phase6(base, recording)
    write_phase7(root, rng)
    return root


def main():
    """Build the tree under the output root given on the command line."""
    if len(sys.argv) < 2:
        print("usage: make_fixtures.py <output root>")
        return 2
    root = build(os.path.abspath(sys.argv[1]))
    print(f"fixture tree written -> {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
