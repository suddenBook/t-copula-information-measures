"""Cross-recording time scales: each recording's memory, and where it sits on Phase 7's axes.

Outputs, and the question each figure answers:

Phase6/text/timescale_table
    One row per recording, from its Phase 6 `timescale_summary.csv`.
    The columns give the sample rate, duration, segments and lag window, the signal-memory time tau_env, the amplitude-memory time tau_amp and its Gaussian reference tau_amp_ref with the amplitude excess tau_amp / tau_amp_ref, the integrated times and the integrated excess tau_amp_int / tau_amp_ref_int, the ratio tau_amp / tau_env, the samples per tau_env, the observation window T/tau_env, and the lag window in units of tau_env (`window_over_tau_env` = max_lag_s / tau_env_s).
    They go on with each fit's mode, floor flag and error, the Ornstein-Uhlenbeck-equivalent times with the shape memory per tau_env (`tau_amp_ou_over_tau_env` = tau_amp_ou_s / tau_env_s, whole-recording centring only), and the within-trial-centring twins (`<column>_trialc`) of the values the figures draw.
    The Markdown ends in a one-line summary of the ranges the simulation's grid has to cover; the CSV stays one row per recording, from which the ranges recompute.
Phase7/text/timescale_matched_window
    One row per recording and model nu: the model lag window closest to the recording's own lag window in units of tau_env.
    The window is given in tau (`matched_window_tau`) and in the model's own tau_env at that nu and window (`matched_window_tau_env` = window_tau / tau_env of `check_timescale.csv`, the window nearest the recording's `window_over_tau_env` on a log scale), with the model's time and excesses there.
    The model's tau_env falls below tau at heavy tails, so the match can change with nu.
    Beside them stand the shape memory per tau_env of the model at the matched window (`model_tau_amp_ou_over_tau_env` = tau_amp_ou / tau_env of that row) and of the recording (`recording_tau_amp_ou_over_tau_env`), each in its own tau_env, and their ratio `shape_memory_ratio` = recording / model.
    Above 1, the recording carries more shape-relevant memory per tau_env than the model at that nu, and the simulation is optimistic for its shape there; below 1, it is conservative.
Phase6/images/cross_f6_timescales
    How long does each recording remember its signal and its amplitude, and how much longer is its amplitude memory than a Gaussian signal with its autocorrelation would carry?
    tau_env and tau_amp are drawn in seconds on a log axis, one row per recording in its modality colour, with the Ornstein-Uhlenbeck-equivalent time as a tick beside each, and beside them the fitted and the integrated amplitude excess against their Gaussian value 1.
    The marker shape is the fit mode (a hexagon for the integrated excess, which has none), and a hollow marker is a value whose autocorrelation never fell below the floor, so a fitted time is extrapolated and an integrated one cut at the window.
    A trial recording also draws its within-trial-centred times and fitted excess as triangles below its row.
Phase7/images/cross_f7_recordings_on_time_axis
    What observation time and sampling rate does an accurate estimate need, and where do the recordings sit?
    The error of 1/nu_hat, its median and interquartile range over the replicates, is drawn against the window T/tau for a few representative truths (rows) and dimensions (columns) of Phase 7's recovery grid, one line per sampling rate.
    A vertical line stands at each recording's T/tau_env, whose key gives its samples per tau_env (marked when tau_env is extrapolated), and a triangle on the top edge marks a trial recording's within-trial-centred T/tau_env.
    The `_log` twin draws the size of the error, |median error| and the width of the interquartile range, on a log axis, as `phase6_acf_log` draws |ACF| (a signed band has no logarithm), so the error at a long window reads to an order of magnitude beside the short ones.
Phase7/images/cross_f7_amplitude_memory
    Does each recording carry more amplitude memory than the model would at its shape?
    The model's amplitude excess against nu, as the simulator check measured it with the Phase 6 estimator, is read at each nu over the model lag window matched to the recording at that nu (`timescale_matched_window`).
    Each profile of matched windows has one column, with each recording's excess as a horizontal line in the column of its own profile; the excess is 1 at Gaussian truth, for the model and for a Gaussian signal in any band.
    The upper row draws the model over every nu on a log axis, fitted (solid) and integrated (dashed); the rows below draw, on a linear axis around the recordings where the crossings are, the fitted excess and, when the tables carry it, the integrated one.
    The bottom row asks whether the simulation is optimistic or conservative for the recording's shape: it draws the shape memory per tau_env, the model's tau_amp_ou / tau_env at the matched windows (with the Ornstein-Uhlenbeck-equivalent time's tick) against each recording's `tau_amp_ou_over_tau_env`, on a linear axis around the recordings and the model's lowest value, when the check table carries tau_amp_ou.
    A recording whose autocorrelation never fell below the floor is marked in the key, and a trial recording's within-trial-centred fitted excess is a triangle on the right edge.

The recordings on the simulation's axes
    The simulation's grid of windows, sampling rates and lag windows is declared independently of the data and has to cover every recording.
    The Markdown summary gives the recordings' ranges of samples per tau_env, T/tau_env and lag window in tau_env to check that against.
    A recording whose tau_env fit failed contributes no value, and the line says how many recordings did.

Reading the amplitude-memory figure
    The simulation's time axis is tau_env, so what it says transfers to a recording only as far as the recording's amplitude memory resembles the model's.
    A recording whose excess lies above the model's excess at the recording's fitted nu (its median nu_hat, `cross_f1_shape_vs_dimension`), read over the model's lag window nearest its own in units of tau_env, carries more amplitude memory than the model, each relative to its own Gaussian reference, and on its own that makes the simulation optimistic for it; an excess below the model's means less amplitude memory.
    The model's units of tau_env are its own at each nu and window, which is why the matched window is read per nu.
    The fitted excess compares envelope decays; the integrated excess counts a slow component's size and duration together, which the fitted one does not, and is read the same way.
    Neither excess shows the band-pass factor, because each is normalised to its own Gaussian reference.
    At Gaussian truth the shape estimate's score lies in the fourth Wiener chaos of the normal scores, so, to first order, its autocorrelation is rho(k)^4, the square of the squared scores' rho_sq(k), and its integrated autocorrelation time is tau_amp_ou, the Ornstein-Uhlenbeck-equivalent time of the squared scores.
    That time is tau/2 for the model at Gaussian truth and about (3/16) tau_env for a Gaussian narrow-band signal, which thus has more shape-effective samples per tau_env than the model, a factor that works against extra amplitude memory.
    The net direction is read from the bottom row, tau_amp_ou / tau_env, each in its own tau_env (the model's at that nu and window): a recording above the model at its fitted nu has fewer shape-effective samples per tau_env than the model, and the simulation is optimistic for its shape; below it, conservative.
    tau_amp_ou sums over the whole lag window, so where the amplitude's autocorrelation never fell below the floor it is cut at the window, as the integrated time is.
    The model cannot oscillate and the band-passed recordings do, so the comparison is between one estimator's readings on one transform; the time scales are orders of magnitude, not parameters of the method.

Within-trial centring
    A recording of several trial segments (`n_segments` > 1) also carries its time scales with every segment centred on its own mean (`<column>_trialc`).
    The headline comparison stays the whole-recording centring, and the twin is drawn beside it, labelled, never in its place.
    Only its fitted values are drawn: centring each trial forces the within-trial autocovariances to sum to about zero, so no within-trial integrated time is defined (the phase writes NaN).
    No within-trial shape memory is drawn either: the within-trial Ornstein-Uhlenbeck-equivalent times are published but carry the within-trial centring's pull, several per cent of tau_amp_ou on sub-second trials, so `tau_amp_ou_over_tau_env` and the bottom row of `cross_f7_amplitude_memory` are the whole-recording centring's alone.
    A one-segment recording's twins equal its headline and are not drawn.

Recordings take their modality colour and, inside a modality, their own dash pattern (`dash_index`, as in every cross-recording figure).
The sampling rate takes the Purples ramp of `_style.col_rate` in the one figure that draws a line per rate, where the panel carries the dimension.
The zero line, the Gaussian values and every other reference are grey, and a key that stands for a glyph shape is near-black.
Figure text writes the shape as 1/nu, never p.
The Phase 7 figures need Phase 7's outputs under `<results root>/simulation/t_copula/all/Phase7` and are skipped with a printed line when they, or every recording's Phase 6, are absent.
"""

from __future__ import annotations

import os
import string
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _phase_io as IO  # noqa: E402
import _style as S  # noqa: E402
from render_cross_shape import dash_index  # noqa: E402
from render_phase7_shape_recovery import DIMENSIONS, ERROR_LABEL, RATE, TRUTHS, WINDOW, WINDOW_AXIS, key_below, nearest, nu_text, plain_log_axis, truth_title  # noqa: E402

plt = S.style()
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

# The Phase 7 study's base under the results root (`settings.rel_base` of Phase 7).
SIMULATION_REL_BASE = ("simulation", "t_copula", "all")

# The within-trial-centring twin of a Phase 6 column (`<column>_trialc`), drawn beside the headline for a recording of several trial segments.
TRIAL = "_trialc"
# The Phase 6 values the figures draw, whose twins the table carries beside them.
# A trial recording has no within-trial integrated time (NaN), so no integrated twin is drawn.
TRIAL_DRAWN = ("tau_env_s", "tau_amp_s", "tau_amp_ref_s", "amp_excess", "samples_per_tau", "T_over_tau", "env_floor_reached", "amp_floor_reached", "amp_ref_floor_reached", "tau_env_ou_s", "tau_amp_ou_s")
TABLE_COLUMNS = (
    "dataset_key",
    "modality",
    "fs",
    "duration_s",
    "n_segments",
    "max_lag_s",
    "tau_env_s",
    "tau_env_samples",
    "tau_amp_s",
    "tau_amp_ref_s",
    "amp_excess",
    "tau_amp_int_s",
    "tau_amp_ref_int_s",
    "amp_excess_int",
    "ratio_amp_env",
    "samples_per_tau",
    "T_over_tau",
    "env_fit_mode",
    "amp_fit_mode",
    "env_floor_reached",
    "amp_floor_reached",
    "amp_ref_floor_reached",
    "env_fit_error",
    "amp_fit_error",
    "amp_ref_fit_error",
    "tau_env_ou_s",
    "tau_amp_ou_s",
    "ratio_amp_env_ou",
) + tuple(column + TRIAL for column in TRIAL_DRAWN)
TEXT_COLUMNS = ("dataset_key", "modality", "env_fit_mode", "amp_fit_mode", "env_fit_error", "amp_fit_error", "amp_ref_fit_error")
# Derived here, after the summary's own columns; the shape memory per tau_env (tau_amp_ou_s / tau_env_s) is the whole-recording centring's alone.
DERIVED_COLUMNS = ("window_over_tau_env", "window_over_tau_env" + TRIAL, "tau_amp_ou_over_tau_env")
# One row per recording and model nu (`timescale_matched_window`).
MATCH_COLUMNS = (
    "dataset_key",
    "modality",
    "window_over_tau_env",
    "nu",
    "matched_window_tau",
    "matched_window_tau_env",
    "model_tau_env",
    "model_amp_excess",
    "model_amp_excess_int",
    "model_amp_floor_reached",
    "model_amp_ref_floor_reached",
    "model_tau_amp_ou_over_tau_env",
    "recording_tau_amp_ou_over_tau_env",
    "shape_memory_ratio",
)

# The fit mode is the marker shape, and a fit whose envelope never fell below the floor is hollow.
# A fit without a recorded mode, as in a summary without the mode columns, draws the square.
# The integrated excess is not a fit and takes a hexagon; a within-trial-centred twin is a triangle.
MODE_MARKER = {"peaks": "o", "monotone": "D"}
NO_MODE_MARKER = "s"
INTEGRATED_MARKER = "h"
MODE_LABEL = {"peaks": "oscillating fit (peaks)", "monotone": "monotone fit", "": "fit mode not recorded"}
GAUSSIAN_EXCESS = 1.0
REFERENCE_DASH = (0, (3, 2))
WIDTH_DASH = (0, (4, 2))
# The model's integrated excess beside its fitted one.
INTEGRATED_DASH = (0, (4, 1.6))
# A value whose autocorrelation never fell below the floor, marked in a key.
UNREACHED = "‡"

ROLES = (("env", "signal memory", r"$\tau_\mathrm{env}$"), ("amp", "amplitude memory", r"$\tau_\mathrm{amp}$"))
# (column, flag columns, panel title, axis label): the excess panels of cross_f6, each hollow when the amplitude's or its reference's autocorrelation never fell below the floor.
EXCESSES = (("amp_excess", ("amp_floor_reached", "amp_ref_floor_reached"), "amplitude excess", r"$\tau_\mathrm{amp}\,/\,\tau_\mathrm{amp,ref}$"), ("amp_excess_int", ("amp_floor_reached", "amp_ref_floor_reached"), "integrated excess", r"$\tau_\mathrm{amp,int}\,/\,\tau_\mathrm{amp,ref,int}$"))


# ---------------------------------------------------------------- reading
def _missing(value):
    return value is None or (isinstance(value, float) and not np.isfinite(value)) or str(value).strip().lower() in ("", "nan")


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def recording_rows(recordings):
    """[(recording, row)] for every recording with a Phase 6 summary, in discovery order.

    A row carries TABLE_COLUMNS, and a column the summary does not carry is left empty.
    The identity (dataset key, modality) is the recording's own `recording.json`, as in every cross-recording table, and falls back to the summary's.
    """
    rows = []
    for recording in recordings:
        summary = IO.read_table(IO.recording_phase(recording, 6), "timescale_summary.csv", quiet=True)
        if summary is None:
            continue
        first = summary.iloc[0]
        row = {}
        for column in TABLE_COLUMNS:
            value = first.get(column)
            if column in TEXT_COLUMNS:
                row[column] = "" if _missing(value) else str(value).strip()
            else:
                row[column] = _number(value)
        row["dataset_key"] = str(recording.get("dataset_key") or row["dataset_key"] or recording["rel_base"])
        row["modality"] = str(recording.get("modality") or row["modality"])
        rows.append((recording, row))
    found = {recording["rel_base"] for recording, _ in rows}
    missing = [r.get("dataset_key", r["rel_base"]) for r in recordings if r["rel_base"] not in found]
    if rows and missing:
        print(f"  [note] timescale: no Phase 6 timescale_summary.csv for {', '.join(missing)}")
    return rows


def _finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def model_windows(timescale):
    """The model's lag windows of `check_timescale.csv`, in tau and in its own tau_env.

    One row per table row with a usable window: `nu`, `window_tau`, the model's `tau_env` at that nu and window (in tau), `window_tau_env` = window_tau / tau_env, and `row`, the position of the table row it came from.
    nu = infinity is a nu; NaN is not.
    """
    frame = pd.DataFrame({"nu": IO.numeric(timescale["nu"]), "window_tau": IO.numeric(timescale["window_tau"]), "tau_env": IO.numeric(timescale["tau_env"]), "row": np.arange(len(timescale))})
    frame["window_tau_env"] = frame["window_tau"] / frame["tau_env"]
    usable = ~frame["nu"].isna() & np.isfinite(frame["window_tau_env"]) & (frame["window_tau"] > 0) & (frame["tau_env"] > 0)
    return frame[usable]


def model_shape_memory(timescale):
    """The model's shape memory per tau_env, tau_amp_ou / tau_env, per row of `check_timescale.csv` (both in tau); NaN where the table carries no tau_amp_ou or a row no positive tau_env."""
    if "tau_amp_ou" not in timescale.columns:
        return np.full(len(timescale), np.nan)
    tau_env = IO.numeric(timescale["tau_env"])
    return np.divide(IO.numeric(timescale["tau_amp_ou"]), tau_env, out=np.full(len(timescale), np.nan), where=tau_env > 0)


def matched_windows(window_over_tau_env, windows):
    """Per model nu, the lag window nearest a recording's own W/tau_env on a log scale, with both windows in units of tau_env: the recording's own, and the model's own at that nu and window.

    Returns a frame with `nu`, `matched_window_tau`, `matched_window_tau_env`, `model_tau_env` and `row` in ascending nu, empty when there is nothing to match.
    A tie goes to the shorter window.
    """
    columns = ["nu", "matched_window_tau", "matched_window_tau_env", "model_tau_env", "row"]
    if not (np.isfinite(window_over_tau_env) and window_over_tau_env > 0) or windows.empty:
        return pd.DataFrame(columns=columns)
    matches = []
    for nu, block in windows.sort_values(["nu", "window_tau"]).groupby("nu", sort=True):
        k = int(np.argmin(np.abs(np.log(block["window_tau_env"].to_numpy() / window_over_tau_env))))
        chosen = block.iloc[k]
        matches.append([float(nu), float(chosen["window_tau"]), float(chosen["window_tau_env"]), float(chosen["tau_env"]), int(chosen["row"])])
    return pd.DataFrame(matches, columns=columns)


def matched_table(rows, timescale, windows):
    """`timescale_matched_window` and {dataset key: that recording's matches} for the figure.

    The table holds every recording's per-nu match, with the model's values at the matched row, the shape memory per tau_env of the model there and of the recording, and their ratio (recording / model).
    """
    fitted = IO.numeric(timescale["amp_excess"]) if "amp_excess" in timescale.columns else np.full(len(timescale), np.nan)
    extra = {column: IO.numeric(timescale[source]) if source in timescale.columns else np.full(len(timescale), np.nan) for column, source in (("model_amp_excess_int", "amp_excess_int"), ("model_amp_floor_reached", "amp_floor_reached"), ("model_amp_ref_floor_reached", "amp_ref_floor_reached"))}
    shape = model_shape_memory(timescale)
    records, by_recording = [], {}
    for _, row in rows:
        matches = matched_windows(row["window_over_tau_env"], windows)
        by_recording[row["dataset_key"]] = matches
        recording = row["tau_amp_ou_over_tau_env"]
        for match in matches.itertuples(index=False):
            model = shape[match.row]
            records.append(
                {
                    "dataset_key": row["dataset_key"],
                    "modality": row["modality"],
                    "window_over_tau_env": row["window_over_tau_env"],
                    "nu": match.nu,
                    "matched_window_tau": match.matched_window_tau,
                    "matched_window_tau_env": match.matched_window_tau_env,
                    "model_tau_env": match.model_tau_env,
                    "model_amp_excess": fitted[match.row],
                    **{column: values[match.row] for column, values in extra.items()},
                    "model_tau_amp_ou_over_tau_env": model,
                    "recording_tau_amp_ou_over_tau_env": recording,
                    "shape_memory_ratio": recording / model if model > 0 else np.nan,
                }
            )
    return pd.DataFrame(records, columns=list(MATCH_COLUMNS)), by_recording


def _trial(row):
    """True for a recording of several trial segments, whose within-trial twins differ from its headline; a one-segment recording's twins equal it and are not drawn."""
    return np.isfinite(row["n_segments"]) and row["n_segments"] > 1


def recording_ranges(table):
    """The recordings' ranges on the simulation's axes, over the finite rows."""
    samples = _finite(table["samples_per_tau"])
    windows = _finite(table["T_over_tau"])
    lag = _finite(table["window_over_tau_env"])

    def stat(fn, values):
        return float(fn(values)) if values.size else np.nan

    return {
        "n_recordings": int(len(table)),
        "n_with_tau_env": int(samples.size),
        "samples_per_tau_min": stat(np.min, samples),
        "samples_per_tau_median": stat(np.median, samples),
        "samples_per_tau_max": stat(np.max, samples),
        "T_over_tau_min": stat(np.min, windows),
        "T_over_tau_max": stat(np.max, windows),
        "lag_window_min": stat(np.min, lag),
        "lag_window_median": stat(np.median, lag),
        "lag_window_max": stat(np.max, lag),
    }


def recording_ranges_line(ranges):
    """The one-line Markdown summary of `recording_ranges`."""
    c = ranges
    return (
        f"**The recordings on the simulation's axes** ({c['n_with_tau_env']} of "
        f"{c['n_recordings']} recordings with a tau_env): samples per tau_env "
        f"{c['samples_per_tau_min']:.3g} to {c['samples_per_tau_max']:.3g} (median "
        f"{c['samples_per_tau_median']:.3g}); T/tau_env {c['T_over_tau_min']:,.0f} to "
        f"{c['T_over_tau_max']:,.0f}; lag window {c['lag_window_min']:.3g} to "
        f"{c['lag_window_max']:.3g} tau_env (median {c['lag_window_median']:.3g}), each "
        f"read, per model nu, against the model's lag window nearest it on a log scale "
        f"in the model's own tau_env (Phase7/text/timescale_matched_window).  The "
        f"simulation's grid is declared independently of the data and has to cover these "
        f"ranges."
    )


def simulation_phase(recordings):
    """Phase 7's directory under the results root the recordings were discovered in."""
    recording = recordings[0]
    depth = len([p for p in recording["rel_base"].split("/") if p])
    root = os.path.normpath(os.path.join(recording["base"], *[os.pardir] * depth))
    return os.path.join(root, *SIMULATION_REL_BASE, "Phase7")


# ---------------------------------------------------------------- shared glyphs
def _mode(value):
    mode = str(value).strip().lower()
    return mode if mode in MODE_MARKER else ""


def _decade_text(value, style):
    if style == "plain" and 1e-3 <= value < 1e4:
        return f"{value:g}"
    return rf"$10^{{{int(round(np.log10(value)))}}}$"


def _decades(ax, axis, values, pad=1.0, snap=True, style="plain"):
    """A log axis with a major tick on every decade inside the drawn range.

    `snap` widens the range to whole decades; otherwise it is the data range times `pad`, widened to whole decades only when that would leave fewer than two decade ticks.
    """
    v = _finite(values)
    v = v[v > 0]
    if v.size == 0:
        return
    lo, hi = float(v.min()) / pad, float(v.max()) * pad
    if snap or np.floor(np.log10(hi)) - np.ceil(np.log10(lo)) < 1:
        lo, hi = 10.0 ** np.floor(np.log10(lo)), 10.0 ** np.ceil(np.log10(hi))
        hi = max(hi, lo * 10.0)
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    (ax.set_xlim if axis == "x" else ax.set_ylim)(lo, hi)
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(FixedLocator(10.0 ** np.arange(np.ceil(np.log10(lo) - 1e-9), np.floor(np.log10(hi) + 1e-9) + 1)))
    target.set_major_formatter(FuncFormatter(lambda value, _: _decade_text(value, style)))
    target.set_minor_formatter(NullFormatter())


def _fit_marker(ax, x, y, colour, mode, reached, size=5.0, zorder=4):
    ax.plot([x], [y], ls="none", marker=MODE_MARKER.get(mode, NO_MODE_MARKER), ms=size, mec=colour, mfc=colour if reached else "white", mew=1.0, zorder=zorder)


def _fit_keys(modes, hollow, what, hollow_label="envelope never fell below the floor: extrapolated"):
    """Near-black keys for the marker shapes and the hollow face that occur.

    The hollow key takes the one shape drawn when there is one, the circle otherwise.
    """
    keys = [Line2D([], [], ls="none", marker=MODE_MARKER.get(mode, NO_MODE_MARKER), ms=4.5, color=S.KEY, label=f"{what}, {MODE_LABEL[mode]}") for mode in ("peaks", "monotone", "") if mode in modes]
    if hollow:
        shape = MODE_MARKER.get(next(iter(modes)), NO_MODE_MARKER) if len(modes) == 1 else "o"
        keys.append(Line2D([], [], ls="none", marker=shape, ms=4.5, mec=S.KEY, mfc="white", mew=1.0, label=hollow_label))
    return keys


def _modality_keys(rows):
    present = {row["modality"] for _, row in rows}
    return [Line2D([], [], color=S.modality_color(m), lw=1.4, label=m) for m in list(S.MODALITY_ORDER) + sorted(present - set(S.MODALITY_ORDER)) if m in present]


def _recording_name(row):
    return S.recording_label(row["dataset_key"])


def _placement(row, suffix=""):
    """Where a recording sits on the simulation's axes, T/tau_env and its samples per tau_env, from its headline columns or from their twins (`suffix`)."""
    return r"$T/\tau_\mathrm{env}$ = " + f"{row['T_over_tau' + suffix]:,.0f}".replace(",", " ") + f", {row['samples_per_tau' + suffix]:.0f} per " + r"$\tau_\mathrm{env}$"


def _reached(value):
    """A floor flag read as reached unless it is written 0; NaN (no fit) never draws."""
    return not (np.isfinite(value) and value == 0)


# A within-trial-centred twin sits this far below its recording's row.
TWIN_OFFSET = 0.3


def _twin(ax, row, column, flags, y, colour):
    """A trial recording's within-trial-centred value as a triangle just below its row, hollow on the twin's own floor flags; returns the value drawn, or NaN."""
    value = row.get(column + TRIAL, np.nan)
    if not (_trial(row) and np.isfinite(value) and value > 0):
        return np.nan
    reached = all(_reached(row.get(flag + TRIAL, np.nan)) for flag in flags)
    ax.plot([value], [y + TWIN_OFFSET], ls="none", marker="v", ms=4.0, mec=colour, mfc=colour if reached else "white", mew=0.9, zorder=4)
    return float(value)


# ---------------------------------------------------------------- (b) the time scales
def _timescales(rows, label):
    """The cross_f6_timescales figure: tau_env, tau_amp and the amplitude excesses of every recording, one row each."""
    values = np.concatenate([[row[f"tau_{role}_s"], row[f"tau_{role}_ou_s"], row[f"tau_{role}_s{TRIAL}"] if _trial(row) else np.nan] for _, row in rows for role, _, _ in ROLES])
    if not (_finite(values) > 0).any():
        print("  [skip] cross_f6_timescales: no recording carries a finite tau_env or tau_amp")
        return None
    # The integrated excess stands beside the fitted one when any recording carries it.
    excesses = [entry for entry in EXCESSES if any(np.isfinite(row[entry[0]]) for _, row in rows)] or list(EXCESSES[:1])
    count = len(rows)
    fig, axes = plt.subplots(1, len(ROLES) + len(excesses), figsize=(S.WIDTH, max(1.7, 0.95 + 0.22 * count)), layout="constrained", sharey=True, gridspec_kw={"width_ratios": [1.0] * len(ROLES) + [0.75] * len(excesses)})
    for ax in axes[1 : len(ROLES)]:
        ax.sharex(axes[0])
    modes, hollow, twins = set(), False, False
    for index, (ax, (role, title, symbol)) in enumerate(zip(axes, ROLES)):
        for y, (_, row) in enumerate(rows):
            colour = S.modality_color(row["modality"])
            ax.axhline(y, color=S.GREY, lw=0.4, alpha=0.35, zorder=0)
            tau, ou = row[f"tau_{role}_s"], row[f"tau_{role}_ou_s"]
            if np.isfinite(ou) and ou > 0:
                if np.isfinite(tau) and tau > 0:
                    ax.plot([ou, tau], [y, y], color=colour, lw=0.9, alpha=0.55, zorder=2)
                ax.plot([ou], [y], ls="none", marker="|", ms=7, mew=1.3, color=colour, zorder=3)
            if not (np.isfinite(tau) and tau > 0):
                ax.text(0.02, y, "no fit", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.5, color=S.GREY)
                continue
            mode = _mode(row[f"{role}_fit_mode"])
            reached = row[f"{role}_floor_reached"] != 0
            modes.add(mode)
            hollow |= not reached
            _fit_marker(ax, tau, y, colour, mode, reached)
            twins |= np.isfinite(_twin(ax, row, f"tau_{role}_s", (f"{role}_floor_reached",), y, colour))
        _decades(ax, "x", values)
        ax.set_xlabel("time scale (s)", labelpad=2)
        S.panel_label(ax, string.ascii_uppercase[index], f"{title} {symbol}")

    # Each excess against its Gaussian value, hollow when the amplitude's or its reference's autocorrelation never fell below the floor.
    # The fitted excess takes the marker of the amplitude fit's mode; the integrated one is no fit and takes the hexagon.
    for offset, (column, flags, title, axis_label) in enumerate(excesses):
        ax = axes[len(ROLES) + offset]
        drawn = [GAUSSIAN_EXCESS]
        for y, (_, row) in enumerate(rows):
            ax.axhline(y, color=S.GREY, lw=0.4, alpha=0.35, zorder=0)
            excess, colour = row[column], S.modality_color(row["modality"])
            if not (np.isfinite(excess) and excess > 0):
                ax.text(0.02, y, "no excess", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.5, color=S.GREY)
                continue
            reached = all(_reached(row[flag]) for flag in flags)
            hollow |= not reached
            if column == "amp_excess":
                mode = _mode(row["amp_fit_mode"])
                modes.add(mode)
                _fit_marker(ax, excess, y, colour, mode, reached)
            else:
                ax.plot([excess], [y], ls="none", marker=INTEGRATED_MARKER, ms=5.0, mec=colour, mfc=colour if reached else "white", mew=1.0, zorder=4)
            drawn.append(excess)
            # Only the fitted excess has a within-trial twin, because no integrated time is defined under within-trial centring.
            twin = _twin(ax, row, column, flags, y, colour) if column == "amp_excess" else np.nan
            if np.isfinite(twin):
                twins = True
                drawn.append(twin)
        ax.axvline(GAUSSIAN_EXCESS, color=S.GREY, lw=0.9, ls=REFERENCE_DASH, zorder=1)
        low, high = min(drawn), max(drawn)
        margin = max(0.12 * (high - low), 0.05)
        ax.set_xlim(max(0.0, low - margin), high + margin)
        ax.set_xlabel(axis_label, labelpad=2)
        S.panel_label(ax, string.ascii_uppercase[len(ROLES) + offset], title)

    axes[0].set_yticks(range(count))
    axes[0].set_yticklabels([_recording_name(row) for _, row in rows], fontsize=7)
    for tick, (_, row) in zip(axes[0].get_yticklabels(), rows):
        tick.set_color(S.modality_color(row["modality"]))
    axes[0].set_ylim(count - 0.5, -0.5)  # the first recording on top
    axes[0].tick_params(axis="y", length=0)
    integrated = len(excesses) > 1
    handles = _modality_keys(rows) + _fit_keys(modes, hollow, "envelope time", hollow_label=("autocorrelation never fell below the floor: a fitted value extrapolated, an integrated one cut at the window") if integrated else "envelope never fell below the floor: extrapolated")
    if integrated:
        handles.append(Line2D([], [], ls="none", marker=INTEGRATED_MARKER, ms=4.5, color=S.KEY, label="integrated excess (not a fit)"))
    if twins:
        handles.append(Line2D([], [], ls="none", marker="v", ms=4.0, color=S.KEY, label="within-trial centring, a trial recording"))
    handles += [Line2D([], [], ls="none", marker="|", ms=7, mew=1.3, color=S.KEY, label="Ornstein-Uhlenbeck-equivalent time"), Line2D([], [], color=S.GREY, lw=0.9, ls=REFERENCE_DASH, label="excess of a Gaussian signal, 1")]
    key_below(fig, handles, ncols=3)
    S.source_label(fig, label)
    return fig


# ---------------------------------------------------------------- (c) the time axis
def _placed(rows, recordings, column):
    """(row, colour, dash) for every recording with a finite value in `column`."""
    dashes = dash_index(recordings)
    placed = [(row, S.modality_color(row["modality"]), S.recording_linestyle(dashes[recording["rel_base"]])) for recording, row in rows if np.isfinite(row[column])]
    unplaced = [row["dataset_key"] for _, row in rows if not np.isfinite(row[column])]
    if unplaced:
        print(f"  [note] timescale: no {column} for {', '.join(unplaced)}; not drawn")
    return placed


def _on_time_axis(log, summary, placed, label):
    """The error of 1/nu_hat against T/tau, one line per rate, signed with its interquartile band.

    The `_log` twin draws its size instead, |median error| and the interquartile width, on a log axis, as `phase6_acf_log` draws |ACF|, because a signed band has no logarithm.
    A recording's line stands at its T/tau_env, and a trial recording's within-trial-centred T/tau_env is a triangle on each panel's top edge.
    """
    p_all = IO.numeric(summary["p_true"])
    d_all = IO.numeric(summary["d"])
    truths = nearest(TRUTHS, sorted(set(_finite(p_all))), log=True)
    dims = nearest(DIMENSIONS, sorted(set(_finite(d_all))), log=False)
    if not truths or not dims:
        print("  [skip] cross_f7_recordings_on_time_axis: recovery_summary.csv carries no finite p_true or d")
        return None
    rate_all = IO.numeric(summary[RATE]) if RATE in summary.columns else np.full(len(summary), np.nan)
    rates = sorted(set(_finite(rate_all))) or [np.nan]
    windows = _finite(IO.numeric(summary[WINDOW]))
    band = IO.has(summary, "q25", "q75")
    fig, axes = plt.subplots(len(truths), len(dims), figsize=(S.WIDTH, 1.5 * len(truths) + 0.45), layout="constrained", sharex=True, sharey="row", squeeze=False)
    for r, p in enumerate(truths):
        drawn = [] if log else [0.0]
        for c, d in enumerate(dims):
            ax = axes[r][c]
            for rate in rates:
                at = (p_all == p) & (d_all == d)
                if np.isfinite(rate):
                    at &= rate_all == rate
                block = summary[at].sort_values(WINDOW)
                colour = S.col_rate(rate) if np.isfinite(rate) else S.KEY
                x = IO.numeric(block[WINDOW])
                median = IO.numeric(block["median_bias"])
                if band:
                    low, high = IO.numeric(block["q25"]) - p, IO.numeric(block["q75"]) - p
                if log:
                    series = [(np.abs(median), "-", "o")] + ([(high - low, WIDTH_DASH, "")] if band else [])
                    for values, style, marker in series:
                        ax.plot(x, values, color=colour, lw=1.1, ls=style, marker=marker, ms=2.4, mfc="white", mew=0.8, zorder=4)
                        drawn += list(values)
                else:
                    if band:
                        ax.fill_between(x, low, high, color=colour, alpha=0.18, lw=0, zorder=1)
                        drawn += [*low, *high]
                    ax.plot(x, median, color=colour, lw=1.1, marker="o", ms=2.4, mfc="white", mew=0.8, zorder=4)
                    drawn += list(median)
            if not log:
                ax.axhline(0.0, color=S.GREY, lw=0.9, zorder=0)
            for row, colour, dashes in placed:
                ax.axvline(row["T_over_tau"], color=colour, ls=dashes, lw=1.1, alpha=0.9, zorder=3)
                twin = row["T_over_tau" + TRIAL]
                if _trial(row) and np.isfinite(twin) and twin > 0:
                    reached = _reached(row["env_floor_reached" + TRIAL])
                    ax.plot([twin], [1.0], transform=ax.get_xaxis_transform(), ls="none", marker="v", ms=4.0, mec=colour, mfc=colour if reached else "white", mew=0.9, clip_on=False, zorder=5)
            S.panel_label(ax, string.ascii_uppercase[r * len(dims) + c], truth_title(p, d))
            if r == len(truths) - 1:
                ax.set_xlabel(WINDOW_AXIS, labelpad=2)
        axes[r][0].set_ylabel("size of the error" if log else ERROR_LABEL, labelpad=2)
        if log and (_finite(drawn) > 0).all():
            _decades(axes[r][0], "y", drawn, pad=1.3, snap=False, style="power")
        elif log:
            # An exact zero (a median on the truth, an interquartile range of width zero) pins to the bottom of a symmetric-log axis instead of being dropped.
            S.log_scale(axes[r][0], "y", np.asarray(drawn, dtype=float))
    marks = [row["T_over_tau"] for row, _, _ in placed] + [row["T_over_tau" + TRIAL] for row, _, _ in placed if _trial(row)]
    _decades(axes[0][0], "x", np.concatenate([windows, marks]), pad=1.6, snap=False, style="power")

    handles = [Line2D([], [], color=S.col_rate(rate), lw=1.4, label=f"{rate:g} samples per " + r"$\tau$") for rate in rates if np.isfinite(rate)]
    if log:
        handles += [Line2D([], [], color=S.KEY, lw=1.1, marker="o", ms=2.4, mfc="white", mew=0.8, label=r"$|$median error$|$, in the rate's colour"), Line2D([], [], color=S.KEY, lw=1.1, ls=WIDTH_DASH, label="width of the interquartile range")]
    else:
        handles += [Line2D([], [], color=S.KEY, lw=1.1, marker="o", ms=2.4, mfc="white", mew=0.8, label="median error, in the rate's colour"), Patch(facecolor=S.GREY, alpha=0.28, label="interquartile range over replicates"), Line2D([], [], color=S.GREY, lw=0.9, label="no error")]
    beyond, extrapolated = False, False
    for row, colour, dashes in placed:
        outside = windows.size and not (windows.min() <= row["T_over_tau"] <= windows.max())
        beyond |= bool(outside)
        unreached = not _reached(row["env_floor_reached"])
        extrapolated |= unreached
        handles.append(Line2D([], [], color=colour, ls=dashes, lw=1.1, label=f"{_recording_name(row)}: {_placement(row)}" + (" †" if outside else "") + (f" {UNREACHED}" if unreached else "")))
        twin = row["T_over_tau" + TRIAL]
        if _trial(row) and np.isfinite(twin) and twin > 0:
            reached = _reached(row["env_floor_reached" + TRIAL])
            extrapolated |= not reached
            handles.append(Line2D([], [], ls="none", marker="v", ms=4.0, mec=colour, mfc=colour if reached else "white", mew=0.9, label=f"{_recording_name(row)}, within-trial centring: " + _placement(row, TRIAL) + ("" if reached else f" {UNREACHED}")))
    notes = (
        (["† beyond the simulated windows"] if beyond else [])
        + (
            [
                f"{UNREACHED} " + r"$\tau_\mathrm{env}$ extrapolated: its envelope never fell "
                "below the floor"
            ]
            if extrapolated
            else []
        )
        + _recovery_notes(summary, placed)
    )
    handles += [Line2D([], [], color="none", label=note) for note in notes]
    key_below(fig, handles, ncols=2)
    S.source_label(fig, label)
    return fig


def _recovery_notes(summary, placed):
    """What the simulated cells are and how the recordings are sampled, as key notes."""
    reps = _finite(IO.numeric(IO.pick(summary, "reps", default=pd.Series([np.nan]))))
    run = "Phase 7, nested rates: every rate reads the same paths"
    if reps.size:
        run += f"; {int(reps.max())} replicates per cell"
    notes = [run, "numerically successful replicates only"]
    spread = _finite([row["samples_per_tau"] for row, _, _ in placed])
    if spread.size:
        notes.append(
            f"recordings: {spread.min():.3g} to {spread.max():.3g} samples per "
            r"$\tau_\mathrm{env}$"
        )
    return notes


# ---------------------------------------------------------------- (d) amplitude memory
# (key, column, row title, axis label): each excess of the figure, fitted and integrated, read under one name on the model's table and on the recordings'.
READINGS = (("fitted", "amp_excess", "fitted", "fitted excess\n" + r"$\tau_\mathrm{amp}\,/\,\tau_\mathrm{amp,ref}$"), ("integrated", "amp_excess_int", "integrated", "integrated excess\n" + r"$\tau_\mathrm{amp,int}\,/\,\tau_\mathrm{amp,ref,int}$"))
# The bottom row, in the same form: the shape memory per tau_env of the model (`model_shape_memory`) and of the recordings, which keeps the band-pass factor that each excess, normalised to its own Gaussian reference, cannot show.
SHAPE_MEMORY = ("shape", "tau_amp_ou_over_tau_env", "shape memory", r"shape memory per $\tau_\mathrm{env}$" + "\n" + r"$\tau_\mathrm{amp,OU}\,/\,\tau_\mathrm{env}$")


def _letter(index):
    """Panel letter `index`: A to Z, then AA, AB, ... for a figure of many columns."""
    letters = string.ascii_uppercase
    return letters[index] if index < 26 else letters[index // 26 - 1] + letters[index % 26]


def _profile_title(profile):
    """A column's matched model windows over nu, as (title, note).

    The window at the largest nu titles the column, and every other window, with the nu it is matched at, is a note inside the panel, so a title stays short enough for many columns.
    """
    main = profile[-1][1]
    others = {}
    for nu, window in profile:
        if window != main:
            others.setdefault(window, []).append(nu_text(nu))
    note = "\n".join(rf"{window:g} $\tau$ at $\nu$ = " + ", ".join(nus) for window, nus in others.items())
    return rf"window {main:g} $\tau$", note


def _model_curve(ax, x, rows, values, key, reached, modes_of, modes):
    """Draw the model's excess at the matched rows and return whether any marker was hollow.

    The fitted excess is solid with its fit-mode markers and the integrated one dashed with hexagons, each hollow where the floor was not reached.
    """
    ax.plot(x, values[rows], color=S.KEY, lw=0.9, ls="-" if key == "fitted" else INTEGRATED_DASH, zorder=3)
    for xk, k in zip(x, rows):
        if key == "fitted":
            mode = _mode(modes_of[k])
            modes.add(mode)
            _fit_marker(ax, xk, values[k], S.KEY, mode, bool(reached[k]), size=4.0)
        else:
            ax.plot([xk], [values[k]], ls="none", marker=INTEGRATED_MARKER, ms=4.0, mec=S.KEY, mfc=S.KEY if reached[k] else "white", mew=1.0, zorder=4)
    return bool((~reached[rows]).any())


def _amplitude_memory(timescale, placed, matches, label):
    """The model's excess against nu, read at each nu over the lag window matched to the recording there, beside the recordings' own, with one column per profile of matched windows.

    The upper row draws the model over every nu on a log axis.
    The rows below it draw each excess on a linear axis around the recordings and the Gaussian value, where the crossings the figure is read at sit (the model runs out of that range at heavy tails).
    The bottom row, when the model's table carries tau_amp_ou, draws the shape memory per tau_env the same way, around the recordings and the model's lowest value: the net direction, which the excesses, each normalised to its own Gaussian reference, cannot give.
    It is the whole-recording centring's alone.
    """
    nu = IO.numeric(timescale["nu"])
    order = np.sort(np.unique(nu[~np.isnan(nu)]))
    position = {float(v): k for k, v in enumerate(order)}
    reached = np.ones(len(timescale), dtype=bool)
    for flag in ("amp_floor_reached", "amp_ref_floor_reached"):
        if flag in timescale.columns:
            reached &= IO.flag(timescale[flag])
    modes_of = IO.text(timescale["reading_mode"]) if "reading_mode" in timescale.columns else np.full(len(timescale), "", dtype=object)
    readings = [entry for entry in READINGS if entry[1] in timescale.columns]
    model = {key: IO.numeric(timescale[column]) for key, column, _, _ in readings}
    model["shape"] = model_shape_memory(timescale)
    shape = bool(np.isfinite(model["shape"]).any())
    lower = readings + ([SHAPE_MEMORY] if shape else [])

    # Recordings with one profile of matched windows over nu share a column; a recording without a tau_env has no match and is not drawn.
    columns, unmatched, drawn_recordings = {}, [], []
    for row, colour, dashes in placed:
        match = matches.get(row["dataset_key"])
        if match is None or match.empty:
            unmatched.append(row["dataset_key"])
            continue
        profile = tuple(zip(match["nu"].astype(float), match["matched_window_tau"].astype(float)))
        entry = columns.setdefault(profile, {"rows": match["row"].to_numpy(dtype=int), "members": []})
        entry["members"].append((row, colour, dashes))
        drawn_recordings.append((row, colour, dashes))
    if unmatched:
        print(f"  [note] cross_f7_amplitude_memory: no matched model window for {', '.join(unmatched)}; not drawn")
    if not columns:
        print("  [skip] cross_f7_amplitude_memory: no recording has a matched model window")
        return None
    profiles = sorted(columns, key=lambda profile: (profile[-1][1], profile))
    if not shape:
        print("  [note] cross_f7_amplitude_memory: check_timescale.csv carries no finite tau_amp_ou; the shape-memory row is not drawn")
    else:
        missing = [row["dataset_key"] for row, _, _ in drawn_recordings if not np.isfinite(row["tau_amp_ou_over_tau_env"])]
        if missing:
            print(f"  [note] cross_f7_amplitude_memory: no tau_amp_ou_over_tau_env for {', '.join(missing)}; not drawn in the shape-memory row")

    fig, axes = plt.subplots(1 + len(lower), len(profiles), figsize=(S.WIDTH, 1.45 * (1 + len(lower)) + 0.3), layout="constrained", sharex=True, squeeze=False)
    modes, hollow, unreached = set(), False, False
    for c, profile in enumerate(profiles):
        rows, members = columns[profile]["rows"], columns[profile]["members"]
        x = np.array([position[float(nu[k])] for k in rows])
        ax = axes[0, c]
        drawn = [GAUSSIAN_EXCESS]
        for key, _, _, _ in readings:
            hollow |= _model_curve(ax, x, rows, model[key], key, reached, modes_of, modes)
            drawn += list(model[key][rows])
        ax.axhline(GAUSSIAN_EXCESS, color=S.GREY, lw=0.9, ls=REFERENCE_DASH, zorder=0)
        drawn = _finite(drawn)
        plain_log_axis(ax, "y", drawn[drawn > 0])
        title, note = _profile_title(profile)
        S.panel_label(ax, _letter(c), title)
        if note:
            # The upper right is clear: the model's excess falls towards the Gaussian end.
            ax.text(0.97, 0.95, note, transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=S.KEY)
        for r, (key, column, title, _) in enumerate(lower, start=1):
            ax = axes[r, c]
            if key == "shape":
                # The OU-equivalent time takes its tick marker from cross_f6_timescales.
                # No value is common to the model and a Gaussian recording here (1/2 for the model at Gaussian truth, about 3/16 for a narrow band), so there is no reference line, and the range takes in the model's lowest value instead.
                ax.plot(x, model[key][rows], color=S.KEY, lw=0.9, marker="|", ms=7, mew=1.3, zorder=3)
                lowest = _finite(model[key][rows])
                recorded = [float(lowest.min())] if lowest.size else []
            else:
                hollow |= _model_curve(ax, x, rows, model[key], key, reached, modes_of, modes)
                ax.axhline(GAUSSIAN_EXCESS, color=S.GREY, lw=0.9, ls=REFERENCE_DASH, zorder=0)
                recorded = [GAUSSIAN_EXCESS]
            for row, colour, dashes in members:
                value = row[column]
                if np.isfinite(value) and value > 0:
                    ax.axhline(value, color=colour, ls=dashes, lw=1.2, alpha=0.9, zorder=2)
                    recorded.append(value)
                # The fitted excess alone has a within-trial twin, because no integrated time is defined under within-trial centring.
                twin = row[column + TRIAL] if key == "fitted" else np.nan
                if _trial(row) and np.isfinite(twin) and twin > 0:
                    # The within-trial-centred excess, on the right edge beside the line.
                    ok = _reached(row["amp_floor_reached" + TRIAL]) and _reached(row["amp_ref_floor_reached" + TRIAL])
                    ax.plot([order.size - 0.35], [twin], ls="none", marker="<", ms=4.5, mec=colour, mfc=colour if ok else "white", mew=0.9, zorder=5)
                    recorded.append(twin)
            if recorded:
                low, high = min(recorded), max(recorded)
                margin = max(0.35 * (high - low), 0.10 * high)
                ax.set_ylim(max(0.0, low - margin), high + margin)
            S.panel_label(ax, _letter(r * len(profiles) + c), title)
    tick_size = 7 if len(profiles) <= 4 else 6
    for ax in axes.ravel():
        ax.set_xticks(range(order.size))
        ax.set_xticklabels([nu_text(v) for v in order], fontsize=tick_size)
        ax.set_xlim(-0.5, order.size - 0.1)
    for ax in axes[-1]:
        ax.set_xlabel(r"model $\nu$", labelpad=2)
    axes[0, 0].set_ylabel("model's excess\n(log axis)", labelpad=2)
    for r, (_, _, _, axis_label) in enumerate(lower, start=1):
        axes[r, 0].set_ylabel(axis_label, labelpad=2)

    handles = [Line2D([], [], color=S.KEY, lw=0.9, label="model, fitted excess")]
    if len(readings) > 1:
        handles.append(Line2D([], [], color=S.KEY, lw=0.9, ls=INTEGRATED_DASH, marker=INTEGRATED_MARKER, ms=4.0, label="model, integrated excess"))
    if shape:
        handles.append(Line2D([], [], color=S.KEY, lw=0.9, marker="|", ms=7, mew=1.3, label=r"model, shape memory per $\tau_\mathrm{env}$"))
    handles += _fit_keys(modes, hollow, "model", hollow_label="a model autocorrelation never fell below the floor")
    handles.append(Line2D([], [], color=S.GREY, lw=0.9, ls=REFERENCE_DASH, label="excess at Gaussian truth, 1"))
    for row, colour, dashes in drawn_recordings:
        flag = _reached(row["amp_floor_reached"]) and _reached(row["amp_ref_floor_reached"])
        unreached |= not flag
        # The key carries the headline's fitted excess; the integrated one is read off its row and written in `timescale_table`, which keeps the key inside the page.
        handles.append(Line2D([], [], color=colour, ls=dashes, lw=1.2, label=f"{_recording_name(row)}: {row['amp_excess']:.2f}, lags to {row['window_over_tau_env']:.3g} " + r"$\tau_\mathrm{env}$" + ("" if flag else f" {UNREACHED}")))
        if _trial(row) and np.isfinite(row["amp_excess" + TRIAL]):
            ok = _reached(row["amp_floor_reached" + TRIAL]) and _reached(row["amp_ref_floor_reached" + TRIAL])
            unreached |= not ok
            handles.append(Line2D([], [], ls="none", marker="<", ms=4.5, mec=colour, mfc=colour if ok else "white", mew=0.9, label=f"{_recording_name(row)}, within-trial centring: {row['amp_excess' + TRIAL]:.2f}" + ("" if ok else f" {UNREACHED}")))
    cut = (["integrated"] if len(readings) > 1 else []) + ([r"$\tau_\mathrm{amp,OU}$"] if shape else [])
    flagged = f"{UNREACHED} never fell below the floor: fitted excess extrapolated" + (f", {' and '.join(cut)} cut at the window" if cut else "")
    notes = ([flagged] if unreached else []) + [
        r"model window matched per $\nu$, nearest the recording's own in the model's "
        r"$\tau_\mathrm{env}$",
        r"excess rows, above the model at the recording's fitted $\nu$: more amplitude "
        "memory",
        "than the model, which on its own makes the simulation optimistic",
    ]
    if shape:
        notes += [
            "shape-memory row, the net direction, with the band-pass factor the excess rows cannot show:",
            r"above the model at the fitted $\nu$, simulation optimistic for the "
            "shape; below, conservative",
        ]
    notes += _model_notes(timescale)
    handles += [Line2D([], [], color="none", label=note) for note in notes]
    key_below(fig, handles, ncols=2)
    S.source_label(fig, label)
    return fig


def _model_notes(timescale):
    """The model's measurement settings."""
    parts = []
    for column, template in (("D", "D = {:g}"), ("T_over_tau", r"$T/\tau$ = {:g}"), ("n_paths", "{:g} paths per " + r"$\nu$")):
        values = _finite(IO.numeric(timescale[column])) if column in timescale.columns else np.array([])
        if values.size:
            parts.append(template.format(float(np.median(values))))
    return ["model: " + ", ".join(parts)] if parts else []


# ---------------------------------------------------------------- driver entry
def render(recordings, output_base, label="cross-recording"):
    """Write the cross-recording time-scale tables and figures under `output_base` and return the paths written."""
    rows = recording_rows(recordings)
    if not rows:
        print("  [skip] timescale_table, cross_f6_timescales, cross_f7_*: no recording carries a Phase 6 timescale_summary.csv")
        return []
    phase7 = simulation_phase(recordings)
    timescale = IO.read_table(phase7, "check_timescale.csv", quiet=True)
    for _, row in rows:
        for suffix in ("", TRIAL):
            tau = row["tau_env_s" + suffix]
            row["window_over_tau_env" + suffix] = row["max_lag_s"] / tau if np.isfinite(tau) and tau > 0 else np.nan
        tau = row["tau_env_s"]
        row["tau_amp_ou_over_tau_env"] = row["tau_amp_ou_s"] / tau if np.isfinite(tau) and tau > 0 else np.nan

    written = []
    table = pd.DataFrame([row for _, row in rows], columns=list(TABLE_COLUMNS) + list(DERIVED_COLUMNS))
    text_dir = os.path.join(output_base, "Phase6", "text")
    written += IO.write_table(table, text_dir, "timescale_table")
    line = recording_ranges_line(recording_ranges(table))
    with open(os.path.join(text_dir, "timescale_table.md"), "a") as handle:
        handle.write(f"\n{line}\n")
    print(f"  {line.replace('**', '')}")

    # The model's lag windows in its own tau_env, matched to every recording per nu.
    matches = {}
    if not IO.has(timescale, "nu", "window_tau", "tau_env", "amp_excess"):
        print(f"  [skip] timescale_matched_window: no check_timescale.csv with nu, window_tau, tau_env and amp_excess under {phase7}/text")
    else:
        frame, matches = matched_table(rows, timescale, model_windows(timescale))
        written += IO.write_table(frame, os.path.join(output_base, "Phase7", "text"), "timescale_matched_window")

    figure = _timescales(rows, label)
    if figure is not None:
        written += S.savefig_pair(figure, os.path.join(output_base, "Phase6", "images"), "cross_f6_timescales")

    images7 = os.path.join(output_base, "Phase7", "images")
    summary = IO.read_table(phase7, "recovery_summary.csv", quiet=True)
    placed = _placed(rows, recordings, "T_over_tau")
    if summary is None or not IO.has(summary, "p_true", "d", WINDOW, "median_bias"):
        print(f"  [skip] cross_f7_recordings_on_time_axis: no recovery_summary.csv with p_true, d, {WINDOW} and median_bias under {phase7}/text")
    elif not placed:
        print("  [skip] cross_f7_recordings_on_time_axis: no recording carries a finite T_over_tau")
    else:
        written += S.log_twin(_on_time_axis, images7, "cross_f7_recordings_on_time_axis", summary, placed, label)

    placed = _placed(rows, recordings, "amp_excess")
    if not matches:
        print("  [skip] cross_f7_amplitude_memory: no model lag window to match (see timescale_matched_window above)")
    elif not placed:
        print("  [skip] cross_f7_amplitude_memory: no recording carries a finite amp_excess")
    else:
        figure = _amplitude_memory(timescale, placed, matches, label)
        if figure is not None:
            written += S.savefig_pair(figure, images7, "cross_f7_amplitude_memory")
    return written
