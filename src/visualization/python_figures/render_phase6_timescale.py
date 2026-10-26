"""Phase 6 figures: the autocorrelation time scales of one recording.

phase6_acf
    How long does the recording remember its own signal and its own amplitude, and how much longer is the amplitude memory than a Gaussian signal with the same autocorrelation would carry?
    One panel per transform draws the channel-averaged rank-normal autocorrelation against lag in seconds, with its spread across channels, the envelope the fit reads and the fitted exponential exp(-lag / tau).
    The envelope is read at lag 0 and the score's peaks when the score oscillates, and along the function itself when it does not; the score decides once for every curve.
    The corner gives tau, the fit mode and the Ornstein-Uhlenbeck-equivalent time.
    The square's panel also draws its Gaussian reference (the channel mean of the squared score autocorrelations) with that reference's fitted exponential.
    Its corner adds tau_amp_ref, the amplitude excess tau_amp / tau_amp_ref and, when the phase wrote it, the integrated excess tau_amp_int / tau_amp_ref_int (integrated autocorrelation times over the whole window), each 1 for a Gaussian signal in any band.
phase6_channel_timescales
    How far do the channels agree?
    It draws the distribution of each channel's own tau per transform and of its own amplitude excess, fitted and integrated, with the value of the channel average marked, the excess of a Gaussian signal (1) as a reference, and every channel that has no value counted by its reason.

A recording of several trial segments also carries every time scale with each segment centred on its own mean (`<column>_trialc`).
The whole-recording centring is the headline, and the within-trial-centred times and fitted excess are drawn beside it, labelled, never in its place: as a line of the corner notes, and in the channel figure as an outline histogram and a dashed line for the channel average.
No within-trial integrated value is drawn, because centring each trial forces its autocovariances to sum to about zero, so the phase writes NaN there.
A one-segment recording's twins equal its headline and are not drawn.

The `_log` twin of phase6_acf draws |ACF| on a logarithmic axis, on which the fitted exponential is a straight line, so where the envelope stops decaying like one is seen at once.
The across-channel band is a band of signed values and has no logarithm, so that twin leaves it out.
The `_log` twin of phase6_channel_timescales puts tau and the excess on a logarithmic axis.

The autocorrelation is an empirical object and takes the empirical viridis mid tone.
The envelope and the fitted exponential have no colour shared across the figures and take two colours of the categorical ramp.
The Gaussian reference is the Gaussian grey, its fitted exponential dashed like the other fit, and the zero line, the envelope floor, the end of the fit range and the excess of 1 are reference grey.
Every number comes from the phase's CSVs: `acf_curves.csv`, `timescale_summary.csv` and `timescale_by_channel.csv`.
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _phase_io as IO  # noqa: E402
import _style as S  # noqa: E402

plt = S.style()
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator  # noqa: E402

EMPIRICAL = S.model_color("empirical")
ENVELOPE = S.categorical_color(0)
FITTED = S.categorical_color(4)
GAUSSIAN = S.model_color("gaussian")
FIT_DASH = (0, (4, 2))
RANGE_DOTS = (0, (1, 1.4))
TWIN_DASH = (0, (2, 1.2))
# The within-trial-centring twin of a column (`<column>_trialc`).
TRIAL = "_trialc"

# (transform, time scale, panel title, symbol): the score transform gives the signal memory tau_env and the square transform the amplitude memory tau_amp.
TRANSFORMS = (("score", "env", "signal, rank-normal score $Z$", r"$\tau_\mathrm{env}$"), ("square", "amp", "amplitude, $Z^2$", r"$\tau_\mathrm{amp}$"))


def _value(summary, column):
    """One number of the one-row summary, or NaN when the table or the column is absent."""
    if summary is None or column not in summary.columns:
        return np.nan
    values = IO.numeric(summary[column])
    return float(values[0]) if values.size else np.nan


def _text(summary, column):
    if summary is None or column not in summary.columns:
        return ""
    value = summary[column].iloc[0]
    return "" if value is None or (isinstance(value, float) and not np.isfinite(value)) else str(value).strip()


def _trial(summary):
    """True for a recording of several trial segments, whose within-trial-centred twins differ from its headline."""
    segments = _value(summary, "n_segments")
    return bool(np.isfinite(segments) and segments > 1)


def _band(curves, transform):
    """(low, high, low percent, high percent) of the widest quantile band, or None."""
    found = {}
    for column in curves.columns:
        match = re.fullmatch(rf"acf_{transform}_q(\d+)", column)
        if match:
            found[int(match.group(1))] = column
    if len(found) < 2:
        return None
    low, high = min(found), max(found)
    return IO.numeric(curves[found[low]]), IO.numeric(curves[found[high]]), low, high


def _tau_note(summary, role, symbol):
    """The fitted tau in seconds and samples, or why there is none."""
    error = _text(summary, f"{role}_fit_error")
    tau_s = _value(summary, f"tau_{role}_s")
    if error or not np.isfinite(tau_s):
        return f"{symbol}: no fit" + (f"\n({error})" if error else "")
    note = f"{symbol} = {tau_s:.3g} s\n({_value(summary, f'tau_{role}_samples'):.3g} samples)"
    mode = _text(summary, f"{role}_fit_mode")
    if mode:
        note += f"\n{mode} fit"
    if _value(summary, f"{role}_floor_reached") == 0:
        note += "\nno peak below the floor\nwithin the window"
    tau_ou = _value(summary, f"tau_{role}_ou_s")
    if np.isfinite(tau_ou):
        note += f"\nOU-equivalent {tau_ou:.3g} s"
    twin = _value(summary, f"tau_{role}_s{TRIAL}")
    if _trial(summary) and np.isfinite(twin):
        note += f"\nwithin-trial centring {twin:.3g} s"
    return note


def _reference_note(summary):
    """The Gaussian reference's tau and the amplitude excesses, or why there is none."""
    error = _text(summary, "amp_ref_fit_error")
    tau_ref = _value(summary, "tau_amp_ref_s")
    if error or not np.isfinite(tau_ref):
        note = "Gaussian reference: no fit" + (f"\n({error})" if error else "")
    else:
        note = r"Gaussian reference $\tau_\mathrm{amp,ref}$" + f" = {tau_ref:.3g} s"
        if _value(summary, "amp_ref_floor_reached") == 0:
            note += "\n(no peak below the floor)"
    excess = _value(summary, "amp_excess")
    if np.isfinite(excess):
        note += "\nexcess " + r"$\tau_\mathrm{amp}/\tau_\mathrm{amp,ref}$" + f" = {excess:.3g}"
    integrated = _value(summary, "amp_excess_int")
    if np.isfinite(integrated):
        note += "\nintegrated " + r"$\tau_\mathrm{amp,int}/\tau_\mathrm{amp,ref,int}$" + f" = {integrated:.3g}"
    # The fitted excess alone has a within-trial twin, because no integrated time is defined under within-trial centring (the phase writes NaN).
    twin = _value(summary, "amp_excess" + TRIAL)
    if _trial(summary) and np.isfinite(twin):
        note += f"\nwithin-trial centring: excess {twin:.3g}"
    return note


def _acf(log, curves, summary, label):
    """The phase6_acf figure, one panel per transform; the `log` twin draws |ACF| on a logarithmic axis."""
    panels = [entry for entry in TRANSFORMS if f"acf_{entry[0]}" in curves.columns]
    if not panels:
        print("  [skip] phase6_acf: acf_curves.csv carries no acf_score or acf_square column")
        return None
    lag = IO.numeric(IO.require(curves, "lag_s"))
    floor = _value(summary, "envelope_floor")
    fig, axes = plt.subplots(1, len(panels), figsize=(S.WIDTH, 2.8), layout="constrained", squeeze=False)
    handles = {}
    for index, (transform, role, title, symbol) in enumerate(panels):
        ax = axes[0, index]
        acf = IO.numeric(curves[f"acf_{transform}"])
        band = _band(curves, transform)
        if band is not None and not log:
            low, high, p_low, p_high = band
            handles["band"] = ax.fill_between(lag, low, high, color=EMPIRICAL, alpha=0.22, lw=0, label=f"channels, {p_low}th to {p_high}th percentile")
        (handles["acf"],) = ax.plot(lag, np.abs(acf) if log else acf, color=EMPIRICAL, lw=1.0, label="|channel average|" if log else "channel average")
        for column, key, color, style, name in ((f"envelope_{transform}", "envelope", ENVELOPE, "-", "envelope read by the fit"), (f"fitted_{transform}", "fitted", FITTED, FIT_DASH, r"fitted $\exp(-\mathrm{lag}/\tau)$")):
            if column in curves.columns and np.isfinite(IO.numeric(curves[column])).any():
                (handles[key],) = ax.plot(lag, IO.numeric(curves[column]), color=color, lw=1.1, ls=style, label=name)
        note = _tau_note(summary, role, symbol)
        if f"acf_{transform}_gaussian" in curves.columns:
            # The square's Gaussian reference: what its autocorrelation would be if every channel were Gaussian with its own score autocorrelation.
            # It is non-negative, so the log twin draws it as it stands.
            reference = IO.numeric(curves[f"acf_{transform}_gaussian"])
            (handles["reference"],) = ax.plot(lag, reference, color=GAUSSIAN, lw=0.9, zorder=1, label=r"Gaussian reference, mean of $\rho_Z^2$")
            column = f"fitted_{transform}_gaussian"
            if column in curves.columns and np.isfinite(IO.numeric(curves[column])).any():
                (handles["reference_fit"],) = ax.plot(lag, IO.numeric(curves[column]), color=GAUSSIAN, lw=1.0, ls=FIT_DASH, zorder=1, label=r"its fitted $\exp(-\mathrm{lag}/\tau_\mathrm{amp,ref})$")
            note += "\n" + _reference_note(summary)
        if not log:
            ax.axhline(0.0, color=S.GREY, lw=0.8, zorder=0)
        if np.isfinite(floor):
            handles["floor"] = ax.axhline(floor, color=S.GREY, lw=0.8, ls=(0, (3, 2)), zorder=0, label=f"envelope floor {floor:g}")
        last = _value(summary, f"{role}_fit_last_lag_s")
        if np.isfinite(last):
            handles["range"] = ax.axvline(last, color=S.GREY, lw=0.8, ls=RANGE_DOTS, zorder=0, label="end of the fit range")
        ax.text(0.97, 0.95, note, transform=ax.transAxes, ha="right", va="top", fontsize=7, color=S.KEY, bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.85))
        if log:
            positive = np.abs(acf[np.isfinite(acf) & (acf != 0)])
            S.log_scale(ax, "y", positive)
            bottom = floor / 10.0 if np.isfinite(floor) else max(float(positive.min()), 1e-3)
            ax.set_ylim(bottom, 1.3)
        ax.set_xlim(0.0, float(np.nanmax(lag)))
        ax.set_xlabel("lag (s)", labelpad=2)
        if index == 0:
            ax.set_ylabel("|autocorrelation|" if log else "autocorrelation", labelpad=2)
        S.panel_label(ax, "AB"[index], title)
    order = ("acf", "band", "envelope", "fitted", "reference", "reference_fit", "floor", "range")
    S.legend_below(fig, [handles[k] for k in order if k in handles], ncols=3, reserve=0.20)
    S.source_label(fig, _source(summary, label))
    return fig


def _channel_timescales(log, channels, summary, label):
    """The phase6_channel_timescales figure: one histogram panel per time scale, then the fitted and the integrated amplitude excess when the table carries them."""
    panels = [entry for entry in TRANSFORMS if f"tau_{entry[1]}_s" in channels.columns]
    if not panels:
        print("  [skip] phase6_channel_timescales: timescale_by_channel.csv carries no tau_env_s or tau_amp_s column")
        return None
    excess = "amp_excess" in channels.columns
    integrated = "amp_excess_int" in channels.columns
    trial = _trial(summary)
    fig, axes = plt.subplots(1, len(panels) + int(excess) + int(integrated), figsize=(S.WIDTH, 2.6), layout="constrained", squeeze=False)
    twins = False
    for index, (transform, role, title, symbol) in enumerate(panels):
        ax = axes[0, index]
        tau = IO.numeric(channels[f"tau_{role}_s"])
        status = _status(channels, role)
        fitted = (status == "ok") & np.isfinite(tau) & (tau > 0)
        missing = [f"{int(count)} {reason}" for reason, count in _reasons(channels, role, status, fitted)]
        note = f"{int(fitted.sum())} of {status.size} channels fitted"
        if missing:
            note += "\nno tau: " + "\n".join(missing)
        twin, twin_average = _twin(channels, summary, f"tau_{role}_s", trial)
        twins |= twin is not None
        _channel_histogram(ax, log, tau[fitted], note, _value(summary, f"tau_{role}_s"), twin=twin, twin_average=twin_average)
        ax.set_xlabel(f"{symbol} (s)", labelpad=2)
        if index == 0:
            ax.set_ylabel("channels", labelpad=2)
        S.panel_label(ax, "ABCD"[index], title)
    handles = [Patch(facecolor=EMPIRICAL, alpha=0.75, label="each channel's own value"), Line2D([], [], color=FITTED, lw=1.2, label="value of the channel-average autocorrelation")]
    if excess:
        # A channel has no excess when its amplitude time or its Gaussian reference has no fit; the amplitude's reason is named first, the reference's otherwise.
        ax = axes[0, len(panels)]
        value = IO.numeric(channels["amp_excess"])
        has = np.isfinite(value) & (value > 0)
        amp, reference = _status(channels, "amp"), _status(channels, "amp_ref")
        from_amp = ~has & (amp != "ok")
        from_reference = ~has & ~from_amp
        missing = _reasons(channels, "amp", amp, ~from_amp) + [(f"reference {reason}", count) for reason, count in _reasons(channels, "amp_ref", reference, ~from_reference)]
        note = f"{int(has.sum())} of {value.size} channels"
        if missing:
            note += "\nno excess: " + "\n".join(f"{int(count)} {reason}" for reason, count in missing)
        twin, twin_average = _twin(channels, summary, "amp_excess", trial)
        twins |= twin is not None
        _channel_histogram(ax, log, value[has], note, _value(summary, "amp_excess"), reference=1.0, twin=twin, twin_average=twin_average)
        ax.set_xlabel(r"$\tau_\mathrm{amp}\,/\,\tau_\mathrm{amp,ref}$", labelpad=2)
        S.panel_label(ax, "ABCD"[len(panels)], "amplitude excess")
    if integrated:
        # The integrated excess is no fit: a channel has none only where its integrated times are not finite, which is counted without a fit reason.
        ax = axes[0, len(panels) + int(excess)]
        value = IO.numeric(channels["amp_excess_int"])
        has = np.isfinite(value) & (value > 0)
        note = f"{int(has.sum())} of {value.size} channels"
        if (~has).any():
            note += f"\nno integrated excess: {int((~has).sum())}"
        # No within-trial twin, because no integrated time is defined under within-trial centring.
        _channel_histogram(ax, log, value[has], note, _value(summary, "amp_excess_int"), reference=1.0)
        ax.set_xlabel(r"$\tau_\mathrm{amp,int}\,/\,\tau_\mathrm{amp,ref,int}$", labelpad=2)
        S.panel_label(ax, "ABCD"[len(panels) + int(excess)], "integrated excess")
    if excess or integrated:
        handles.append(Line2D([], [], color=S.GREY, lw=0.9, ls=(0, (3, 2)), label="excess of a Gaussian signal, 1"))
    if twins:
        handles += [Line2D([], [], color=S.KEY, lw=0.8, ls=TWIN_DASH, label="within-trial centring, each channel"), Line2D([], [], color=FITTED, lw=1.0, ls=TWIN_DASH, label="within-trial centring, the channel average")]
    # The key spans nearly the page, so it sits above the source stamp's strip.
    # A key of more than three entries takes a second row.
    ncols = len(handles) if len(handles) <= 3 else int(np.ceil(len(handles) / 2))
    rows = int(np.ceil(len(handles) / ncols))
    S.legend_below(fig, handles, ncols=ncols, y=0.085, reserve=0.19 + 0.07 * (rows - 1))
    S.source_label(fig, _source(summary, label))
    return fig


def _twin(channels, summary, column, trial):
    """A trial recording's within-trial-centred channel values of `column` and the channel average's, or (None, NaN) when the recording is not one or the tables carry no twin."""
    if not (trial and column + TRIAL in channels.columns):
        return None, np.nan
    values = IO.numeric(channels[column + TRIAL])
    return values[np.isfinite(values) & (values > 0)], _value(summary, column + TRIAL)


def _status(channels, role):
    """Each channel's fit status for one time scale; derived from its tau when unwritten."""
    if f"{role}_status" in channels.columns:
        return IO.text(channels[f"{role}_status"])
    tau = IO.numeric(channels[f"tau_{role}_s"]) if f"tau_{role}_s" in channels.columns else np.full(len(channels), np.nan)
    return np.where(np.isfinite(tau), "ok", "no_tau").astype(object)


def _channel_histogram(ax, log, values, note, average, reference=None, twin=None, twin_average=np.nan):
    """The channels' values as a histogram, the channel average's value as a line.

    A vertical line widens the autoscaled view to hold itself, so the average and the reference stay in view when every channel sits on one side of them; the log ticks are spread over the same span.
    A within-trial-centred `twin` of the values is an outline on the same bins, and its channel average a dashed line.
    """
    tallest = 1.0
    marks = [v for v in (average, reference, twin_average) if v is not None and np.isfinite(v) and v > 0]
    twin = np.array([]) if twin is None else twin
    pooled = np.concatenate([values, twin])
    if pooled.size:
        bins = int(np.clip(np.ceil(np.sqrt(values.size or twin.size)), 5, 30))
        low, high = float(pooled.min()), float(pooled.max())
        if high <= low:
            low, high = low * 0.9, high * 1.1
        edges = np.geomspace(low, high, bins + 1) if log else np.linspace(low, high, bins + 1)
        if values.size:
            counts, _, _ = ax.hist(values, bins=edges, color=EMPIRICAL, alpha=0.75, lw=0.3, edgecolor="white")
            tallest = float(np.max(counts))
        if twin.size:
            counts, _, _ = ax.hist(twin, bins=edges, histtype="step", color=S.KEY, lw=0.8, ls=TWIN_DASH, zorder=2)
            tallest = max(tallest, float(np.max(counts)))
        if log:
            S.log_scale(ax, "x", pooled)
            _plain_log_ticks(ax.xaxis, min([low] + marks), max([high] + marks))
    if reference is not None:
        ax.axvline(reference, color=S.GREY, lw=0.9, ls=(0, (3, 2)), zorder=2)
    if average is not None and np.isfinite(average):
        ax.axvline(average, color=FITTED, lw=1.2, zorder=3)
    if np.isfinite(twin_average):
        ax.axvline(twin_average, color=FITTED, lw=1.0, ls=TWIN_DASH, zorder=3)
    # Headroom above the tallest bar for the note, one step per line of it.
    ax.set_ylim(0.0, tallest * (1.15 + 0.14 * (note.count("\n") + 1)))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.text(0.97, 0.95, note, transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=S.KEY, bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.85))


def _plain_log_ticks(axis, low, high, count=4):
    """Four plain-number ticks across [low, high] on a log axis.

    The channels of one recording often span less than a decade, where the default locator labels the minor ticks in scientific notation; round geometric steps read better.
    """
    ticks = np.unique([float(f"{value:.2g}") for value in np.geomspace(low, high, count)])
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_minor_locator(NullLocator())


def _reasons(channels, role, status, fitted):
    """(reason, count) for every channel without a tau: its status, and the error it raised.

    An empty error cell reads back as a float NaN (pandas 3 keeps missing strings missing), so anything that is not a non-empty string counts as no error.
    The error is a MATLAB identifier `component:mnemonic`, and the mnemonic alone is kept, so the note fits a third of the page.
    """
    column = f"{role}_error"
    errors = channels[column].to_numpy(dtype=object) if column in channels.columns else np.full(status.size, "", dtype=object)
    reasons = []
    for state, error in zip(status[~fitted], errors[~fitted]):
        error = error.strip().split(":")[-1] if isinstance(error, str) else ""
        reasons.append(f"{state} ({error})" if error and error.lower() != "nan" else str(state))
    return sorted((reason, reasons.count(reason)) for reason in set(reasons))


def _source(summary, label):
    """The source stamp: the recording label with its sample rate, duration and segment count when the summary carries them."""
    fs = _value(summary, "fs")
    segments = _value(summary, "n_segments")
    duration = _value(summary, "duration_s")
    if not np.isfinite(fs):
        return label
    return f"{label}  fs {fs:g} Hz, {duration:.4g} s in {int(segments)} segment(s)" if np.isfinite(segments) and np.isfinite(duration) else f"{label}  fs {fs:g} Hz"


def render(source, out_dir):
    """Render both Phase 6 figures of one recording, each with its `_log` twin, from the CSVs of the phase folder `source`, and return the paths written."""
    label = IO.phase_paths(source)["label"]
    summary = IO.read_table(source, "timescale_summary.csv")
    curves = IO.read_table(source, "acf_curves.csv")
    channels = IO.read_table(source, "timescale_by_channel.csv")
    paths = []
    if curves is not None:
        paths += S.log_twin(_acf, out_dir, "phase6_acf", curves, summary, label)
    if channels is not None:
        paths += S.log_twin(_channel_timescales, out_dir, "phase6_channel_timescales", channels, summary, label)
    return paths
