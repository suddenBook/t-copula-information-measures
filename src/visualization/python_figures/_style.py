"""One colour system, one page geometry and one save path for every t-copula figure.

The module fixes the page width, the fonts, the panel letters, one colour per scientific object and the colour maps, so every figure of the pipeline follows the same conventions.
Nothing here computes science: it maps values to ink and writes files.

The colour system: one scientific object, one colour, and never two objects in one colour

    dimension d        `col_d(d)`, a viridis ramp read by value (d = 2 light, d = 20 dark), so one dimension is one colour in every figure.
                       Colorbars and `c=d` scatters use `cmap_d` with `norm_d`.
    sample size n      `col_n(n)`, a Reds ramp on log10(n).
                       Red means nothing else.
    window T/tau       `col_T(T)`, the same Reds on log10 T/tau over `WINDOW_RANGE` (10 to 1e5), which covers the simulation's declared windows with a margin.
                       In Phase 7 the window is the amount of data, so it takes the colour of the amount of data.
    sampling rate      `col_rate(s)`, the `RATE_CMAP` ramp (Purples) on log2 of the samples per tau over `RATE_RANGE` (5 to 160), one equal step per doubling and no two declared rates alike.
                       Purple stays clear of blue (the surrogate), of Reds (the amount of data) and of the modality hues the rates are drawn beside.
                       It comes nearest the categorical indigo and the darkest dimension, which never share a figure with it.
    modality           `MODALITY` / `modality_color`: SEEG magenta, LFP orange, ECoG brown, used only where a recording is drawn.
    model              `MODEL` / `model_color`: the empirical object viridis mid-tone, the Gaussian reference grey, the fitted t near-black, a surrogate blue.
    status             `STATUS` / `status_color`: the per-unit fit outcomes, on a categorical palette outside every channel above.
    a copula density   `DENS_CMAP` (viridis); a signed departure from it `DIFF_CMAP` (RdBu_r).
    every reference    `GREY`: Gaussian line, zero line, identity line, constant-percentage guide, n^-1/2 guide, resolution floor, image spine.
    legend proxies     `KEY` (near-black) for a key that stands for a glyph shape or a line style rather than for a colour.
    measures           MI, TC, DTC and O are separated by panel, never by colour (`MEASURES`, `MEASURE_LABEL`).

The page

    `WIDTH`, 7.1 in, is the printed text width, and every figure is that wide.
    Text is 8 pt sans-serif, axes have no top or right spine, legends are frameless, and `pdf.fonttype 42` makes the PDF carry real text.
    `style()` sets these and returns pyplot on the Agg backend.
    `panel_label(ax, 'A', 'short text')` letters a panel in bold above its left edge, and `row_label` letters a row of images.
    `legend_below(fig, handles)` puts a frameless key outside the lower centre, and `source_label(fig, text)` stamps the recording or run a figure was drawn from.

The axes

    `dimension_axis(ax, values)` forces integer ticks on a dimension axis and widens the limits to every dimension that was attempted, so a dimension whose fits all failed still shows as an empty slot instead of silently shortening the sweep.
    `log_scale(ax, 'y', values)` turns on a logarithmic axis, or a symmetric-log axis when the values include zero or a negative number, because p_hat is exactly 0 at the Gaussian copula and a log axis would drop that mass without saying so.
    `percent_guides`, `identity_line` and `floor_strip` draw the three grey references that recur across phases.

The files

    `savefig_pair(fig, images_dir, stem)` writes `<stem>.pdf` and `<stem>.png` at 300 dpi and returns both paths.
    `log_twin(fn, images_dir, stem, ...)` calls a render function twice, as `fn(False, ...)` and `fn(True, ...)`, and writes `<stem>` and `<stem>_log`.
    `manifest(images_dir, paths, ...)` adds the files to `<images_dir>/figures_manifest.json`.

The recording

    `modality_of(dataset_key)` and `recording_label(dataset_key)` read a dataset key (`<family>/<subject>/<condition>`).
    A modality outside the three declared names is drawn neutral grey rather than guessed, because a wrong modality colour would mislabel the recording.
"""

from __future__ import annotations

import datetime as _dt
import json
import os

import numpy as np
from matplotlib import cm, colormaps
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, SymLogNorm, to_rgba
from matplotlib.ticker import FixedLocator, MaxNLocator

# ---------------------------------------------------------------- page geometry
WIDTH = 7.1  # the printed text width, in inches
DPI_PDF = 300
DPI_PNG = 300

GREY = "#767676"  # every reference object
KEY = "#1a1a1a"  # legend proxies for a glyph shape or a line style


# ---------------------------------------------------------------- dimension and n
def col_d(d):
    """Colour of dimension d, read by value: d = 2 light, d = 20 dark."""
    position = 0.85 - 0.79 * (float(d) - 2.0) / 18.0
    return cm.viridis(float(np.clip(position, 0.06, 0.85)))


# the same truncated, reversed viridis ramp as col_d, for colorbars and c=d scatters
cmap_d = LinearSegmentedColormap.from_list("dimension", [col_d(d) for d in range(2, 21)], N=256)
norm_d = Normalize(2, 20)


def col_n(n):
    """Colour of sample size n, read by value on a logarithmic scale."""
    t = (np.log10(max(float(n), 1.0)) - np.log10(500.0)) / (np.log10(500000.0) - np.log10(500.0))
    return cm.Reds(float(np.clip(0.35 + 0.60 * t, 0.20, 0.95)))


def cmap_n(values):
    """(cmap, norm) for a sample-size colorbar over the sizes actually run."""
    n = np.unique(np.asarray(values, dtype=float))
    n = n[np.isfinite(n) & (n > 0)]
    if n.size == 0:
        return cm.Reds, Normalize(0.0, 1.0)
    ramp = LinearSegmentedColormap.from_list("sample_size", [col_n(v) for v in np.geomspace(n.min(), max(n.max(), n.min() * 1.001), 32)], N=256)
    return ramp, LogNorm(n.min(), max(n.max(), n.min() * 1.001))


# ---------------------------------------------------------------- the simulation's two axes
# The colour ranges of Phase 7's observation windows and sampling rates.
# They cover the declared grid (T/tau from 10 to 3e4, 5 to 160 samples per tau), the windows with a margin up to 1e5, so a window or a rate keeps its shade whatever a run selects.
WINDOW_RANGE = (10.0, 1e5)
RATE_RANGE = (5.0, 160.0)
RATE_CMAP = "Purples"


def col_T(window):
    """Colour of an observation window T/tau: Reds, read by value on log10 over `WINDOW_RANGE`.

    The span is Reds 0.25 to 1.0, where nine windows a half-decade apart stay distinct.
    """
    low, high = np.log10(WINDOW_RANGE[0]), np.log10(WINDOW_RANGE[1])
    t = (np.log10(max(float(window), 1.0)) - low) / (high - low)
    return cm.Reds(float(np.clip(0.25 + 0.75 * t, 0.20, 1.0)))


def col_rate(rate):
    """Colour of a sampling rate in samples per tau: `RATE_CMAP`, read by value on log2 over `RATE_RANGE`, from 0.40 (light enough to stay visible) to 0.95."""
    low, high = np.log2(RATE_RANGE[0]), np.log2(RATE_RANGE[1])
    t = (np.log2(max(float(rate), 1.0)) - low) / (high - low)
    return colormaps[RATE_CMAP](float(np.clip(0.40 + 0.55 * t, 0.30, 0.95)))


# ---------------------------------------------------------------- categorical objects
MODALITY = {"SEEG": "#e7298a", "LFP": "#d95f02", "ECoG": "#8c510a"}  # magenta, orange, brown
MODALITY_ORDER = ("SEEG", "LFP", "ECoG")

DENS_CMAP = "viridis"  # a copula density
DIFF_CMAP = "RdBu_r"  # a signed difference of copula densities

# The empirical object is the viridis mid-tone, so a line or a marker that stands for the empirical copula carries the colour of the map its density is drawn in.
MODEL = {"empirical": cm.viridis(0.45), "gaussian": GREY, "t": KEY, "surrogate": "#4477aa"}
MODEL_LABEL = {"empirical": "empirical", "gaussian": "Gaussian", "t": "t", "surrogate": "surrogate"}

# The per-unit fit outcomes `tFitUnit` assigns.
# The palette stays off viridis, Reds, the modality hues and grey, so a status never reads as a dimension, a sample size, a modality or a reference.
STATUS = {"ok": "#117733", "heavy_boundary": "#ddcc77", "ambiguous_profile": "#88ccee", "degenerate_pair": "#882255", "numerical_failure": "#aa4499"}
STATUS_ORDER = ("ok", "heavy_boundary", "ambiguous_profile", "degenerate_pair", "numerical_failure")

# A categorical palette for an object with no declared colour (a simulated correlation design), chosen off viridis, Reds, the modality hues and grey.
CATEGORICAL = ("#332288", "#117733", "#88ccee", "#999933", "#cc6677", "#aa4499", "#44aa99")

MEASURES = ("MI", "TC", "DTC", "O")
MEASURE_LABEL = {"MI": "mutual information", "TC": "total correlation", "DTC": "dual total correlation", "O": "O-information"}
MEASURE_MATH = {"MI": r"$\Delta\mathrm{MI}$", "TC": r"$\Delta\mathrm{TC}$", "DTC": r"$\Delta\mathrm{DTC}$", "O": r"$\Delta\Omega$"}


def model_color(name):
    """Colour of a model object; neutral grey for a name outside the declared four."""
    return MODEL.get(str(name).strip().lower(), GREY)


def status_color(name):
    """Colour of a fit status; neutral grey for a status this module does not declare."""
    return STATUS.get(str(name).strip().lower(), GREY)


# Dash patterns that separate the recordings inside one modality colour.
LINESTYLES = ("-", (0, (4, 2)), (0, (1, 1.4)), (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)))


def recording_linestyle(index):
    """Dash pattern number `index`, so two recordings of one modality stay apart."""
    return LINESTYLES[int(index) % len(LINESTYLES)]


def categorical_color(index):
    """Colour number `index` of the categorical ramp, wrapping when it runs out."""
    return CATEGORICAL[int(index) % len(CATEGORICAL)]


def modality_color(modality):
    """The modality colour, or neutral grey for a modality outside the declared three."""
    key = str(modality).strip()
    for name in MODALITY_ORDER:
        if name.lower() == key.lower():
            return MODALITY[name]
    return GREY


def modality_of(dataset_key):
    """Modality named inside a dataset key, or '' when none of the three appears.

    The match is a case-insensitive substring match on the three declared modality names only.
    A recording whose modality is recorded in `recording.json` should be read from there instead; this is the fallback for a table that carries only the key.
    """
    key = str(dataset_key).lower()
    for name in MODALITY_ORDER:
        if name.lower() in key:
            return name
    return ""


def recording_label(dataset_key):
    """A short human label for a dataset key `<family>/<subject>/<condition>`.

    The family is carried by the modality colour, so the label keeps the subject and the condition: `monkey_ECoG/EC1/eyes_closed` reads 'EC1, eyes closed'.
    A key with a single component is kept whole, with its underscores as spaces.
    """
    parts = [p for p in str(dataset_key).replace("\\", "/").split("/") if p]
    if len(parts) >= 2:
        parts = parts[1:]
    return ", ".join(p.replace("_", " ") for p in parts) if parts else str(dataset_key)


# ---------------------------------------------------------------- rcParams
def style():
    """Set the shared rcParams and return pyplot on a non-interactive backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.linewidth": 0.9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "mathtext.fontset": "dejavusans",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "figure.dpi": 150,
        }
    )
    return plt


def panel_label(ax, letter, text=""):
    """Bold panel letter above the left edge of the axes, with an optional short title after it."""
    ax.set_title(r"$\bf{" + letter + "}$" + ("   " + text if text else ""), loc="left", fontsize=8, pad=3)


def row_label(ax, letter, text):
    """Panel letter and title rotated to the left of a row of images."""
    ax.text(-0.50, 0.5, r"$\bf{" + letter + "}$" + ("  " + text if text else ""), transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=8, linespacing=1.3)


def legend_below(fig, handles, ncols=None, y=0.0, reserve=0.12, fontsize=7):
    """One frameless key outside the lower centre, on a strip taken out of the layout; None without handles.

    Every figure carries a legend, and a legend over the data is worse than no legend, so the key sits under the panels and the constrained layout is told to leave room for it.
    `reserve` is the share of the figure height left free at the bottom.
    """
    if not handles:
        return None
    legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), bbox_transform=fig.transFigure, ncols=ncols or min(len(handles), 6), fontsize=fontsize, handlelength=1.3, columnspacing=1.1, handletextpad=0.45, borderpad=0.1)
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(rect=(0, reserve, 1, 1 - reserve))
    return legend


def source_label(fig, text, fontsize=6):
    """Stamp the recording or run a figure was drawn from, bottom right, in grey.

    Provenance, not a caption: a diagnostic figure that does not name its recording is unreadable the moment it leaves its folder.
    """
    if not text:
        return
    # A run of two or more spaces is a line break, because a one-line stamp long enough to carry both the recording and the pair runs into the legend centred beside it.
    import re

    lines = [part for part in re.split(r"\s{2,}", str(text).strip()) if part]
    fig.text(0.995, 0.004, "\n".join(lines), ha="right", va="bottom", fontsize=fontsize, color=GREY, linespacing=1.35)


# ---------------------------------------------------------------- axes helpers
def dimension_axis(ax, values=None, axis="x", label=True, max_ticks=10):
    """Integer ticks on a dimension axis, over every dimension that was attempted.

    `values` is the dimension column of the source table, not of the plotted points: a dimension whose fits all failed keeps its slot on the axis instead of shortening the sweep without saying so.
    """
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(MaxNLocator(integer=True))
    if values is not None:
        d = np.unique(np.asarray(values, dtype=float).ravel())
        d = d[np.isfinite(d)]
        if d.size:
            step = max(1, int(np.ceil(d.size / max(1, max_ticks))))
            # A sweep that skips dimensions (2, 3, 10, 20) would print two labels on top of each other, so a tick closer than a twelfth of the span is dropped.
            minimum_gap = max(1.0, (float(d.max()) - float(d.min())) / 12.0)
            ticks = []
            for value in d[::step]:
                if not ticks or float(value) - ticks[-1] >= minimum_gap:
                    ticks.append(float(value))
            target.set_major_locator(FixedLocator(ticks))
            lo, hi = float(d.min()) - 0.6, float(d.max()) + 0.6
            (ax.set_xlim if axis == "x" else ax.set_ylim)(lo, hi)
    if label:
        (ax.set_xlabel if axis == "x" else ax.set_ylabel)("dimension $d$", labelpad=2)


def log_scale(ax, axis="y", values=None, linthresh=None, numticks=5):
    """A logarithmic axis, or a symmetric-log axis when the values include zero or a negative number; returns the scale used.

    p_hat is exactly 0 at the Gaussian copula and the O-information changes sign, so a plain log axis would drop precisely the values the figure is about.
    The linear window of a symmetric-log axis is the smallest positive value when the quantity is non-negative (a zero then pins to the bottom of the axis and nothing else moves) and a low decile of the magnitudes when the quantity takes both signs (so the bulk near zero gets a readable window instead of a sliver), unless `linthresh` is given.
    The ticks are set explicitly, because `linscale` compresses the linear window and the default locator prints the decades around zero on top of each other.
    """
    v = np.asarray(values, dtype=float).ravel() if values is not None else np.array([])
    v = v[np.isfinite(v)]
    if v.size == 0 or (v > 0).all():
        (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
        return "log"
    signed = bool((v < 0).any())
    if linthresh:
        thresh = float(linthresh)
    elif signed:
        thresh = float(np.percentile(np.abs(v[v != 0]), 10)) if (v != 0).any() else 1e-6
    else:
        positive = v[v > 0]
        thresh = float(positive.min()) if positive.size else 1e-6
    thresh = max(thresh, 1e-12)
    (ax.set_xscale if axis == "x" else ax.set_yscale)("symlog", linthresh=thresh, linscale=0.35)
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(FixedLocator(symlog_ticks(v, thresh, 3 if signed else numticks, signed=signed)))
    if not signed:
        # The quantity never goes negative, so the negative branch carries no data and its decade label would only collide with the zero.
        (ax.set_xlim if axis == "x" else ax.set_ylim)(**{("left" if axis == "x" else "bottom"): 0.0})
    return "symlog"


def symlog_ticks(values, linthresh, count=5, signed=False):
    """Zero plus a thinned set of decades, clear of the compressed linear window."""
    v = np.abs(np.asarray(values, dtype=float).ravel())
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return [0.0]
    first = int(np.ceil(np.log10(max(linthresh, 1e-12)))) + 1
    last = int(np.floor(np.log10(float(v.max()))))
    decades = np.arange(first, last + 1) if last >= first else np.array([last])
    wanted = max(1, (int(count) - 1) // 2 if signed else int(count) - 1)
    descending = decades[::-1]
    step = max(1, int(np.ceil(descending.size / wanted)))
    positive = sorted(10.0 ** descending[::step][:wanted])
    if signed:
        return [-x for x in reversed(positive)] + [0.0] + list(positive)
    return [0.0] + list(positive)


def identity_line(ax, lo=None, hi=None, label="equality"):
    """The grey y = x reference of an estimated-versus-true panel."""
    xlo, xhi = ax.get_xlim() if lo is None else (lo, hi)
    span = [min(xlo, ax.get_ylim()[0]), max(xhi, ax.get_ylim()[1])]
    ax.plot(span, span, color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=1, label=label)


def percent_guides(ax, xlo, xhi, percentages=(1, 10, 100), annotate=True):
    """Lines of constant percentage of the horizontal value, labelled at the left edge."""
    xs = np.array([xlo, xhi], dtype=float)
    for pct in percentages:
        ax.plot(xs, xs * pct / 100.0, color=GREY, lw=0.6, ls=(0, (4, 3)), zorder=1)
        if annotate:
            ax.annotate(f"{pct:g}%", (xlo * 1.9, xlo * 1.9 * pct / 100.0), fontsize=6, color=GREY, ha="left", va="bottom", zorder=4, bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))


def floor_strip(ax, lo, hi, label=None):
    """The grey band spanning a resolution floor, drawn behind the data."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    return ax.axhspan(min(lo, hi), max(lo, hi), color=GREY, alpha=0.20, lw=0, zorder=0, label=label)


def density_norm(values, log=False, vmin=None, vmax=None):
    """Colour normalisation for a copula density panel, linear or logarithmic."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    lo = float(vmin) if vmin is not None else (float(v.min()) if v.size else 0.0)
    hi = float(vmax) if vmax is not None else (float(v.max()) if v.size else 1.0)
    if not log:
        return Normalize(lo, max(hi, lo + 1e-12))
    positive = v[v > 0]
    lo = float(positive.min()) if positive.size else 1e-6
    return LogNorm(lo, max(hi, lo * 10.0))


def departure_norm(values, log=False):
    """Symmetric colour normalisation for a signed departure panel, linear or symlog."""
    v = np.asarray(values, dtype=float)
    v = np.abs(v[np.isfinite(v)])
    hi = float(v.max()) if v.size and v.max() > 0 else 1.0
    if not log:
        return Normalize(-hi, hi)
    positive = v[v > 0]
    thresh = float(np.percentile(positive, 5)) if positive.size else hi / 1000.0
    return SymLogNorm(linthresh=max(thresh, hi / 1e4), vmin=-hi, vmax=hi, base=10)


# ---------------------------------------------------------------- saving
def savefig_pair(fig, images_dir, stem, close=True):
    """Write `<stem>.pdf` and `<stem>.png` at 300 dpi into `images_dir`, close the figure unless `close` is False, and return both paths."""
    os.makedirs(images_dir, exist_ok=True)
    paths = []
    for ext, dpi in (("pdf", DPI_PDF), ("png", DPI_PNG)):
        path = os.path.join(images_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=dpi)
        paths.append(path)
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return paths


def log_twin(render_fn, images_dir, stem, *args, **kwargs):
    """Render one figure twice and save both: `<stem>` linear, `<stem>_log` logarithmic.

    `render_fn(log, *args, **kwargs)` returns a Figure, or None when this twin has nothing to draw (a quantity that is all zero has no log twin).
    Returns the paths written.
    """
    paths = []
    for log, name in ((False, stem), (True, f"{stem}_log")):
        fig = render_fn(log, *args, **kwargs)
        if fig is None:
            continue
        paths.extend(savefig_pair(fig, images_dir, name))
    return paths


def manifest(images_dir, paths, driver="", source="", ok=True):
    """Add the files just written to `<images_dir>/figures_manifest.json`; return its path.

    The manifest mirrors the `text/figures_manifest.json` that `renderPhaseFigures.m` writes: which driver ran, what it read, whether it succeeded, and every figure file written into the directory.
    Adding to the existing list rather than overwriting it lets several renderers and several driver runs land in one manifest.
    An empty `driver` or `source` keeps the value already recorded, and an unreadable manifest is started afresh.
    """
    os.makedirs(images_dir, exist_ok=True)
    path = os.path.join(images_dir, "figures_manifest.json")
    payload = {"driver": str(driver), "source": str(source), "ok": bool(ok), "figures": [], "n_figures": 0, "timestamp": ""}
    if os.path.isfile(path):
        try:
            with open(path) as handle:
                previous = json.load(handle)
            if isinstance(previous, dict):
                payload["figures"] = list(previous.get("figures", []))
                if not driver:
                    payload["driver"] = previous.get("driver", "")
                if not source:
                    payload["source"] = previous.get("source", "")
        except (ValueError, OSError):
            pass
    names = {os.path.basename(str(p)) for p in paths}
    payload["figures"] = sorted(set(payload["figures"]) | names)
    payload["n_figures"] = len(payload["figures"])
    payload["timestamp"] = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    return path


# ---------------------------------------------------------------- recurring glyphs
def violin(ax, groups, positions, colors, widths=0.8, median=True, orientation="vertical"):
    """One violin per group, body in the group's colour, median as a near-black tick; returns the positions drawn.

    The kernel estimate runs past the data at both ends, so each body is trimmed back to the range it was built from.
    A group with fewer than two distinct values has no kernel estimate at all and is drawn as a tick at its value, so its slot on the axis is never silently empty.
    A group with no finite value is skipped.
    """
    value_axis = 1 if orientation == "vertical" else 0
    drawn = []
    for values, position, color in zip(groups, positions, colors):
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        if v.size < 2 or np.allclose(v, v[0]):
            point = ([position], [v[0]]) if orientation == "vertical" else ([v[0]], [position])
            ax.plot(*point, marker="_" if orientation == "vertical" else "|", color=color, ms=6, mew=1.2)
        else:
            parts = ax.violinplot([v], positions=[position], widths=widths, orientation=orientation, showextrema=False, showmedians=False)
            for body in parts["bodies"]:
                path = body.get_paths()[0]
                path.vertices[:, value_axis] = np.clip(path.vertices[:, value_axis], v.min(), v.max())
                body.set_alpha(None)
                body.set_facecolor(to_rgba(color, 0.35))
                body.set_edgecolor(color)
                body.set_linewidth(0.7)
        if median:
            centre = np.median(v)
            point = ([position], [centre]) if orientation == "vertical" else ([centre], [position])
            ax.plot(*point, marker="_" if orientation == "vertical" else "|", color=KEY, ms=5, mew=1.0, zorder=3)
        drawn.append(position)
    return drawn


def image_axes(ax, ticks=(0.0, 0.5, 1.0)):
    """Square axes with a full grey frame, for a copula density or a departure map."""
    ax.set_aspect("equal")
    ax.set_xticks(list(ticks))
    ax.set_yticks(list(ticks))
    ax.tick_params(labelsize=7, length=2.0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(GREY)
        spine.set_linewidth(0.6)
    return ax


def dimension_colorbar(fig, axes, label="dimension $d$", fraction=0.020, pad=0.012):
    """The shared dimension colorbar of a `c=d` scatter."""
    import matplotlib.pyplot as plt

    mappable = plt.cm.ScalarMappable(cmap=cmap_d, norm=norm_d)
    bar = fig.colorbar(mappable, ax=axes, fraction=fraction, pad=pad)
    bar.set_label(label, fontsize=7.5)
    bar.set_ticks([2, 5, 10, 15, 20])
    bar.ax.tick_params(labelsize=7)
    return bar
