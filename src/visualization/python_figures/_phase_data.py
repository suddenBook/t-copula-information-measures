"""Shared HDF5 reader for the PhaseN MAT files every Python renderer reads.

Each phase writes results/<family>/<subject>/<condition>/PhaseN/mat/*.mat as MATLAB -v7.3 files (HDF5) through an explicit save list.
The render_phaseN_*.py figures read the persisted top-level variables and diagnostic structs from them.
This module holds the h5py access and the transpose from MATLAB's column-major to h5py's row-major layout, so every renderer reads the same way and the smoke fixtures write the same layout.

Nothing here computes science: it only reads arrays MATLAB has persisted.

The module does not import h5py: every reader takes an already-open handle, so a renderer that reads only CSV files does not need h5py among its dependencies.
"""

from __future__ import annotations

import numpy as np

# Colour palette shared by the PhaseN QC renderers.
PALETTE = {"blue_main": "#0F4D92", "blue_secondary": "#3775BA", "red_strong": "#B64342", "green_3": "#8BCF8B", "neutral": "#CFCECE", "teal": "#42949E", "violet": "#9A4D8E"}


def apply_publication_style(font_size: int = 12, axes_lw: float = 1.4) -> None:
    """Set the shared matplotlib style: sans-serif text, no top or right spines, frameless legends and embeddable (Type 42) PDF fonts."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": font_size, "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"], "axes.linewidth": axes_lw, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.default": "regular"}
    )


def finalize(fig, out_dir, name: str, dpi: int = 300):
    """Save fig as <name>.pdf and <name>.png in out_dir, close it and return the two paths."""
    import os
    import matplotlib.pyplot as plt

    saved = []
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
        saved.append(p)
    plt.close(fig)
    return saved


def recording_name(mat_path) -> str:
    """'family/subject/condition' recovered from .../<f>/<s>/<c>/PhaseN/mat/<x>.mat.

    Figure titles name the recording they belong to rather than a fixed modality.
    Off the expected layout, the MAT file's base name is returned.
    """
    import os

    parts = os.path.abspath(mat_path).split(os.sep)
    # ... family, subject, condition, PhaseN, mat, file.mat
    if len(parts) >= 6 and parts[-2] == "mat" and parts[-3].startswith("Phase"):
        return "/".join(parts[-6:-3])
    return os.path.splitext(os.path.basename(mat_path))[0]


def phase_paths(source) -> dict:
    """Resolve a phase's {phase_dir, mat_dir, text_dir, images_dir, label} from any path inside it.

    Phase 1 hands its renderers a single .mat file.
    Phases 2 to 7 write several files (Phase 2, for example, writes mat/fit_all.mat and one mat/fit_dXX.mat per dimension), so they hand their drivers the PhaseN directory instead, and their tables live in PhaseN/text.
    `source` may be a .mat file, the PhaseN/mat directory, the PhaseN/text directory, the PhaseN/images directory or PhaseN itself.

    `label` is 'family/subject/condition', or the phase directory's name off that layout.
    """
    import os

    src = os.path.abspath(source)
    if os.path.isfile(src):
        phase_dir = os.path.dirname(os.path.dirname(src))
    elif os.path.basename(src) in ("mat", "text", "images"):
        phase_dir = os.path.dirname(src)
    else:
        phase_dir = src
    parts = phase_dir.split(os.sep)
    label = "/".join(parts[-4:-1]) if len(parts) >= 4 else os.path.basename(phase_dir)
    return {"phase_dir": phase_dir, "mat_dir": os.path.join(phase_dir, "mat"), "text_dir": os.path.join(phase_dir, "text"), "images_dir": os.path.join(phase_dir, "images"), "label": label}


def read_csv(path) -> dict:
    """A MATLAB writetable CSV as {column -> np.ndarray}.

    Numeric columns become float arrays (empty cells, NaN and Inf are accepted); MATLAB's logical columns ('true'/'false') become bool arrays; any other column stays an object array of strings.
    This keeps the renderers free of a pandas dependency.
    """
    import csv as _csv

    with open(path, newline="") as fh:
        rows = list(_csv.reader(fh))
    if not rows:
        return {}
    header, body = rows[0], rows[1:]
    out = {}
    for c, name in enumerate(header):
        col = [r[c] if c < len(r) else "" for r in body]
        low = [v.strip().lower() for v in col]
        if col and all(v in ("true", "false") for v in low):
            out[name] = np.array([v == "true" for v in low], dtype=bool)
            continue
        try:
            out[name] = np.array([float(v) if v.strip() != "" else np.nan for v in col], dtype=float)
        except ValueError:
            out[name] = np.array(col, dtype=object)
    return out


def attempted_dimension_span(d_column) -> tuple[int, int] | None:
    """(lowest, highest) interaction order the sweep attempted, or None if unreadable.

    The span is taken from every row of the source table, not from the rows that ended up with a finite estimate.
    A dimension whose valid set is empty still carries a complete summary row (summarizeUnitRowsByDimension writes it with n_ok = 0 and NaN statistics, and applies no dimension cutoff).
    A figure that sized its axis from the plotted points alone would stop at the last dimension with a successful fit and show a collapse as the end of the sweep.
    """
    d = np.asarray(d_column, dtype=float).ravel()
    d = d[np.isfinite(d)]
    if not d.size:
        return None
    return int(np.min(d)), int(np.max(d))


def dimension_ticks(d_column, max_ticks: int = 10) -> np.ndarray:
    """Ticks on the dimensions actually attempted, thinned to at most `max_ticks`.

    Ticking on the attempted values rather than on a contiguous range keeps the axis honest when a sweep skips dimensions: a pilot d_list of [2, 3, 5, 8, 10, 20] gets no ticks at dimensions where nothing was fitted.
    """
    d = np.asarray(d_column, dtype=float).ravel()
    d = np.unique(d[np.isfinite(d)])
    if not d.size:
        return d
    step = max(1, int(np.ceil(d.size / max(1, max_ticks))))
    return d[::step]


def read_matrix(h, name: str) -> np.ndarray:
    """A MATLAB [samples x channels] matrix as [n_channels, n_samples].

    h5py reads a MATLAB [m x n] array transposed to (n, m), which for signals with more samples than channels already yields (n_channels, n_samples).
    The guard puts the longer axis second, assuming more samples than channels, so a matrix with more channels than samples is returned as [n_samples, n_channels]."""
    X = np.asarray(h[name], dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    if X.shape[0] > X.shape[1]:
        X = X.T
    return X


def read_vector(h, name: str) -> np.ndarray:
    """A MATLAB numeric array as a flat float vector."""
    return np.asarray(h[name], dtype=float).ravel()


def read_scalar(h, name: str) -> float:
    """The first element of a MATLAB numeric array as a float."""
    return float(np.asarray(h[name]).ravel()[0])


def read_struct(h, name: str) -> dict:
    """A MATLAB struct as a dict of field -> ravelled numeric array."""
    g = h[name]
    return {k: np.asarray(g[k]).ravel() for k in g.keys()}


def read_str_cell(h, name: str) -> list[str]:
    """A MATLAB cell array of char arrays (stored as object references) as a list of str; an unreadable cell becomes ''."""
    out = []
    for ref in np.asarray(h[name]).ravel():
        try:
            arr = np.asarray(h[ref]).ravel()
            out.append("".join(chr(int(c)) for c in arr if int(c) > 0))
        except Exception:
            out.append("")
    return out


def has(h, name: str) -> bool:
    """True when `name` holds a real value.

    A MATLAB `[]` written under -v7.3 is not a zero-size dataset: it is stored as a shape-(2,) uint64 dimension marker carrying attrs['MATLAB_empty'] = 1.
    A plain `.size > 0` test would report such an empty variable as present (the SEEG `channel_labels` is one) and let the caller read the marker as data.
    A variable with the MATLAB_empty flag therefore counts as absent.
    """
    if name not in h:
        return False
    node = h[name]
    if node.attrs.get("MATLAB_empty", 0):
        return False
    return np.asarray(node).size > 0


def channel_indices(h, path: str) -> np.ndarray:
    """A MATLAB 1-based channel-index vector as 0-based, [] when empty or absent.

    `path` is an h5py path, for example 'clean_info/excluded_channels'.
    A MATLAB `[]` is the shape-(2,) MATLAB_empty marker (see `has`), not a zero-size dataset.
    """
    if path not in h:
        return np.array([], dtype=int)
    node = h[path]
    if node.attrs.get("MATLAB_empty", 0):
        return np.array([], dtype=int)
    return np.asarray(node, dtype=float).ravel().astype(int) - 1


def artifact_time_profile(h) -> dict | None:
    """artifact_info.time_profile as a dict, or None when the file has no time profile.

    The dict holds fraction as [n_channels, n_windows] (h5py reads MATLAB's [channels x windows] transposed) and the window bookkeeping needed to weight the short trailing window.
    """
    if "artifact_info/time_profile/fraction" not in h:
        return None
    tp = h["artifact_info"]["time_profile"]
    # The channel count comes from channel_artifact_pct, which is indexed by the same original channel set as the profile.
    # time_profile.n_channels is the fallback when channel_artifact_pct is absent.
    n_ch = None
    if "artifact_info/channel_artifact_pct" in h:
        n_ch = int(np.asarray(h["artifact_info"]["channel_artifact_pct"]).size)
    elif "n_channels" in tp:
        n_ch = int(np.asarray(tp["n_channels"]).ravel()[0])
    frac = np.asarray(tp["fraction"], dtype=float)
    if frac.ndim == 2 and n_ch is not None and frac.shape[0] != n_ch and frac.shape[1] == n_ch:
        frac = frac.T
    get = lambda k: float(np.asarray(tp[k]).ravel()[0]) if k in tp else None  # noqa: E731
    return {"fraction": frac, "centers": (np.asarray(tp["window_centers_sec"], dtype=float).ravel() if "window_centers_sec" in tp else np.arange(frac.shape[1], dtype=float)), "n_samples": get("n_samples"), "window_sec": get("window_sec"), "fs": read_scalar(h, "fs") if "fs" in h else None}


def sentinel_channels(h) -> np.ndarray:
    """0-based channels whose channel_artifact_pct is the flat 100 sentinel.

    Section 8b of detectSEEGArtifacts.m overwrites channel_artifact_pct with 100 for every channel that carries a channel-level diagnostic flag.
    A 100 there means "diagnostic candidate", not "every sample is an artifact" and not "automatically removed".
    """
    if "artifact_info/channel_artifact_pct" not in h:
        return np.array([], dtype=int)
    pct = np.asarray(h["artifact_info"]["channel_artifact_pct"], dtype=float).ravel()
    return np.flatnonzero(np.isclose(pct, 100.0))


def true_temporal_load_pct(h):
    """(load_pct per original channel, is_true_load).

    The true per-channel artifact percentage is reconstructed from artifact_info.time_profile, weighting each window by the samples it holds so the short trailing window is not reported as low-artifact.
    The result equals 100 * sum(mask) / n_samples, the value of channel_artifact_pct before section 8b of detectSEEGArtifacts writes the sentinel, and satisfies `flag_temporal_rate == (load > temporal_rate_threshold_pct)` exactly.

    Without a usable time profile the sentinel-contaminated channel_artifact_pct is returned with is_true_load=False.
    Callers state which of the two they plot.
    """
    pct = np.asarray(h["artifact_info"]["channel_artifact_pct"], dtype=float).ravel()
    p = artifact_time_profile(h)
    if not p or not p["n_samples"] or not p["window_sec"] or not p["fs"]:
        return pct, False
    frac = p["fraction"]
    win = max(1, int(round(p["window_sec"] * p["fs"])))
    n_win = frac.shape[1]
    samples_per_window = np.full(n_win, float(win))
    samples_per_window[-1] = p["n_samples"] - (n_win - 1) * win
    load = (frac @ samples_per_window) / samples_per_window.sum() * 100.0
    return load, True


def rank_uniform(X: np.ndarray) -> np.ndarray:
    """Per-channel rank transform u = rank / (n_valid + 1) with average ranks for ties, NaN-safe.

    The QC figures use it to recompute the pseudo-observations from the persisted X or X_filtered.
    Input is [n_channels, n_samples], as read_matrix returns it; ranks run along the samples of each channel, and non-finite samples stay NaN and are left out of the ranking.
    On finite data the rule equals that of computeCopulaPseudoObservations.m."""
    from scipy.stats import rankdata

    U = np.full(X.shape, np.nan, dtype=float)
    for c in range(X.shape[0]):
        col = X[c]
        valid = np.isfinite(col)
        n = int(valid.sum())
        if n > 0:
            U[c, valid] = rankdata(col[valid], method="average") / (n + 1)
    return U
