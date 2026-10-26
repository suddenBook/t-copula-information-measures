"""Readers for the tables and MAT files the t-copula phases write, and the writer of the cross-recording tables.

Every phase writes `<rel_base>/PhaseN/{mat,text,images}`: the numbers land in CSVs under `text/`, and the arrays too wide for a CSV (the 41-node profiles, the packed tau and R, the density grids) stay in `-v7.3` MAT files under `mat/`, which are HDF5 and are read here with h5py.
A renderer therefore never builds a path itself: it asks this module for a table by name and gets a DataFrame, or for a MAT field and gets an array.

What the readers guarantee

  * A missing table is not an error.
    `read_table` returns None and says which file was absent, so a driver run on a phase that skipped a step draws what exists and skips the rest instead of aborting the phase's figures.
  * A missing column is an error that names the column (`require`), because a figure drawn from a column that silently defaulted misrepresents the data.
    `pick` is the tolerant variant for a statistic a summary table may spell more than one way.
  * `units(source)` concatenates `pairs.csv` (d = 2) and `groups_long.csv` (d >= 3) into one flat unit table, so a figure over all dimensions is written once.
  * `unit_key` and `join_units` join any two unit tables on (d, unit_index), the identity of a unit.
  * MATLAB's column-major arrays arrive from h5py transposed.
    A `[u x v]` density grid therefore arrives as `[v, u]`, which is exactly what `pcolormesh(u_edges, v_edges, C)` expects, and the readers leave it that way.
  * Unit rows are stored column-wise: one dataset per field with one row per unit, not a struct array of per-element object references.
    `struct_columns` reads such a struct into `{field: array}`, undoing the transposition so that a `[n x 20]` MATLAB array is `(n, 20)` here, and decodes the `status` and `reason` code columns through the `status_labels` and `reason_labels` lists stored beside them.

Discovery

  `discover_recordings(results_root)` globs `*/*/*/recording.json`, the sidecar Phase 1 writes at each recording's base, and returns the recordings sorted by `seed_base`, so every cross-recording figure and table orders them the same way.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

UNIT_KEY = ["d", "unit_index"]


# ---------------------------------------------------------------- paths
def phase_paths(source):
    """Resolve `{phase_dir, mat_dir, text_dir, images_dir, phase, label}` from a phase directory.

    `source` is normally the phase directory a driver receives, but a path to its `mat/`, `text/` or `images/` child, or to a file inside one of them, is accepted too, so the same resolver serves an interactive call.
    `label` is the recording's `<family>/<subject>/<condition>`, the three folders above the phase directory.
    """
    src = os.path.abspath(str(source))
    if os.path.isfile(src):
        src = os.path.dirname(src)
    if os.path.basename(src) in ("mat", "text", "images"):
        src = os.path.dirname(src)
    parts = src.split(os.sep)
    label = "/".join(parts[-4:-1]) if len(parts) >= 4 else os.path.basename(src)
    return {"phase_dir": src, "mat_dir": os.path.join(src, "mat"), "text_dir": os.path.join(src, "text"), "images_dir": os.path.join(src, "images"), "phase": os.path.basename(src), "label": label}


# ---------------------------------------------------------------- tables
def read_table(source, name, quiet=False, columns=None):
    """`<source>/text/<name>` as a DataFrame, or None when it is absent or has no rows.

    `columns` keeps only the named columns the file actually has, which keeps a cross-recording rollup small: the per-unit tables of several recordings run to millions of rows, of which a figure needs a few columns.
    """
    path = os.path.join(phase_paths(source)["text_dir"], name)
    if not os.path.isfile(path):
        if not quiet:
            print(f"  [skip] {name}: not found under {phase_paths(source)['text_dir']}")
        return None
    wanted = set(columns) if columns else None
    frame = pd.read_csv(path, usecols=(lambda c: c in wanted) if wanted else None)
    if frame.empty:
        if not quiet:
            print(f"  [skip] {name}: no rows")
        return None
    frame.attrs["path"] = path
    frame.attrs["name"] = name
    return frame


def read_tables(source, pattern, quiet=False):
    """Every `text/` table matching a glob pattern, concatenated in sorted path order, or None when none has rows."""
    text_dir = phase_paths(source)["text_dir"]
    paths = sorted(glob.glob(os.path.join(text_dir, pattern)))
    frames = [pd.read_csv(p) for p in paths]
    frames = [f for f in frames if not f.empty]
    if not frames:
        if not quiet:
            print(f"  [skip] {pattern}: no table matched under {text_dir}")
        return None
    out = pd.concat(frames, ignore_index=True)
    out.attrs["name"] = pattern
    return out


def require(frame, *columns):
    """The named columns (one Series, or a list for several); raises a KeyError naming the table and every missing column."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"{frame.attrs.get('name', 'table')} ({frame.attrs.get('path', 'unknown path')}) is missing required column(s): {', '.join(missing)}")
    return [frame[c] for c in columns] if len(columns) > 1 else frame[columns[0]]


def has(frame, *columns):
    """True when the table is present and carries every named column."""
    return frame is not None and all(c in frame.columns for c in columns)


def pick(frame, *candidates, default=None):
    """The first candidate column present, or `default` (None unless given).

    A summary table may spell one statistic more than one way (`median_p_hat`, `p_hat_median`), and a figure that needs the statistic rather than the spelling asks here.
    With None back, the caller can skip the panel and say which names it looked for.
    """
    for name in candidates:
        if frame is not None and name in frame.columns:
            return frame[name]
    return default


def numeric(series):
    """A column as a float array, with every entry that does not parse as a number set to NaN."""
    return pd.to_numeric(pd.Series(series), errors="coerce").to_numpy(dtype=float)


def flag(series):
    """A logical column as a bool array, whether it was written true/false or 1/0.

    The branch is chosen by dtype and never by a test for `object`: pandas 3 gives a column of strings its own `str` dtype, and an `object` test would send `true`/`false` down the numeric branch, where every row coerces to NaN and the whole column reads as false.
    """
    column = pd.Series(series)
    if column.dtype == bool:
        return column.to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(column):
        return (pd.to_numeric(column, errors="coerce").fillna(0.0) > 0).to_numpy()
    spelled = column.astype(str).str.strip().str.lower()
    return spelled.isin(("true", "1", "1.0", "yes")).to_numpy()


def text(series):
    """A column as a stripped string array."""
    return pd.Series(series).astype(str).str.strip().to_numpy(dtype=object)


# ---------------------------------------------------------------- the unit table
def units(source, quiet=False):
    """`pairs.csv` and `groups_long.csv` as one flat unit table, or None when both are absent.

    Either half may be absent (a run that fitted only d = 2, or only the groups), and what is present is returned.
    Rows keep every column the CSV carried.
    """
    frames = []
    for name in ("pairs.csv", "groups_long.csv"):
        frame = read_table(source, name, quiet=quiet)
        if frame is not None:
            frames.append(frame)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    out.attrs["name"] = "pairs.csv + groups_long.csv"
    out.attrs["path"] = phase_paths(source)["text_dir"]
    return out


def unit_key(frame):
    """The identity of each unit, (d, unit_index), as a pandas MultiIndex."""
    require(frame, *UNIT_KEY)
    return pd.MultiIndex.from_arrays([frame["d"].to_numpy(), frame["unit_index"].to_numpy()], names=UNIT_KEY)


def join_units(left, right, columns=None, how="left"):
    """Join two unit tables on (d, unit_index), keeping `columns` (by default every column) from the right."""
    require(left, *UNIT_KEY)
    require(right, *UNIT_KEY)
    keep = UNIT_KEY + [c for c in (columns or [c for c in right.columns if c not in UNIT_KEY]) if c in right.columns]
    merged = left.merge(right[keep], on=UNIT_KEY, how=how, suffixes=("", "_right"))
    merged.attrs.update(left.attrs)
    return merged


def ok_units(frame, column="numerical_ok"):
    """The numerically successful rows, the population every distribution is drawn over.

    A finite estimate does not mean a successful fit: a failed refinement and a rejected endpoint both leave a finite `p_hat` behind, so a quantitative figure filters on the validity flag and never on `isfinite`.
    `column` names that flag: `numerical_ok` for a fitted unit or a simulation replicate, `numerical_ok_obs` for the observed fit behind a Phase 3 surrogate unit.
    Without a `numerical_ok` column, the rows whose `status` is not `numerical_failure` are kept.
    A table carrying neither the flag nor a `status` column is returned whole, so a phase that wrote neither still draws.
    """
    if column in frame.columns:
        return frame[flag(frame[column])]
    if column == "numerical_ok" and "status" in frame.columns:
        return frame[text(frame["status"]) != "numerical_failure"]
    return frame


def dimensions(frame, column="d"):
    """Every dimension the table carries, sorted, as ints."""
    d = np.unique(numeric(frame[column]))
    return d[np.isfinite(d)].astype(int)


def by_dimension(frame, column, statistic="median", dimension_column="d"):
    """(dimensions, statistic of `column` per dimension), over the rows given."""
    d = dimensions(frame, dimension_column)
    grouped = frame.groupby(dimension_column)[column]
    values = getattr(grouped, statistic)() if isinstance(statistic, str) else grouped.apply(statistic)
    return d, np.array([values.get(k, np.nan) for k in d], dtype=float)


def quantile_by_dimension(frame, column, q, dimension_column="d"):
    """The q-quantile of `column` per dimension, aligned to `dimensions(frame)`."""
    d = dimensions(frame, dimension_column)
    grouped = frame.groupby(dimension_column)[column].quantile(q)
    return d, np.array([grouped.get(k, np.nan) for k in d], dtype=float)


def fraction_by_dimension(frame, mask, dimension_column="d"):
    """The fraction of rows satisfying a boolean mask, per dimension."""
    d = dimensions(frame, dimension_column)
    work = frame.assign(_flag=np.asarray(mask, dtype=float))
    grouped = work.groupby(dimension_column)["_flag"].mean()
    return d, np.array([grouped.get(k, np.nan) for k in d], dtype=float)


# ---------------------------------------------------------------- mats
def mat_path(source, name):
    """`<source>/mat/<name>` when it exists, else None."""
    path = os.path.join(phase_paths(source)["mat_dir"], name)
    return path if os.path.isfile(path) else None


def open_mat(path):
    """A read-only h5py handle on a `-v7.3` MAT file, which the caller closes."""
    import h5py

    return h5py.File(path, "r")


def _deref(handle, value):
    """An HDF5 object reference resolved to the array it points to; any other value as an array."""
    import h5py

    if isinstance(value, h5py.Reference):
        return np.asarray(handle[value])
    return np.asarray(value)


def _matlab_class(node):
    """The `MATLAB_class` attribute of a dataset, as a string ('' when absent)."""
    value = node.attrs.get("MATLAB_class", b"")
    return value.decode() if isinstance(value, bytes) else str(value)


def struct_columns(handle, name, fields=None):
    """A MATLAB column struct of unit rows as `{field: array}`, one row per unit.

    Every field is one dataset: `[n x 1]` arrives as `(n,)`, `[n x w]` as `(n, w)` with the transposition undone, a char stored once (`dataset_key`) as a string, and a cellstr (`status_labels`) as a list of strings.
    A field with a `<name>_labels` list beside it is a code column and is returned decoded, so `status` reads as the label strings the CSVs carry and no caller has to know it is stored as int8 codes.
    A field stored empty arrives as '' for a char and as an empty array otherwise, and a code outside its label list decodes as ''.

    `fields` keeps only the named fields the struct actually has; the label list a requested code column needs is read for the decoding whether or not it was requested.
    """
    group = handle[name] if isinstance(name, str) else name
    keys = list(group.keys())
    wanted = [k for k in (fields if fields else keys) if k in keys]
    needed = list(wanted)
    for key in wanted:
        if f"{key}_labels" in keys and f"{key}_labels" not in needed:
            needed.append(f"{key}_labels")

    out = {}
    for key in needed:
        node = group[key]
        if node.attrs.get("MATLAB_empty", 0):
            # A -v7.3 file stores an empty array as a dataset of its dimensions, flagged MATLAB_empty.
            out[key] = "" if _matlab_class(node) == "char" else np.zeros(0)
            continue
        raw = np.asarray(node)
        if raw.dtype.kind == "O":
            out[key] = [as_text(_deref(handle, ref)) for ref in raw.ravel()]
        elif _matlab_class(node) == "char" or raw.dtype == np.uint16:
            out[key] = as_text(raw)
        else:
            block = raw.T if raw.ndim == 2 else raw.reshape(-1, 1)
            out[key] = block[:, 0] if block.shape[1] == 1 else block

    for key in wanted:
        labels = out.get(f"{key}_labels")
        if labels is None:
            continue
        code = np.asarray(out[key], dtype=int).ravel()
        out[key] = np.array([labels[c - 1] if 1 <= c <= len(labels) else "" for c in code], dtype=object)
    return {key: out[key] for key in wanted}


def concat_columns(blocks):
    """Several column dicts stacked row-wise into one, keeping the fields every block carries."""
    blocks = [b for b in blocks if b]
    if not blocks:
        return {}
    if len(blocks) == 1:
        return blocks[0]
    shared = [k for k in blocks[0] if all(k in b for b in blocks)]
    out = {}
    for key in shared:
        parts = [np.asarray(b[key]) for b in blocks]
        if parts[0].ndim == 0:
            out[key] = blocks[0][key]  # a scalar the files share, such as dataset_key
            continue
        if parts[0].ndim == 2 and len({p.shape[1] for p in parts}) != 1:
            continue  # a width that differs between files cannot be stacked
        out[key] = np.concatenate(parts, axis=0)
    return out


def struct_array(handle, name, fields=None):
    """A MATLAB struct array as a list of dicts, one per element.

    A `1 x N` struct array stores each field as an N-long dataset of object references, and a scalar struct stores the field's data directly.
    Both are read here, so a run that produced one unit and a run that produced ten thousand read the same way.
    """
    group = handle[name] if isinstance(name, str) else name
    keys = list(fields) if fields else list(group.keys())
    keys = [k for k in keys if k in group]
    if not keys:
        return []
    lengths = []
    for key in keys:
        node = group[key]
        lengths.append(int(np.asarray(node).size) if node.dtype.kind == "O" else 1)
    count = max(lengths) if lengths else 0
    out = []
    for index in range(count):
        element = {}
        for key in keys:
            node = group[key]
            raw = np.asarray(node)
            if raw.dtype.kind == "O":
                flat = raw.ravel()
                element[key] = _deref(handle, flat[index]) if index < flat.size else np.array([])
            else:
                element[key] = raw
        out.append(element)
    return out


def find_struct_array(handle, fields):
    """The name of the first top-level struct that carries every named field, or None.

    A renderer finds its struct by the fields it reads rather than by the variable name (`units`, `pairs`, ...), so it does not depend on how a phase names its variables.
    """
    import h5py

    for key in handle.keys():
        node = handle[key]
        if isinstance(node, h5py.Group) and all(f in node for f in fields):
            return key
    return None


def fit_units(source, fields=("d", "unit_index", "p_hat", "status", "profile_p_grid", "profile_loglik_grid")):
    """Phase 2's unit columns, for the fields a figure needs, from the fit MAT files.

    `fit_all.mat`, the merged index, is read first, but it leaves out the two 41-node profile vectors, because at production scale a second copy of them would double a multi-gigabyte tree for numbers only the profile figure reads.
    So when the merged file does not carry every requested field, the per-dimension `fit_dXX.mat` files, which carry them all, are read instead, in dimension order.
    A phase interrupted before its merge has only those files and draws from them the same way.

    Returns `{field: array}` with one row per unit (see `struct_columns`), or `{}` when no fit MAT file is there.
    """
    mat_dir = phase_paths(source)["mat_dir"]
    merged = mat_path(source, "fit_all.mat")
    # The pattern is `fit_d[0-9][0-9].mat` and not `fit_d*.mat` because Phase 2 saves through a staging file `<name>.<token>.tmp.mat`, and a run cut off mid-save leaves a half-written one that the looser pattern would match and try to open.
    per_dimension = sorted(glob.glob(os.path.join(mat_dir, "fit_d[0-9][0-9].mat")))
    paths = [merged] if merged else per_dimension
    if merged and per_dimension and fields:
        with open_mat(merged) as handle:
            complete = "units" in handle and all(f in handle["units"] for f in fields)
        if not complete:
            paths = per_dimension

    blocks = []
    for path in paths:
        with open_mat(path) as handle:
            name = "units" if "units" in handle else find_struct_array(handle, ["p_hat"])
            if name is None:
                continue
            blocks.append(struct_columns(handle, name, fields))
    return concat_columns(blocks)


def pair_density(source):
    """Phase 5's per-pair density grids as a list of dicts in the MAT file's order, or [] when the file or its struct is missing."""
    path = mat_path(source, "pair_density.mat")
    if path is None:
        print(f"  [skip] pair_density.mat: not found under {phase_paths(source)['mat_dir']}")
        return []
    with open_mat(path) as handle:
        name = find_struct_array(handle, ["empirical", "gaussian", "t"])
        if name is None:
            print(f"  [skip] {path}: no struct carries empirical / gaussian / t")
            return []
        return struct_array(handle, name)


def as_text(value):
    """A MATLAB char array read from a MAT file as a Python string.

    Control characters other than tab and newline are dropped, not rendered: a number that reached a char field by accident would otherwise arrive here as an unprintable codepoint, and matplotlib would warn once per figure that the glyph is missing from the font.
    """
    arr = np.asarray(value).ravel()
    return "".join(chr(int(c)) for c in arr if int(c) >= 32 or int(c) in (9, 10))


def vector(value):
    """A MAT field as a 1-D float array."""
    return np.asarray(value, dtype=float).ravel()


def scalar(value, default=np.nan):
    """The first element of a MAT field as a float, or `default` when the field is empty."""
    arr = np.asarray(value, dtype=float).ravel()
    return float(arr[0]) if arr.size else float(default)


# ---------------------------------------------------------------- discovery
def discover_recordings(results_root, keys=None):
    """Every recording under a results root, from its `recording.json`, by `seed_base`.

    Returns one dict per recording: the sidecar's own fields plus `base` (the recording directory) and `rel_base` (its three-component relative path), with `dataset_key` defaulting to `rel_base`.
    A recording without a `seed_base` sorts last, ties sort by `rel_base`, and an unreadable sidecar is reported and skipped.
    `keys` optionally keeps only the named dataset keys, in the discovered order.
    """
    pattern = os.path.join(str(results_root), "*", "*", "*", "recording.json")
    found = []
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path) as handle:
                meta = json.load(handle)
        except (ValueError, OSError) as error:
            print(f"  [skip] unreadable recording.json: {path} ({error})")
            continue
        meta["base"] = os.path.dirname(path)
        meta["rel_base"] = os.path.relpath(meta["base"], str(results_root)).replace(os.sep, "/")
        meta.setdefault("dataset_key", meta["rel_base"])
        found.append(meta)
    found.sort(key=lambda r: (float(r.get("seed_base", np.inf) or np.inf), r["rel_base"]))
    if keys:
        wanted = list(keys)
        found = [r for r in found if r.get("dataset_key") in wanted]
    return found


def recording_phase(recording, phase):
    """`<recording base>/Phase<N>`, whether or not it exists."""
    return os.path.join(recording["base"], f"Phase{int(phase)}")


def has_phase(recording, phase):
    """True when the recording's `Phase<N>` directory has a `text/` folder."""
    return os.path.isdir(os.path.join(recording_phase(recording, phase), "text"))


# ---------------------------------------------------------------- writing tables
def write_table(frame, text_dir, stem, float_format="%.6g"):
    """Write one table as `<stem>.csv` and `<stem>.md` into `text_dir`; return both paths.

    Every cross-recording number is read by a person in the Markdown and by a script in the CSV, and the two are generated from one frame so they cannot disagree.
    """
    os.makedirs(text_dir, exist_ok=True)
    csv_path = os.path.join(text_dir, f"{stem}.csv")
    md_path = os.path.join(text_dir, f"{stem}.md")
    frame.to_csv(csv_path, index=False, float_format=float_format)
    with open(md_path, "w") as handle:
        handle.write(markdown_table(frame, float_format))
    return [csv_path, md_path]


def markdown_table(frame, float_format="%.6g"):
    """A DataFrame as a GitHub-flavoured Markdown table, with None and non-finite floats as empty cells."""

    def cell(value):
        if isinstance(value, float) and np.isfinite(value):
            return float_format % value
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return ""
        return str(value)

    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join(["---"] * len(frame.columns)) + "|"
    rows = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows]) + "\n"
