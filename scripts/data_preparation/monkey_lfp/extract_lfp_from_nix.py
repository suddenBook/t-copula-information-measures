"""
Extract the neural LFP, the behavioural analog signals and the merged TrialEvents of the Brochier et al. reach-to-grasp NIX files into flat MATLAB .mat files, the input of scripts/data_preparation/monkey_lfp/prepare_lfp.m.

Both monkeys are read the same way, from the dataset's *_no_raw.nix files.

Reads:
    data/monkey_LFP/multielectrode_grasp/datasets_nix/i140703-001_no_raw.nix
    data/monkey_LFP/multielectrode_grasp/datasets_nix/l101210-001_no_raw.nix

Writes (the directory prepare_lfp.m reads by default):
    data/monkey_LFP/raw/lfp_extracted/i140703-001_lfp.mat
    data/monkey_LFP/raw/lfp_extracted/l101210-001_lfp.mat
    data/monkey_LFP/raw/lfp_extracted/{session}_log.txt

Both roots are resolved from the repository root, so the script runs from any working directory.
Set TC_DATA_ROOT to point at a data tree held outside the repository.

Run with uv, which supplies the dependencies:
    uv run --with neo --with nixio --with scipy --with numpy --with quantities --with packaging python scripts/data_preparation/monkey_lfp/extract_lfp_from_nix.py

Pass session IDs as positional arguments to extract only a subset:
    uv run ... python extract_lfp_from_nix.py i140703-001

Flat schema of {session}_lfp.mat (1-D arrays are stored as columns, dicts as MATLAB structs):
    X                                 [samples x 96] single, neural LFP in uV, columns sorted by channel ID
    fs, t_start, t_start_units        neural sampling rate (Hz) and start time (s)
    signal_units                      'uV'
    channel_ids                       [96 x 1] channel IDs 1..96 in column order
    connector_aligned_ids             [96 x 1] connector-aligned electrode IDs
    electrode_coords_mm               [96 x 2] electrode x and y coordinates in mm
    electrode_reject_{HFC,LFC,IFC}    [96 x 1] logical per-channel rejection flags of the three quality bands
    X_behav                           [samples x 6] single, behavioural analog signals in behav_signal_units
    behav_fs, behav_t_start, behav_t_start_units, behav_channel_ids, behav_channel_names, behav_signal_units
    trial_events                      struct of per-event columns: time_sec, label, trial_id, performance_str, performance_code, trial_type, reject_HFC, reject_LFC, reject_IFC
    trial_events_units                's'
    metadata                          struct with the monkey, session ID, source NIX path and SHA-256, NIX start time, filter-shift correction (ms), nsx number (-99 when absent), extraction date, script version and a note

Hard failures (no silent fallbacks):
    a missing NIX file;
    a block without exactly one segment, one neural AnalogSignal, one behavioural AnalogSignal or one Event named 'TrialEvents';
    a neural signal other than 96 channels at 1 kHz with channel IDs 1..96, or a behavioural signal without 6 channels;
    missing array_annotations or TrialEvents annotations of unequal length;
    a spike train that clears a rejection flag an earlier spike train of the same channel set;
    performance_str == 'correct_trial' disagreeing with performance_code == 255;
    an output above 2 GiB;
    a write outside the lfp_extracted output folder.

Each run overwrites the session's outputs in data/monkey_LFP/raw/lfp_extracted/, writing each file to a temporary file first and renaming it into place.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import quantities as pq
import scipy.io
import neo.io


SCRIPT_VERSION = "0.1.0"

MONKEY_NAMES = {"i140703-001": "N (Nikos)", "l101210-001": "L (Lilou)"}

REQUIRED_NEURAL_ARRAY_ANNOTATIONS = ("channel_ids", "connector_aligned_ids", "coordinates_x", "coordinates_y")

# The NIX export carries the per-channel electrode-rejection flags on the spike-train annotations, not on the AnalogSignal.
# _extract_electrode_rejection_from_spiketrains collects them per channel for these quality bands.
REJECT_BANDS = ("HFC", "LFC", "IFC")

REQUIRED_EVENT_ARRAY_ANNOTATIONS = ("trial_event_labels", "trial_id", "performance_in_trial", "performance_in_trial_str", "belongs_to_trialtype", "trial_reject_HFC", "trial_reject_LFC", "trial_reject_IFC")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def sha256_of_file(path: Path) -> str:
    """SHA-256 hex digest of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def short_unit_name(quantity) -> str:
    """Map a quantities unit to a readable short name when possible."""
    units = quantity.units
    for short, ref in (("uV", pq.uV), ("mV", pq.mV), ("V", pq.V)):
        try:
            if units == ref:
                return short
        except Exception:
            pass
    # Fall back to quantities' own string rendering.
    return str(units.dimensionality)


def to_object_string_array(seq) -> np.ndarray:
    """Convert a sequence of anything-stringy into a (N,) object array of str."""
    return np.asarray([str(x) for x in seq], dtype=object)


def filter_shift_correction_ms(anasig) -> float:
    """The filter_shift_correction annotation in ms, or NaN when it is absent."""
    val = anasig.annotations.get("filter_shift_correction", None)
    if val is None:
        return float("nan")
    if hasattr(val, "rescale"):
        return float(val.rescale(pq.ms).magnitude)
    # A plain number is in ms, the convention of the NIX metadata.
    return float(val)


def _repository_root(start: Path) -> Path:
    """Walk up from `start` to the first directory that holds both src/ and scripts/.

    It applies the same rule as src/shared/io/repoRoot.m, so MATLAB and Python agree on the repository root wherever this file sits in the tree.
    """
    start_resolved = start.resolve()
    for candidate in (start_resolved, *start_resolved.parents):
        if (candidate / "src").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError(f"repository root (a directory containing src/ and scripts/) not found above {start_resolved}")


def _resolve_root_from_env(default_root: Path, environment_name: str) -> Path:
    """Apply an optional root override from the environment, with the same rule as resolveRootFromEnv.m.

    An absolute value is taken as is; a relative one is resolved against the current working directory.
    No directory is created here.
    """
    raw = os.environ.get(environment_name, "").strip()
    if not raw:
        return default_root
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


def _assert_path_within_root(target: Path, root: Path) -> Path:
    """Resolve target and assert it stays within the expected output root."""
    root_resolved = root.resolve()
    target_resolved = target.resolve()
    try:
        target_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to write outside expected output root.\n  target: {target_resolved}\n  root:   {root_resolved}") from exc
    return target_resolved


def _atomic_savemat(target: Path, mdict: dict) -> int:
    """Write a compressed MAT file (v5 format) through a temporary file and an atomic rename, and return its size in bytes.

    An output above 2 GiB raises RuntimeError, because scipy.io.savemat cannot write the v7.3 format.
    """
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.stem}_", suffix=".tmp", dir=str(target.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        scipy.io.savemat(str(tmp_path), mdict, do_compression=True, format="5", oned_as="column", long_field_names=True)
        size_bytes = tmp_path.stat().st_size
        size_gib = size_bytes / (1024**3)
        if size_bytes > 2 * (1024**3):
            raise RuntimeError(f"output {target.name} is {size_gib:.2f} GB, exceeds the MATLAB v7 2 GB limit. scipy.io.savemat cannot write v7.3 — reduce payload or switch writer.")
        os.replace(tmp_path, target)
        return size_bytes
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _atomic_write_text(target: Path, text: str) -> None:
    """Write a text file through a temporary file and an atomic rename."""
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.stem}_", suffix=".tmp", dir=str(target.parent), text=True)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with open(tmp_path, "w") as f:
            f.write(text)
        os.replace(tmp_path, target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ----------------------------------------------------------------------------
# Core extractor
# ----------------------------------------------------------------------------


def _neural_channel_metadata(anasig):
    """Channel IDs, connector-aligned IDs and x, y coordinates (mm) from the array_annotations of the neural AnalogSignal."""
    aa = anasig.array_annotations
    missing = [k for k in REQUIRED_NEURAL_ARRAY_ANNOTATIONS if k not in aa]
    if missing:
        raise RuntimeError(f"Neural AnalogSignal is missing required array_annotations: {missing}")

    channel_ids = np.asarray(aa["channel_ids"], dtype=np.float64)
    connector_aligned_ids = np.asarray(aa["connector_aligned_ids"], dtype=np.float64)

    cx = aa["coordinates_x"]
    cy = aa["coordinates_y"]
    if hasattr(cx, "rescale"):
        cx = cx.rescale(pq.mm).magnitude
    if hasattr(cy, "rescale"):
        cy = cy.rescale(pq.mm).magnitude
    coords_x = np.asarray(cx, dtype=np.float64)
    coords_y = np.asarray(cy, dtype=np.float64)

    return {"channel_ids": channel_ids, "connector_aligned_ids": connector_aligned_ids, "coords_x": coords_x, "coords_y": coords_y}


def _extract_electrode_rejection_from_spiketrains(seg, channel_ids, session_id):
    """
    Collect the per-channel electrode_reject_{HFC,LFC,IFC} flags from the spike-train annotations of the NIX segment.

    The NIX export attaches these flags to each spike train's annotations (keyed by `channel_id`), not to the array_annotations of the neural AnalogSignal.
    The flag is a per-channel odML property, so all spike trains of a channel should carry the same value.
    A channel is flagged when one of its spike trains carries the flag, and a spike train that clears a flag set by an earlier spike train on the same channel raises RuntimeError.
    Spike trains without a channel_id, or on a channel outside channel_ids, are skipped.

    A channel without any spike train is not flagged.
    In this export a silent channel has no spike trains and no known quality issue, while a channel flagged in the odML metadata keeps at least one spike train that carries the flag.
    """
    n = len(channel_ids)
    flags = {b: np.zeros(n, dtype=bool) for b in REJECT_BANDS}
    seen = np.zeros(n, dtype=bool)
    channel_to_index = {int(c): i for i, c in enumerate(channel_ids)}

    for st in seg.spiketrains:
        ann = st.annotations
        ch = ann.get("channel_id", None)
        if ch is None:
            continue
        ch = int(ch)
        if ch not in channel_to_index:
            continue
        idx = channel_to_index[ch]
        seen[idx] = True
        for band in REJECT_BANDS:
            key = f"electrode_reject_{band}"
            if key not in ann:
                continue
            val = bool(ann[key])
            if flags[band][idx] and not val:
                # An earlier unit on this channel set the flag and this one clears it: the spike trains disagree.
                raise RuntimeError(f"[{session_id}] spike trains on channel {ch} disagree on {key}: at least one says True and at least one says False. Refusing to silently pick one.")
            if val:
                flags[band][idx] = True

    return flags


def _extract_neural(seg, session_id: str):
    """The neural LFP of the segment as float32 [samples x 96] in uV, columns sorted by channel ID, with its channel metadata and rejection flags."""
    neural = [a for a in seg.analogsignals if a.annotations.get("neural_signal", False)]
    if len(neural) != 1:
        raise RuntimeError(f"[{session_id}] expected exactly 1 neural AnalogSignal (annotations.neural_signal=True), got {len(neural)}")
    anasig = neural[0]

    if anasig.shape[1] != 96:
        raise RuntimeError(f"[{session_id}] expected 96 neural channels, got shape {anasig.shape}")

    fs_hz = float(anasig.sampling_rate.rescale(pq.Hz).magnitude)
    if abs(fs_hz - 1000.0) > 1e-9:
        raise RuntimeError(f"[{session_id}] expected 1 kHz neural sampling rate, got {fs_hz} Hz")

    # Rescale to microvolts and extract as float32 [T x 96].
    try:
        X_q = anasig.rescale(pq.uV)
    except Exception as exc:
        raise RuntimeError(f"[{session_id}] neural AnalogSignal could not be rescaled to uV: {exc}") from exc
    X = np.asarray(X_q.magnitude, dtype=np.float32)
    t_start = float(anasig.t_start.rescale(pq.s).magnitude)

    meta = _neural_channel_metadata(anasig)

    # Deterministic column order: sort by channel_id.
    order = np.argsort(meta["channel_ids"])
    X = X[:, order]
    channel_ids = meta["channel_ids"][order]
    connector_aligned_ids = meta["connector_aligned_ids"][order]
    coords_x = meta["coords_x"][order]
    coords_y = meta["coords_y"][order]

    expected_ids = np.arange(1, 97, dtype=np.float64)
    if not np.array_equal(channel_ids, expected_ids):
        raise RuntimeError(f"[{session_id}] channel_ids after sorting != 1..96; first 10 = {channel_ids[:10]}, last 5 = {channel_ids[-5:]}")

    # The per-channel electrode-rejection flags live on the spike-train annotations, not on the AnalogSignal, so they are collected from the spike trains.
    electrode_flags = _extract_electrode_rejection_from_spiketrains(seg_for_spikes(anasig), channel_ids, session_id)

    electrode_coords_mm = np.column_stack([coords_x, coords_y])

    return {
        "X": X,
        "fs": fs_hz,
        "t_start": t_start,
        "channel_ids": channel_ids,
        "connector_aligned_ids": connector_aligned_ids,
        "electrode_coords_mm": electrode_coords_mm,
        "electrode_reject_HFC": electrode_flags["HFC"],
        "electrode_reject_LFC": electrode_flags["LFC"],
        "electrode_reject_IFC": electrode_flags["IFC"],
        "nix_nsx": anasig.annotations.get("nsx", None),
        "nix_filter_shift_correction_ms": filter_shift_correction_ms(anasig),
    }


def seg_for_spikes(anasig):
    """Return the segment that owns this AnalogSignal (needed to find its spike trains)."""
    seg = getattr(anasig, "segment", None)
    if seg is None:
        raise RuntimeError("AnalogSignal is not attached to a Segment — cannot locate its spike trains")
    return seg


def _extract_behavioural(seg, session_id: str):
    """The single behavioural AnalogSignal (6 channels) as float32 in its own units, with its rate, start time and channel IDs and names."""
    behav = [a for a in seg.analogsignals if not a.annotations.get("neural_signal", True)]
    if len(behav) != 1:
        raise RuntimeError(f"[{session_id}] expected exactly 1 behavioural AnalogSignal (annotations.neural_signal=False), got {len(behav)}")
    ban = behav[0]

    if ban.shape[1] != 6:
        raise RuntimeError(f"[{session_id}] expected 6 behavioural channels, got shape {ban.shape}")

    X_behav = np.asarray(ban.magnitude, dtype=np.float32)
    behav_fs = float(ban.sampling_rate.rescale(pq.Hz).magnitude)
    behav_t_start = float(ban.t_start.rescale(pq.s).magnitude)
    behav_units_str = short_unit_name(ban)

    aa = ban.array_annotations
    if "channel_ids" in aa:
        behav_channel_ids = np.asarray(aa["channel_ids"], dtype=np.float64)
    else:
        behav_channel_ids = np.arange(1, ban.shape[1] + 1, dtype=np.float64)
    if "channel_names" in aa:
        behav_channel_names = to_object_string_array(aa["channel_names"])
    else:
        behav_channel_names = np.asarray([f"chan{int(c)}" for c in behav_channel_ids], dtype=object)

    return {"X_behav": X_behav, "behav_fs": behav_fs, "behav_t_start": behav_t_start, "behav_signal_units": behav_units_str, "behav_channel_ids": behav_channel_ids, "behav_channel_names": behav_channel_names}


def _extract_trial_events(seg, session_id: str):
    """The TrialEvents annotations as per-event columns, after checking their lengths and the consistency of the two performance fields, plus a summary for the log."""
    matches = [e for e in seg.events if e.name == "TrialEvents"]
    if len(matches) != 1:
        raise RuntimeError(f"[{session_id}] expected exactly 1 Event with name=='TrialEvents', got {len(matches)}")
    ev = matches[0]

    aa = ev.array_annotations
    missing = [k for k in REQUIRED_EVENT_ARRAY_ANNOTATIONS if k not in aa]
    if missing:
        raise RuntimeError(f"[{session_id}] TrialEvents is missing required array_annotations: {missing}")

    time_sec = np.asarray(ev.times.rescale(pq.s).magnitude, dtype=np.float64)
    label_arr = to_object_string_array(aa["trial_event_labels"])
    trial_id_arr = np.asarray(aa["trial_id"], dtype=np.float64)
    perf_str_arr = to_object_string_array(aa["performance_in_trial_str"])
    perf_code_arr = np.asarray(aa["performance_in_trial"], dtype=np.float64)
    trial_type_arr = to_object_string_array(aa["belongs_to_trialtype"])
    reject_HFC_arr = np.asarray(aa["trial_reject_HFC"], dtype=bool)
    reject_LFC_arr = np.asarray(aa["trial_reject_LFC"], dtype=bool)
    reject_IFC_arr = np.asarray(aa["trial_reject_IFC"], dtype=bool)

    N_events = len(time_sec)
    for name, arr in (
        ("trial_event_labels", label_arr),
        ("trial_id", trial_id_arr),
        ("performance_in_trial_str", perf_str_arr),
        ("performance_in_trial", perf_code_arr),
        ("belongs_to_trialtype", trial_type_arr),
        ("trial_reject_HFC", reject_HFC_arr),
        ("trial_reject_LFC", reject_LFC_arr),
        ("trial_reject_IFC", reject_IFC_arr),
    ):
        if len(arr) != N_events:
            raise RuntimeError(f"[{session_id}] TrialEvents annotation {name!r} length {len(arr)} != N_events {N_events}")

    # ------------------------------------------------------------------
    # Internal consistency between performance_str and performance_code.
    #
    # The event-level equivalence `perf_str=='correct_trial' iff perf_code==255` does not hold on this dataset.
    # The analog-derived events (OT, OR, DO, OBB, FSRplat-ON/OFF, HEplat-ON/OFF) carry `perf_str='unknown'` even in a correct trial with `perf_code=255`.
    # In the NIX export the string field is set only for digital events; analog-derived events carry the numeric code of their trial but not its string label.
    #
    # The checks are therefore:
    #   (1) forward: perf_str=='correct_trial'  ->  perf_code==255
    #   (2) trial level: every trial has one perf_code across all of its events
    #   (3) set equality: the trials with any perf_str=='correct_trial' event are the trials with any perf_code==255 event
    #
    # A failure of any of the three indicates corrupted data.
    # ------------------------------------------------------------------
    is_correct_str = np.asarray([s == "correct_trial" for s in perf_str_arr], dtype=bool)
    is_code_255 = perf_code_arr == 255

    # (1) forward implication
    bad_forward = is_correct_str & ~is_code_255
    n_bad_forward = int(bad_forward.sum())
    if n_bad_forward > 0:
        raise RuntimeError(f"[{session_id}] consistency check FAIL: perf_str=='correct_trial' but perf_code!=255 on {n_bad_forward}/{N_events} events")

    # (2) trial-level perf_code uniformity
    mask_trial = trial_id_arr > 0
    mixed_trials = []
    for tid in np.unique(trial_id_arr[mask_trial]):
        codes = np.unique(perf_code_arr[trial_id_arr == tid])
        if len(codes) > 1:
            mixed_trials.append((int(tid), [int(c) for c in codes]))
    if mixed_trials:
        raise RuntimeError(f"[{session_id}] consistency check FAIL: trials with mixed perf_code across their events: {mixed_trials[:10]} (showing up to 10 of {len(mixed_trials)})")

    # (3) set equality at the trial level
    correct_trials_by_str = set(int(t) for t in trial_id_arr[is_correct_str & mask_trial])
    correct_trials_by_code = set(int(t) for t in trial_id_arr[is_code_255 & mask_trial])
    only_in_str = sorted(correct_trials_by_str - correct_trials_by_code)
    only_in_code = sorted(correct_trials_by_code - correct_trials_by_str)
    if only_in_str or only_in_code:
        raise RuntimeError(f"[{session_id}] consistency check FAIL: correct-trial sets differ. only-in-str={only_in_str[:10]} only-in-code={only_in_code[:10]}")

    trial_events = {"time_sec": time_sec, "label": label_arr, "trial_id": trial_id_arr, "performance_str": perf_str_arr, "performance_code": perf_code_arr, "trial_type": trial_type_arr, "reject_HFC": reject_HFC_arr, "reject_LFC": reject_LFC_arr, "reject_IFC": reject_IFC_arr}

    summary = {"N_events": N_events, "n_distinct_trial_ids": len(set(int(t) for t in trial_id_arr[trial_id_arr > 0])), "n_correct_trials": len(correct_trials_by_str), "perf_str_counts": dict(zip(*np.unique(perf_str_arr, return_counts=True)))}
    return trial_events, summary


def _rejection_unions(neural: dict, trial_events: dict):
    """The 1-based electrode indices and the positive trial IDs flagged in each quality band, and their unions, for the log."""
    el_HFC = np.where(neural["electrode_reject_HFC"])[0] + 1
    el_LFC = np.where(neural["electrode_reject_LFC"])[0] + 1
    el_IFC = np.where(neural["electrode_reject_IFC"])[0] + 1
    el_all = np.where(neural["electrode_reject_HFC"] | neural["electrode_reject_LFC"] | neural["electrode_reject_IFC"])[0] + 1

    tid = trial_events["trial_id"]
    valid = tid > 0

    def trial_union(mask):
        """Sorted positive trial IDs of the events selected by mask."""
        return sorted(set(int(t) for t in tid[mask & valid]))

    tr_HFC = trial_union(trial_events["reject_HFC"])
    tr_LFC = trial_union(trial_events["reject_LFC"])
    tr_IFC = trial_union(trial_events["reject_IFC"])
    tr_all = sorted(set(tr_HFC) | set(tr_LFC) | set(tr_IFC))

    return {"electrode_HFC": sorted(int(x) for x in el_HFC), "electrode_LFC": sorted(int(x) for x in el_LFC), "electrode_IFC": sorted(int(x) for x in el_IFC), "electrode_union": sorted(int(x) for x in el_all), "trial_HFC": tr_HFC, "trial_LFC": tr_LFC, "trial_IFC": tr_IFC, "trial_union": tr_all}


def extract_session(session_id: str, repo_root: Path, monkey_root: Path, out_dir: Path) -> None:
    """Extract one session's NIX file into {session}_lfp.mat and {session}_log.txt in out_dir."""
    nix_dir = monkey_root / "multielectrode_grasp" / "datasets_nix"
    nix_path = nix_dir / f"{session_id}_no_raw.nix"
    mat_path = out_dir / f"{session_id}_lfp.mat"
    log_path = out_dir / f"{session_id}_log.txt"

    if not nix_path.exists():
        raise FileNotFoundError(f"Nix file not found: {nix_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_resolved = out_dir.resolve()
    mat_path = _assert_path_within_root(mat_path, out_dir_resolved)
    log_path = _assert_path_within_root(log_path, out_dir_resolved)

    print(f"[{session_id}] computing sha256 of {nix_path.name} ...", flush=True)
    nix_sha = sha256_of_file(nix_path)
    print(f"[{session_id}]   sha256 = {nix_sha}", flush=True)

    print(f"[{session_id}] opening via neo.io.NixIO ...", flush=True)
    reader = neo.io.NixIO(str(nix_path), mode="ro")
    try:
        block = reader.read_block(lazy=False)
    finally:
        reader.close()

    if len(block.segments) != 1:
        raise RuntimeError(f"[{session_id}] expected exactly 1 segment, got {len(block.segments)}")
    seg = block.segments[0]

    print(f"[{session_id}] extracting neural LFP ...", flush=True)
    neural = _extract_neural(seg, session_id)

    print(f"[{session_id}] extracting behavioural analog ...", flush=True)
    behavioural = _extract_behavioural(seg, session_id)

    print(f"[{session_id}] extracting TrialEvents ...", flush=True)
    trial_events, events_summary = _extract_trial_events(seg, session_id)

    unions = _rejection_unions(neural, trial_events)

    monkey_name = MONKEY_NAMES.get(session_id, session_id)
    rel_nix_path = os.path.relpath(nix_path, repo_root)
    metadata = {
        "monkey": monkey_name,
        "session_id": session_id,
        "source_nix_path": rel_nix_path,
        "source_nix_sha256": nix_sha,
        "nix_t_start_s": float(neural["t_start"]),
        "nix_filter_shift_correction_ms": float(neural["nix_filter_shift_correction_ms"]),
        "nix_source_event_multitag": "TrialEvents",
        "nix_nsx": (int(neural["nix_nsx"]) if neural["nix_nsx"] is not None else -99),  # -99 = absent
        "extraction_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "extraction_script_version": SCRIPT_VERSION,
        "notes": (f"Extracted from {rel_nix_path} via neo.io.NixIO (extract_lfp_from_nix.py v{SCRIPT_VERSION})."),
    }

    mdict = {
        # Neural
        "X": neural["X"],
        "fs": np.float64(neural["fs"]),
        "channel_ids": neural["channel_ids"],
        "t_start": np.float64(neural["t_start"]),
        "t_start_units": "s",
        "signal_units": "uV",
        "connector_aligned_ids": neural["connector_aligned_ids"],
        "electrode_coords_mm": neural["electrode_coords_mm"],
        "electrode_reject_HFC": neural["electrode_reject_HFC"],
        "electrode_reject_LFC": neural["electrode_reject_LFC"],
        "electrode_reject_IFC": neural["electrode_reject_IFC"],
        # Behavioural
        "X_behav": behavioural["X_behav"],
        "behav_fs": np.float64(behavioural["behav_fs"]),
        "behav_channel_ids": behavioural["behav_channel_ids"],
        "behav_channel_names": behavioural["behav_channel_names"],
        "behav_signal_units": behavioural["behav_signal_units"],
        "behav_t_start": np.float64(behavioural["behav_t_start"]),
        "behav_t_start_units": "s",
        # Events
        "trial_events": trial_events,
        "trial_events_units": "s",
        # Metadata
        "metadata": metadata,
    }

    print(f"[{session_id}] writing {mat_path.name} ...", flush=True)
    size_bytes = _atomic_savemat(mat_path, mdict)

    _write_log(log_path, session_id=session_id, monkey_name=monkey_name, rel_nix_path=rel_nix_path, nix_sha=nix_sha, mat_path=mat_path, size_bytes=size_bytes, neural=neural, behavioural=behavioural, events_summary=events_summary, unions=unions)


def _write_log(log_path: Path, *, session_id, monkey_name, rel_nix_path, nix_sha, mat_path, size_bytes, neural, behavioural, events_summary, unions) -> None:
    """Write the plain-text extraction log of one session and print it."""
    lines = []
    L = lines.append
    L(f"=== extract_lfp_from_nix.py v{SCRIPT_VERSION} ===")
    L(f"Session:    {session_id} ({monkey_name})")
    L(f"Source nix: {rel_nix_path}")
    L(f"SHA-256:    {nix_sha}")
    L(f"Output:     {mat_path.name}")
    L(f"File size:  {size_bytes / 1e6:.2f} MB")
    L("")
    L("--- Neural LFP ---")
    L(f"X shape:            {neural['X'].shape}    (samples x channels, {neural['X'].dtype})")
    L(f"fs:                 {neural['fs']} Hz")
    L("signal_units:       uV")
    L(f"t_start:            {neural['t_start']} s")
    if session_id == "i140703-001":
        L("  (= -2.1 ms; the previous prepare_lfp.m read the convenience-file")
        L("  value -2.1 as seconds, shifting events by +2098 samples relative to")
        L("  ground truth. The nix value above is already in seconds.)")
    L(f"nix filter_shift_correction: {neural['nix_filter_shift_correction_ms']} ms")
    L(f"nix nsx:            {neural['nix_nsx']}")
    L(f"channel_ids:        {int(neural['channel_ids'].min())}..{int(neural['channel_ids'].max())} ({len(neural['channel_ids'])} channels, sorted)")
    L(f"electrode reject HFC ({len(unions['electrode_HFC'])}): {unions['electrode_HFC']}")
    L(f"electrode reject LFC ({len(unions['electrode_LFC'])}): {unions['electrode_LFC']}")
    L(f"electrode reject IFC ({len(unions['electrode_IFC'])}): {unions['electrode_IFC']}")
    L(f"electrode reject union ({len(unions['electrode_union'])}): {unions['electrode_union']}")
    L("")
    L("--- Behavioural analog ---")
    L(f"X_behav shape:      {behavioural['X_behav'].shape}")
    L(f"behav_fs:           {behavioural['behav_fs']} Hz")
    L(f"behav_signal_units: {behavioural['behav_signal_units']}")
    if behavioural["behav_signal_units"] != "uV":
        L("  WARNING: behavioural unit differs from neural (uV). Preserved as-is, not re-scaled.")
    L(f"behav_t_start:      {behavioural['behav_t_start']} s")
    L(f"behav_channel_ids:  {[int(c) for c in behavioural['behav_channel_ids']]}")
    L("")
    L("--- TrialEvents ---")
    L(f"N_events:                    {events_summary['N_events']}")
    L(f"distinct positive trial_ids: {events_summary['n_distinct_trial_ids']}")
    L(f"behaviourally correct:       {events_summary['n_correct_trials']}")
    L("performance_in_trial_str distribution:")
    for k, v in sorted(events_summary["perf_str_counts"].items()):
        L(f"  {k!r}: {int(v)}")
    L("consistency checks (all PASS):")
    L("  (1) perf_str=='correct_trial' -> perf_code==255  (forward implication)")
    L("  (2) every trial has uniform perf_code across its events")
    L("  (3) correct-trial set via perf_str == set via perf_code")
    L(f"trial reject HFC ({len(unions['trial_HFC'])}): {unions['trial_HFC']}")
    L(f"trial reject LFC ({len(unions['trial_LFC'])}): {unions['trial_LFC']}")
    L(f"trial reject IFC ({len(unions['trial_IFC'])}): {unions['trial_IFC']}")
    L(f"trial reject union ({len(unions['trial_union'])}): {unions['trial_union']}")

    text = "\n".join(lines) + "\n"
    _atomic_write_text(log_path, text)
    print()
    print(text, end="")
    print(f"[{session_id}] log written to {log_path.name}", flush=True)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


def main(argv=None) -> int:
    """Extract the sessions named on the command line, or both sessions when none is named, and return 0."""
    argv = list(sys.argv[1:] if argv is None else argv)
    # Resolve the data tree from the repository root, never from this file's location: the script lives in scripts/data_preparation/monkey_lfp/ while its inputs and outputs live under data/monkey_LFP/.
    repo_root = _repository_root(Path(__file__).resolve().parent)
    data_root = _resolve_root_from_env(repo_root / "data", "TC_DATA_ROOT")
    monkey_root = data_root / "monkey_LFP"
    # The output folder is the default input root of prepare_lfp.m (TC_LFP_INPUT_ROOT).
    out_dir = monkey_root / "raw" / "lfp_extracted"

    sessions = argv if argv else ["i140703-001", "l101210-001"]
    for sid in sessions:
        print(f"\n==================== {sid} ====================")
        extract_session(sid, repo_root, monkey_root, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
