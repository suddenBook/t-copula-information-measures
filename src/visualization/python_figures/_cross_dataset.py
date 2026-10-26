"""Strict helpers for prespecified cross-recording analysis strata.

A cross-recording output never mixes the ten non-ictal primary recordings with the two ictal sensitivity recordings.
Each cross-recording table is expected in a role-specific output directory and holds the recordings of one role only.
These helpers let every Python renderer check that separation itself before drawing anything.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

import numpy as np

import _phase_data as D


PRIMARY = "primary"
SECONDARY = "secondary_ictal_sensitivity"
ROLES = (PRIMARY, SECONDARY)
EXPECTED_DATASETS = {
    PRIMARY: ("human_SEEG/001_sleep", "human_SEEG/001_wake", "human_SEEG/002_sleep", "human_SEEG/002_wake", "monkey_LFP/monkey_L/baseline_correct_epoch", "monkey_LFP/monkey_N/baseline_correct_epoch", "monkey_ECoG/EC1", "monkey_ECoG/EC2", "monkey_ECoG/PF1", "monkey_ECoG/PF2"),
    SECONDARY: ("human_epileptic_eeg/sub-01_seizure-04_ictal", "human_epileptic_eeg/sub-04_seizure-01_ictal"),
}
EXPECTED_RECORDING_COUNTS = {role: len(datasets) for role, datasets in EXPECTED_DATASETS.items()}
ROLE_LABELS = {PRIMARY: "prespecified primary case series (10 non-ictal recordings)", SECONDARY: "prespecified secondary ictal sensitivity series (2 recordings)"}


def table_path(source, filename: str) -> str:
    """Resolve ``filename`` from a role-specific PhaseN text directory."""
    return os.path.join(D.phase_paths(source)["text_dir"], filename)


def read_role_table(source, filename: str, required_columns=()):
    """Read and validate one complete, single-role cross-recording table, and return (table, role, path).

    The check rests on the explicit ``analysis_role`` column only, never on modality, family name, fitted values or path substrings.
    Every prespecified recording of the role must be present, as contiguous blocks in the order of EXPECTED_DATASETS, including rows whose scientific estimates are invalid or non-reportable.
    A missing file raises FileNotFoundError; a missing column, a mixed or unknown role, or a wrong recording set or order raises ValueError.
    """
    path = table_path(source, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"required cross-recording table is missing: {path}")
    table = D.read_csv(path)
    required = {"dataset", "analysis_role", *required_columns}
    missing = sorted(required - set(table))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

    roles = np.asarray([str(value) for value in table["analysis_role"]], dtype=object)
    unique_roles = sorted(set(roles.tolist()))
    if len(unique_roles) != 1 or unique_roles[0] not in ROLES:
        raise ValueError(f"{path} must contain exactly one canonical analysis_role; found {unique_roles!r}")
    role = unique_roles[0]

    datasets = np.asarray([str(value) for value in table["dataset"]], dtype=object)
    dataset_runs = []
    for dataset in datasets:
        if not dataset_runs or dataset != dataset_runs[-1]:
            dataset_runs.append(dataset)
    expected_datasets = list(EXPECTED_DATASETS[role])
    if dataset_runs != expected_datasets:
        raise ValueError(f"{path} must contain the exact canonical {role} recording blocks in manifest order; found {dataset_runs!r}")
    if any(not value for value in dataset_runs):
        raise ValueError(f"{path} contains an empty dataset key")
    return table, role, path


def role_stem(base: str, role: str) -> str:
    """Output stem ``<base>_<role>``; an unknown role raises ValueError."""
    if role not in ROLES:
        raise ValueError(f"unknown analysis role: {role!r}")
    return f"{base}_{role}"


def safe_result_path(results_root, result_relpath, *suffix) -> Path:
    """Resolve a canonical result-relative path without permitting traversal."""
    text = str(result_relpath).strip().replace("\\", "/")
    relative = PurePosixPath(text)
    if not text or relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise ValueError(f"unsafe result_relpath: {result_relpath!r}")
    root = Path(results_root).resolve()
    candidate = root.joinpath(*relative.parts, *suffix).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"result_relpath escapes the results root: {result_relpath!r}") from error
    return candidate


def results_root_from_cross_source(source) -> Path:
    """Return the directory immediately above the ``cross_dataset`` component."""
    path = Path(source).resolve()
    if path.is_file():
        path = path.parent
    parts = path.parts
    positions = [index for index, part in enumerate(parts) if part == "cross_dataset"]
    if not positions:
        raise ValueError(f"source is not inside a cross_dataset tree: {source}")
    index = positions[-1]
    if index == 0:
        raise ValueError(f"cross_dataset source has no results-root parent: {source}")
    return Path(*parts[:index])


def unique_record_rows(table):
    """Index of the first row of each recording, in CSV order; a recording whose rows disagree on result_relpath raises ValueError."""
    datasets = [str(value) for value in table["dataset"]]
    indices = []
    seen = {}
    has_paths = "result_relpath" in table
    for index, dataset in enumerate(datasets):
        if dataset not in seen:
            seen[dataset] = index
            indices.append(index)
            continue
        if has_paths:
            first = str(table["result_relpath"][seen[dataset]])
            current = str(table["result_relpath"][index])
            if current != first:
                raise ValueError(f"dataset {dataset!r} has inconsistent result_relpath values")
    return indices
