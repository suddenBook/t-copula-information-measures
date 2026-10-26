"""Self-test: build the fixture results tree and run every figure driver on it.

A renderer is trusted only once this check passes.
It writes a synthetic tree with the layout and columns of a real run (`fixtures/make_fixtures.py`), runs each phase driver the way `renderPhaseFigures.m` runs it (`uv run --quiet <driver>.py <source> <images dir>`), runs the cross-recording driver on the tree root, and then checks what landed on disk:

  * every driver exited zero;
  * every figure stem carries both a `.pdf` and a `.png`;
  * every images directory carries a `figures_manifest.json` that lists every figure file in it;
  * the Phase 6 and Phase 7 drivers write exactly the stems they declare (`PHASE_EXPECTED`): Phase 7's rate-dependent figures once per rate of the fixture, `_s05` to `_s160`, and no unsuffixed stem beside them;
  * the cross-recording tree carries the time-scale table, the per-nu table of the model windows matched to the recordings, and the figures that place the recordings on Phase 7's axes (`CROSS_EXPECTED`), which skip themselves when an input is absent and would otherwise pass unseen;
  * the time-scale table places every recording against the model (`CROSS_TABLE_COLUMNS`, finite in every row) and carries the within-trial twins of the trial recording (`CROSS_TRIAL_COLUMNS`), and the matched table holds one finite match per recording and model nu (`MATCH_COLUMNS`);
  * a figure summarizes the population the scientific summaries use, so a numerically failed fit that kept a finite estimate does not move a plotted statistic (`check_numerical_ok_filter`);
  * a signed statistic is drawn on an axis that keeps its sign, so a negative median relative error lands on the figure (`check_signed_relative_error`).

The last two checks go through the per-rate split the Phase 7 drivers use, on frames of two rates with distinct estimates, so a split that dropped or mixed rows fails them too.

Run:
    uv run src/visualization/python_figures/render_selftest.py [keep dir]

With no argument the tree is built in a temporary directory and removed afterwards; with a directory it is built there and kept, so the figures can be inspected.
"""

# /// script
# requires-python = ">=3.10"
# dependencies = ["h5py>=3.10", "matplotlib>=3.10", "numpy>=1.26", "pandas>=2.0"]
# ///
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "fixtures"))

import make_fixtures  # noqa: E402

PHASES = (2, 3, 4, 5, 6)
# The cross-recording tree also carries Phase 7: the recordings placed on the simulation's time axis.
# Its outputs named here must exist, so a figure that skips itself on the complete fixture fails the test instead of passing it with a stem fewer.
CROSS_PHASES = PHASES + (7,)
CROSS_EXPECTED = (
    "Phase6/text/timescale_table.csv",
    "Phase6/text/timescale_table.md",
    "Phase7/text/timescale_matched_window.csv",
    "Phase7/text/timescale_matched_window.md",
    "Phase6/images/cross_f6_timescales.png",
    "Phase7/images/cross_f7_recordings_on_time_axis.png",
    "Phase7/images/cross_f7_recordings_on_time_axis_log.png",
    "Phase7/images/cross_f7_amplitude_memory.png",
)
# CROSS_TABLE_COLUMNS: the time-scale table's columns that place a recording against the model, finite in every row of the complete fixture.
# CROSS_TRIAL_COLUMNS: the within-trial twins, finite on its trial recording (n_segments > 1).
# MATCH_COLUMNS: the matched table's columns, finite in its one row per recording and model nu.
CROSS_TABLE_COLUMNS = ("tau_amp_ref_s", "amp_excess", "amp_excess_int", "window_over_tau_env", "tau_amp_ou_over_tau_env")
CROSS_TRIAL_COLUMNS = ("amp_excess_trialc", "T_over_tau_trialc")
MATCH_COLUMNS = ("matched_window_tau", "matched_window_tau_env", "model_amp_excess", "model_amp_excess_int", "model_tau_amp_ou_over_tau_env", "recording_tau_amp_ou_over_tau_env", "shape_memory_ratio")


def _twins(*stems):
    """Each stem and its `_log` twin."""
    return tuple(name for stem in stems for name in (stem, f"{stem}_log"))


def _per_rate(stems, rates):
    """Each stem once per rate, with the rate's suffix (`_s05`, `_s10`, ..., `_s160`)."""
    return tuple(f"{stem}_s{rate:02d}" for rate in rates for stem in stems)


# The stems the Phase 6 and Phase 7 drivers write on the complete fixture, and nothing else: a figure that skips itself, or a rate-dependent stem written without its rate, fails the test.
# Phase 7's rate-dependent stems follow the fixture's rates.
PHASE7_PER_RATE = ("phase7_recovery_spread", "phase7_information_estimated_vs_true", "phase7_information_relative_error")
PHASE7_PER_RATE_TWINS = ("phase7_recovery_bias", "phase7_recovery_gaussian_truth", "phase7_recovery_heavy", "phase7_information_components", "phase7_information_floor")
PHASE_EXPECTED = {
    6: _twins("phase6_acf", "phase6_channel_timescales"),
    7: (
        _per_rate(PHASE7_PER_RATE, make_fixtures.RATES)
        + _twins(*_per_rate(PHASE7_PER_RATE_TWINS, make_fixtures.RATES))
        + ("phase7_recovery_sampling", "phase7_check_law", "phase7_check_coarse", "phase7_check_refinement", "phase7_check_timescale", "phase7_check_sampling")
        + _twins("phase7_factorial_bias", "phase7_check_discretisation")
    ),
}


def check_numerical_ok_filter():
    """A numerically failed fit must not move a plotted statistic; returns the problems found.

    Two replicates at p_true = 0.1 carry a valid estimate of 0.1 and a failed one of 3.
    Both rows are finite, so a renderer that filters on `isfinite` draws a bias of 1450 per cent, while the reporting population `simulationSummaries` uses holds the valid row alone, whose bias is zero.
    A second rate carries its own pair, a valid 0.3 and a failed 3, whose bias is 200 per cent, so a split by rate that mixed the two would draw neither expected value.
    Both spellings of the flag are exercised: the phases write logicals as 1/0, and a table from another writer, or a hand-built frame, may spell them true/false.
    """
    import matplotlib.pyplot as plt
    import render_phase7_shape_recovery as recovery_renderer

    problems = []
    expected = {10.0: 0.0, 20.0: 200.0}
    for spelling in (["true", "false"] * 2, [1, 0] * 2):
        rows = pd.DataFrame({"d": [2] * 4, "T_over_tau": [10] * 4, "samples_per_tau": [10, 10, 20, 20], "p_true": [0.1] * 4, "p_hat": [0.1, 3.0, 0.3, 3.0], "numerical_ok": spelling})
        split = list(recovery_renderer.by_rate(rows))
        if [rate for rate, _ in split] != sorted(expected):
            problems.append(f"numerical_ok filter ({spelling}): the split by rate gave the rates {[rate for rate, _ in split]}, expected {sorted(expected)}")
            continue
        for rate, block in split:
            figure = recovery_renderer._bias(False, block, "self-test")
            if figure is None:
                problems.append(f"numerical_ok filter ({spelling}, rate {rate:g}): the bias panel drew nothing")
                continue
            for index, name in ((0, "median"), (1, "mean")):
                drawn = np.asarray(figure.axes[index].lines[0].get_ydata(), dtype=float)
                if drawn.size != 1 or abs(float(drawn[0]) - expected[rate]) > 1e-9:
                    problems.append(f"numerical_ok filter ({spelling}, rate {rate:g}): {name} bias drew {drawn.tolist()} per cent, expected [{expected[rate]}] from the one valid replicate at that rate")
            plt.close(figure)
    return problems


def _visible(ax, x, y):
    """True where a data point maps to finite display coordinates inside the view.

    `get_ydata` returns what was passed in, not what the axis could draw: on a log axis a negative value is kept in the line and masked by the transform, so a check has to go through the transform and the view limits to see what a reader would see.
    """
    display = ax.transData.transform(np.column_stack([x, y]))
    low, high = sorted(ax.get_ylim())
    return np.isfinite(display).all(axis=1) & (y >= low) & (y <= high)


def check_signed_relative_error():
    """A negative median relative error must be drawn; returns the problems found.

    One dimension and two observation windows carry correction relative errors of -0.2 and -0.1, which a plain log axis would not draw at all.
    The signed row must show both, and the magnitude row must show 0.2 and 0.1.
    A second rate carries -0.4 and -0.3 at the same windows, so the rows of the first rate, taken the way the driver takes them, must hold that rate's values alone.
    """
    import matplotlib.pyplot as plt
    import render_phase7_information as information_renderer
    import render_phase7_shape_recovery as recovery_renderer

    summary = pd.DataFrame({"p_true": [0.1] * 4, "d": [3] * 4, "T_over_tau": [10, 100, 10, 100], "samples_per_tau": [10, 10, 20, 20], "TC_delta_relative_error": [-0.2, -0.1, -0.4, -0.3]})
    figure = information_renderer._relative_error(recovery_renderer.at_rate(summary, 10.0), "self-test")
    if figure is None:
        return ["signed relative error: the relative-error panel drew nothing"]
    problems = []
    for axis_index, name, expected in ((0, "signed", [-0.2, -0.1]), (1, "magnitude", [0.2, 0.1])):
        ax = figure.axes[axis_index]
        series = [line for line in ax.lines if np.allclose(np.asarray(line.get_ydata(), dtype=float), expected)]
        if not series:
            problems.append(f"signed relative error: the {name} row carries no line with the values {expected}")
            continue
        x = np.asarray(series[0].get_xdata(), dtype=float)
        y = np.asarray(series[0].get_ydata(), dtype=float)
        shown = _visible(ax, x, y)
        if not shown.all():
            problems.append(f"signed relative error: the {name} row ({ax.get_yscale()} axis) does not draw {y[~shown].tolist()}")
    plt.close(figure)
    return problems


def _finite_in_every_row(table, path, columns, where=None):
    """One problem for each column that is missing or not finite in every selected row of the table."""
    rows = table if where is None else table[where]
    scope = "every row" if where is None else "every trial recording's row"
    return [f"{path}: no finite {column} in {scope}" for column in columns if column not in rows.columns or rows.empty or not np.isfinite(pd.to_numeric(rows[column], errors="coerce")).all()]


def check_cross_tables(root, cross_base):
    """Check the cross-recording time-scale tables on the complete fixture; returns the problems found.

    The time-scale table must place every recording against the model and carry the trial recording's within-trial twins.
    The matched table must hold exactly one finite match per recording and model nu of the fixture's `check_timescale.csv`.
    """
    problems = []
    table_path = os.path.join(cross_base, "Phase6", "text", "timescale_table.csv")
    match_path = os.path.join(cross_base, "Phase7", "text", "timescale_matched_window.csv")
    model_path = os.path.join(root, "simulation", "t_copula", "all", "Phase7", "text", "check_timescale.csv")
    if not os.path.isfile(table_path):
        return problems  # CROSS_EXPECTED reports the missing table
    table = pd.read_csv(table_path)
    problems += _finite_in_every_row(table, table_path, CROSS_TABLE_COLUMNS)
    trial = pd.to_numeric(table["n_segments"], errors="coerce") > 1
    if not trial.any():
        problems.append(f"{table_path}: no trial recording (n_segments > 1) to carry the within-trial twins")
    else:
        problems += _finite_in_every_row(table, table_path, CROSS_TRIAL_COLUMNS, trial)
    if os.path.isfile(match_path) and os.path.isfile(model_path):
        matches = pd.read_csv(match_path)
        problems += _finite_in_every_row(matches, match_path, MATCH_COLUMNS)
        count = len(pd.unique(pd.read_csv(model_path)["nu"]))
        pairs = matches.groupby("dataset_key")["nu"].agg(["size", "nunique"])
        if len(pairs) != len(table) or not ((pairs["size"] == count) & (pairs["nunique"] == count)).all():
            problems.append(f"{match_path}: expected one row per recording and each of the model's {count} nu, found {pairs.to_dict('index')}")
    return problems


def resolve_uv():
    """The `uv` the drivers run under: the one on PATH, else the first found at the install locations `renderPhaseFigures.m` searches, else the bare name."""
    for candidate in (shutil.which("uv"), os.path.join(os.path.expanduser("~"), ".local", "bin", "uv"), "/usr/local/bin/uv", "/usr/bin/uv"):
        if candidate and os.path.isfile(candidate):
            return candidate
    return "uv"


def run_driver(uv, driver, source, images_dir):
    """Run one driver with the command `renderPhaseFigures.m` builds; return its exit code and combined output."""
    command = [uv, "run", "--quiet", os.path.join(HERE, driver), source, images_dir]
    finished = subprocess.run(command, capture_output=True, text=True)
    return finished.returncode, finished.stdout + finished.stderr


def check_images(images_dir):
    """(problems, stems) for one images directory, where `stems` maps each figure stem to the extensions found."""
    problems, stems = [], {}
    if not os.path.isdir(images_dir):
        return [f"{images_dir}: no images directory"], stems
    for name in sorted(os.listdir(images_dir)):
        stem, extension = os.path.splitext(name)
        if extension in (".pdf", ".png"):
            stems.setdefault(stem, set()).add(extension)
    for stem, extensions in stems.items():
        missing = {".pdf", ".png"} - extensions
        if missing:
            problems.append(f"{images_dir}/{stem}: missing {', '.join(sorted(missing))}")
    manifest_path = os.path.join(images_dir, "figures_manifest.json")
    if not os.path.isfile(manifest_path):
        problems.append(f"{images_dir}: no figures_manifest.json")
    elif stems:
        with open(manifest_path) as handle:
            listed = set(json.load(handle).get("figures", []))
        on_disk = {f"{stem}{extension}" for stem, extensions in stems.items() for extension in extensions}
        if not on_disk <= listed:
            problems.append(f"{manifest_path}: does not list {', '.join(sorted(on_disk - listed))}")
    return problems, stems


def main():
    """Build the fixture, run every driver and check, and print the report; return 1 on any failure."""
    keep = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else None
    root = keep or tempfile.mkdtemp(prefix="tcopula_figures_selftest_")
    uv = resolve_uv()
    failures, problems, total_stems = [], [], 0
    try:
        print(f"fixtures -> {root}")
        make_fixtures.build(root)

        targets = []
        for recording in sorted(os.listdir(root)):
            family = os.path.join(root, recording)
            if not os.path.isdir(family) or recording == "simulation":
                continue
            for subject in sorted(os.listdir(family)):
                for condition in sorted(os.listdir(os.path.join(family, subject))):
                    base = os.path.join(family, subject, condition)
                    if os.path.isfile(os.path.join(base, "recording.json")):
                        targets += [(phase, os.path.join(base, f"Phase{phase}")) for phase in PHASES]
        targets.append((7, os.path.join(root, "simulation", "t_copula", "all", "Phase7")))

        for phase, source in targets:
            driver = f"render_phase{phase}_figures.py"
            images = os.path.join(source, "images")
            code, output = run_driver(uv, driver, source, images)
            found, stems = check_images(images)
            if phase in PHASE_EXPECTED:
                expected = set(PHASE_EXPECTED[phase])
                found += [f"{images}/{stem}: not written" for stem in sorted(expected - set(stems))]
                found += [f"{images}/{stem}: written but not declared" for stem in sorted(set(stems) - expected)]
            problems += found
            total_stems += len(stems)
            status = "ok  " if code == 0 and not found else "FAIL"
            print(f"[{status}] {driver:28s} {os.path.relpath(source, root):46s} {len(stems):3d} stems")
            if code != 0:
                failures.append(driver)
                print(output)

        command = [uv, "run", "--quiet", os.path.join(HERE, "render_cross_recording.py"), root]
        finished = subprocess.run(command, capture_output=True, text=True)
        cross_base = os.path.join(root, "cross_recording")
        cross_stems = 0
        for phase in CROSS_PHASES:
            found, stems = check_images(os.path.join(cross_base, f"Phase{phase}", "images"))
            problems += found
            cross_stems += len(stems)
        total_stems += cross_stems
        missing = [f"{cross_base}/{name}: not written" for name in CROSS_EXPECTED if not os.path.isfile(os.path.join(cross_base, name))]
        missing += check_cross_tables(root, cross_base)
        problems += missing
        status = "ok  " if finished.returncode == 0 and not missing else "FAIL"
        print(f"[{status}] {'render_cross_recording.py':28s} {'(results root)':46s} {cross_stems:3d} stems")
        if finished.returncode != 0:
            failures.append("render_cross_recording.py")
            print(finished.stdout + finished.stderr)

        found = check_numerical_ok_filter()
        problems += found
        print(f"[{'ok  ' if not found else 'FAIL'}] {'numerical_ok filter':28s} {'(two-rate regression)':46s}")
        found = check_signed_relative_error()
        problems += found
        print(f"[{'ok  ' if not found else 'FAIL'}] {'signed relative error':28s} {'(two-rate regression)':46s}")

        tables = [name for phase in CROSS_PHASES for name in sorted(os.listdir(os.path.join(cross_base, f"Phase{phase}", "text")))]
        print(f"\ncross-recording tables: {', '.join(tables) or '(none)'}")
        print(f"figure stems: {total_stems}")
        for problem in problems:
            print(f"  [problem] {problem}")
        if failures:
            print(f"\nSELF-TEST FAILED: {', '.join(failures)}")
        elif problems:
            print(f"\nSELF-TEST FAILED: {len(problems)} problems on disk")
        else:
            print("\nSELF-TEST PASSED")
        return 1 if (failures or problems) else 0
    finally:
        if keep is None:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
