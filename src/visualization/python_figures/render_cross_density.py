"""Cross-recording F3: one copula density row per modality.

  cross_f3_density_by_modality
      For one representative recording of each modality, one selected pair: the empirical, Gaussian and fitted-t densities and the three departures, in the Phase 5 layout and on the Phase 5 colour convention.

The representative recording is the first of that modality, in the `seed_base` order, that has a Phase 5 density MAT file.
The pair is the first of the highest concordance band that file carries (high, else mid, else low), so the row shows the regime where a departure is visible.
"""

from __future__ import annotations

import os
import string
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _phase_io as IO  # noqa: E402
import _style as S  # noqa: E402
import render_phase5_density as R_density  # noqa: E402

plt = S.style()

PREFERRED_BANDS = ("high", "mid", "low")


def _pick_pair(pairs):
    """The first pair of the highest concordance band available, else the first pair."""
    by_band = {}
    for pair in pairs:
        band = IO.as_text(pair.get("band", [])).strip().lower() if "band" in pair else ""
        by_band.setdefault(band, pair)
    for band in PREFERRED_BANDS:
        if band in by_band:
            return by_band[band]
    return pairs[0]


def representative_rows(recordings):
    """(modality, recording, pair) for each modality that has a Phase 5 density MAT file."""
    chosen = []
    seen = set()
    for recording in recordings:
        modality = str(recording.get("modality", ""))
        if modality in seen:
            continue
        pairs = IO.pair_density(IO.recording_phase(recording, 5))
        if not pairs:
            continue
        seen.add(modality)
        chosen.append((modality, recording, _pick_pair(pairs)))
    return chosen


def _density_by_modality(log, rows, label):
    """The six Phase 5 panels of each representative pair, one row per modality; None without rows."""
    if not rows:
        return None
    fig, axes = plt.subplots(len(rows), 6, figsize=(S.WIDTH, 1.30 * len(rows) + 1.0), layout="constrained", squeeze=False)
    letters = list(string.ascii_uppercase)
    for index, (modality, recording, pair) in enumerate(rows):
        R_density.draw_row(fig, list(axes[index]), pair, log, letters[6 * index : 6 * index + 6], label_first=(index == 0), short=True)
        name = S.recording_label(recording.get("dataset_key", recording["rel_base"]))
        axes[index][0].set_ylabel(f"{modality}\n{name}", labelpad=1, fontsize=6.5, color=S.modality_color(modality))
    S.legend_below(fig, R_density.colour_keys(), ncols=2, reserve=0.05 + 0.05 / len(rows))
    S.source_label(fig, label)
    return fig


def render(recordings, output_base, label="cross-recording"):
    """Write cross_f3_density_by_modality and its log twin under `<output_base>/Phase5/images`; return the paths."""
    rows = representative_rows(recordings)
    if not rows:
        print("  [skip] cross_f3_density_by_modality: no recording has a Phase 5 density mat")
        return []
    return S.log_twin(_density_by_modality, os.path.join(output_base, "Phase5", "images"), "cross_f3_density_by_modality", rows, label)
