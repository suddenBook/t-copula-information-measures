#!/usr/bin/env python3
"""Unit tests of how render_phase1_excluded_channels finds the artifact-cleaning sidecar and pairs each channel with its reason.

The suite uses synthetic inputs only and reads no recordings.

Run:
    uv run src/visualization/python_figures/test_render_phase1_excluded_channels.py
"""

# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8", "numpy>=1.26", "h5py>=3.10"]
# ///
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import render_phase1_excluded_channels as renderer  # noqa: E402

SIDECAR = "artifact_clean_decision.json"


def _phase_tree(root):
    """Create a Phase1/{mat,text,postprocess} skeleton under root and return the .mat path."""
    for sub in ("mat", "text", "postprocess"):
        (root / "Phase1" / sub).mkdir(parents=True, exist_ok=True)
    return root / "Phase1" / "mat" / "preprocessed_data.mat"


class DefaultSidecarTest(unittest.TestCase):
    """A sidecar in postprocess/ takes precedence over the one in text/, which is the fallback."""

    def test_prefers_postprocess_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mat_path = _phase_tree(root)
            (root / "Phase1" / "text" / SIDECAR).write_text("{}")
            corrected = root / "Phase1" / "postprocess" / SIDECAR
            corrected.write_text("{}")

            resolved = renderer._default_sidecar(str(mat_path))

            self.assertEqual(os.path.realpath(resolved), os.path.realpath(corrected))

    def test_falls_back_to_text_when_postprocess_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mat_path = _phase_tree(root)
            sealed = root / "Phase1" / "text" / SIDECAR
            sealed.write_text("{}")

            resolved = renderer._default_sidecar(str(mat_path))

            self.assertEqual(os.path.realpath(resolved), os.path.realpath(sealed))

    def test_falls_back_when_neither_exists(self):
        """A missing sidecar resolves to the text/ path instead of raising, and _load_reasons then returns an empty mapping, so the figure still renders."""
        with tempfile.TemporaryDirectory() as tmp:
            mat_path = _phase_tree(Path(tmp))

            resolved = renderer._default_sidecar(str(mat_path))

            self.assertTrue(resolved.endswith(os.path.join("text", SIDECAR)))
            self.assertEqual(renderer._load_reasons(resolved), {})


class LoadReasonsTest(unittest.TestCase):
    """How _load_reasons pairs reason strings with channels across the sidecar layouts."""

    def _write(self, directory, payload):
        """Write payload as the sidecar JSON in directory and return its path."""
        path = Path(directory) / SIDECAR
        path.write_text(json.dumps(payload))
        return str(path)

    def test_sealed_misalignment_still_pairs_by_channel(self):
        """A single-array sidecar whose `reasons` is indexed by the 7-entry candidate list while `flagged_channels` holds the 5 removed channels.

        Keying by channel and pairing only arrays of equal length matches the reasons to the candidate array they are indexed by.
        Every channel therefore gets its own reason, and the mismatched 5-against-7 pairing is skipped rather than applied.
        For this layout the re-emitted sidecar and the text/ sidecar give the same reasons."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, {"flagged_channels": [58, 86, 115, 129, 136], "auto_detected_channels": [58, 78, 79, 86, 115, 129, 136], "reasons": ["kurt58", "low78", "low79", "kurt86", "kurt115", "kurt129", "kurt136"]})

            mapping = renderer._load_reasons(path)

            self.assertEqual(mapping[58], "kurt58")
            self.assertEqual(mapping[136], "kurt136")
            self.assertEqual(mapping[78], "low78")

    def test_reasons_matching_neither_array_yield_no_reason(self):
        """When `reasons` matches neither the candidate nor the removed array in length, every pairing is skipped and the figure shows no reason text.

        The loss is silent, which is why the re-emitted sidecar takes precedence."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, {"flagged_channels": [58, 86], "auto_detected_channels": [58, 78, 79], "reasons": ["a", "b", "c", "d"]})

            mapping = renderer._load_reasons(path)

            self.assertEqual(mapping, {})

    def test_aligned_lengths_pair_by_channel(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, {"flagged_channels": [58, 86], "reasons": ["abnormal kurtosis", "abnormal kurtosis"]})

            mapping = renderer._load_reasons(path)

            self.assertEqual(mapping[58], "abnormal kurtosis")
            self.assertEqual(mapping[86], "abnormal kurtosis")

    def test_scalar_flagged_channel_is_accepted(self):
        """A sidecar may store a single removal as a bare scalar rather than a one-element array; np.atleast_1d keeps it readable."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, {"flagged_channels": 40, "reasons": ["abnormal kurtosis"]})

            mapping = renderer._load_reasons(path)

            self.assertEqual(mapping[40], "abnormal kurtosis")

    def test_split_reason_arrays_prefer_the_removed_set(self):
        """When both reason arrays are present the removed set is applied last, so a channel in both carries its removal reason rather than its candidate one."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, {"auto_detected_channels": [24, 31], "auto_detected_reasons": ["isolated", "low power"], "flagged_channels": [24], "reasons": ["abnormal kurtosis"]})

            mapping = renderer._load_reasons(path)

            self.assertEqual(mapping[24], "abnormal kurtosis")
            self.assertEqual(mapping[31], "low power")


if __name__ == "__main__":
    unittest.main()
