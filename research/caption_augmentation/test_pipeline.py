"""Protocol regressions for the registered v10 matrix."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from pipeline import registered_matrix_runs


class RegisteredMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        protocol = Path(__file__).with_name("experiment.json")
        self.config = json.loads(protocol.read_text(encoding="utf-8"))

    def test_expands_all_profiles_across_three_chain_seeds(self) -> None:
        runs = registered_matrix_runs(self.config)
        self.assertEqual(len(runs), 30)
        self.assertEqual(
            {(run["profile"], run["chain_seed"]) for run in runs},
            {
                (profile, seed)
                for profile in self.config["stage_placement_profiles"]
                for seed in (2024, 2025, 2026)
            },
        )

    def test_native_title_uses_real_cues_without_truncation(self) -> None:
        runs = registered_matrix_runs(self.config)
        native = next(run["definition"] for run in runs if run["profile"] == "native-title")
        self.assertEqual(native["csft_cue_arm"], "real")
        self.assertEqual(native["representation_cue_arm"], "real")
        self.assertEqual(native["history_policy"], "native-full-history")
        self.assertIsNone(native["cue_cap_tokens"])

    def test_missing_profile_definition_fails_closed(self) -> None:
        self.config["profile_definitions"].pop("real-cap64")
        with self.assertRaisesRegex(RuntimeError, "must match exactly"):
            registered_matrix_runs(self.config)


if __name__ == "__main__":
    unittest.main()
