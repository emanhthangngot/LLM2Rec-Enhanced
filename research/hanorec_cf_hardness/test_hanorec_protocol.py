from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from audit import audit_artifacts, protocol_errors  # noqa: E402
from prep import _build_frequency_matched_derangement, _seeded_selection  # noqa: E402

PROTOCOL = ROOT / "run-protocol.json"


class HaNoRecProtocolTests(unittest.TestCase):
    def test_protocol_is_complete(self) -> None:
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(protocol_errors(protocol), [])

    def test_selection_is_seeded_and_not_prefix_biased(self) -> None:
        rows = [(index, [index, index + 1, index + 2, index + 3]) for index in range(20)]
        first = _seeded_selection(rows, 5, 2024, "train")
        second = _seeded_selection(rows, 5, 2024, "train")
        other = _seeded_selection(rows, 5, 2025, "train")
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertNotEqual([row[0] for row in first], list(range(5)))

    def test_frequency_shuffle_is_bijective_and_fixed_point_free(self) -> None:
        item_ids = list(range(1, 9))
        frequencies = {1: 5, 2: 5, 3: 4, 4: 4, 5: 3, 6: 3, 7: 2, 8: 2}
        mapping = _build_frequency_matched_derangement(item_ids, frequencies, 2024)
        self.assertEqual(set(mapping), set(item_ids))
        self.assertEqual(set(mapping.values()), set(item_ids))
        self.assertTrue(all(source != target for source, target in mapping.items()))

    def test_artifact_audit_rejects_zero_arms(self) -> None:
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sft_bundle.json").write_text("{}", encoding="utf-8")
            (root / "sft_lora_state.pt").write_bytes(b"state")
            result = audit_artifacts(root, protocol)
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(any("arm matrix mismatch" in error for error in result["errors"]))

    def test_smoke_audit_rejects_missing_compute_telemetry(self) -> None:
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sft_bundle.json").write_text("{}", encoding="utf-8")
            (root / "sft_lora_state.pt").write_bytes(b"state")
            (root / "arm.pt").write_bytes(b"checkpoint")
            (root / "arm_result.json").write_text(
                json.dumps({
                    "weight": 1.0,
                    "image_condition": "real",
                    "status": "COMPLETE",
                    "predictions": [{
                        "ndcg@10": 0.0,
                        "recall@10": 0.0,
                        "candidate_recall@20": 0.0,
                    }],
                    "checkpoint": "arm.pt",
                    "parent_sft_sha256": "hash",
                }),
                encoding="utf-8",
            )
            (root / "smoke_manifest.json").write_text(
                json.dumps({
                    "purpose": "correctness_only_non_signal",
                    "must_not_be_used_as_signal": True,
                }),
                encoding="utf-8",
            )
            result = audit_artifacts(root, protocol)
            self.assertEqual(result["status"], "FAIL")
            self.assertIn("invalid sft_gpu_memory telemetry", result["errors"])
            self.assertIn("invalid dpo_stage_timings optimizer timing", result["errors"])


if __name__ == "__main__":
    unittest.main()
