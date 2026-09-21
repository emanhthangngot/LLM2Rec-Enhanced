from __future__ import annotations

import unittest

from preflight import make_coverage_report, validate_coverage, validate_manifest


class PreflightContractTests(unittest.TestCase):
    def test_coverage_gate_and_manifest_contract(self) -> None:
        report = make_coverage_report(
            catalog_ids=range(20),
            image_ids=range(20),
            test_target_ids=range(10),
            decode_failures=[],
        )
        self.assertAlmostEqual(report.catalog_coverage, 1.0)
        self.assertAlmostEqual(report.test_target_coverage, 1.0)
        validate_coverage(report)
        validate_manifest(
            {
                "source_commit": "a" * 40,
                "clip_model_id": "openai/clip-vit-base-patch32",
                "clip_resolved_revision": "b" * 40,
                "catalog_count": 20,
                "test_target_count": 10,
                "feature_hashes": {
                    "image_manifest_sha256": "c" * 64,
                    "visual_features_sha256": "d" * 64,
                },
                "coverage": report.to_dict(),
            }
        )

    def test_coverage_shortfall_fails_closed(self) -> None:
        report = make_coverage_report(range(10), range(7), range(5), range(2))
        with self.assertRaises(RuntimeError):
            validate_coverage(report)

    def test_id_conflict_fails_closed(self) -> None:
        report = make_coverage_report(range(3), [0, 1, 9], range(2))
        with self.assertRaises(RuntimeError):
            validate_coverage(report)

    def test_revision_must_be_immutable(self) -> None:
        report = make_coverage_report([1], [1], [1])
        with self.assertRaises(ValueError):
            validate_manifest(
                {
                    "source_commit": "a" * 40,
                    "clip_model_id": "openai/clip-vit-base-patch32",
                    "clip_resolved_revision": "main",
                    "catalog_count": 1,
                    "test_target_count": 1,
                    "coverage": report.to_dict(),
                }
            )


if __name__ == "__main__":
    unittest.main()
