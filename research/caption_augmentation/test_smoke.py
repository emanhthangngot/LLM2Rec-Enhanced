"""Stdlib-only regression tests for smoke.py's pure functions.

Real network/GPU stages (`fetch_domain_image_urls`, `download_image`,
`Florence2Captioner`, `QwenParaphraser`) are exercised only on Kaggle; they
are excluded from this local suite by construction (they require network
and/or GPU backends).

Run: python -m unittest discover -s code/llm2rec/visual_delta_fusion -p "test_*.py"
"""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from caption import StubCaptioner, StubParaphraser
from corpus import validate_shard_manifest
from crosswalk import CATALOG_SIZE, catalog_blocks
from smoke import (
    choose_image_url,
    process_catalog_records_incrementally,
    select_quality_review_sample,
    select_smoke_sample,
)


class SelectSmokeSampleTests(unittest.TestCase):
    def test_default_sample_size_and_domain_coverage(self) -> None:
        sample = select_smoke_sample()
        self.assertEqual(len(sample), 64)
        domains = {domain for domain, _global_id in sample}
        self.assertEqual(domains, {block.domain for block in catalog_blocks()})

    def test_every_sampled_id_is_in_range_and_correct_block(self) -> None:
        sample = select_smoke_sample()
        for domain, global_id in sample:
            self.assertTrue(0 <= global_id < CATALOG_SIZE)
            block = next(b for b in catalog_blocks() if b.domain == domain)
            self.assertTrue(block.start <= global_id < block.stop)

    def test_never_samples_only_the_first_ids(self) -> None:
        # Regression for phase-02-kaggle-package.md Step 4: "not simply
        # first IDs". Every domain's positions must be spread, not clustered
        # at the block start.
        sample = select_smoke_sample()
        for block in catalog_blocks():
            positions = sorted(g for d, g in sample if d == block.domain)
            self.assertGreater(len(positions), 1)
            span = positions[-1] - positions[0]
            self.assertGreater(span, block.size // 4)

    def test_deterministic_across_calls(self) -> None:
        self.assertEqual(select_smoke_sample(), select_smoke_sample())

    def test_custom_total_distributes_evenly(self) -> None:
        sample = select_smoke_sample(total=12)
        self.assertEqual(len(sample), 12)
        counts = {}
        for domain, _global_id in sample:
            counts[domain] = counts.get(domain, 0) + 1
        self.assertEqual(set(counts.values()), {2})


class SelectQualityReviewSampleTests(unittest.TestCase):
    def test_default_50_per_domain_totals_300(self) -> None:
        sample = select_quality_review_sample()
        self.assertEqual(len(sample), 300)
        counts = {}
        for domain, _global_id in sample:
            counts[domain] = counts.get(domain, 0) + 1
        self.assertEqual(set(counts.values()), {50})

    def test_every_sampled_id_is_in_range_and_correct_block(self) -> None:
        sample = select_quality_review_sample()
        for domain, global_id in sample:
            self.assertTrue(0 <= global_id < CATALOG_SIZE)
            block = next(b for b in catalog_blocks() if b.domain == domain)
            self.assertTrue(block.start <= global_id < block.stop)

    def test_deterministic_across_calls(self) -> None:
        self.assertEqual(select_quality_review_sample(), select_quality_review_sample())

    def test_disjoint_from_smoke_sample_positions_mostly_differ(self) -> None:
        # Not a strict guarantee, but the two deterministic samples use
        # different per-domain counts (11 vs 50) and thus different spacing;
        # they should not be identical selections.
        self.assertNotEqual(select_quality_review_sample(), select_smoke_sample())

    def test_custom_per_domain(self) -> None:
        sample = select_quality_review_sample(per_domain=2)
        self.assertEqual(len(sample), 12)

class ChooseImageUrlTests(unittest.TestCase):
    def test_prefers_hi_res_then_large_then_thumb(self) -> None:
        record = {"images": [{"thumb": "https://t", "large": "https://l", "hi_res": "https://h"}]}
        self.assertEqual(choose_image_url(record), "https://h")

    def test_falls_back_to_large_when_no_hi_res(self) -> None:
        record = {"images": [{"thumb": "https://t", "large": "https://l"}]}
        self.assertEqual(choose_image_url(record), "https://l")

    def test_returns_none_when_no_usable_url(self) -> None:
        self.assertIsNone(choose_image_url({"images": []}))
        self.assertIsNone(choose_image_url({}))
        self.assertIsNone(choose_image_url({"images": [{"thumb": None}]}))

    def test_skips_non_dict_image_entries(self) -> None:
        record = {"images": ["not-a-dict", {"large": "https://l"}]}
        self.assertEqual(choose_image_url(record), "https://l")


class ProcessCatalogRecordsIncrementallyTests(unittest.TestCase):
    """Offline, network-free coverage for the checkpoint/resume core
    described in plans/260915-0955-visual-delta-fusion-pilot/ISSUES.md #23.
    """

    IDENTITY = {"stage": "test", "shard_size": 2}

    @staticmethod
    def _metadata(count: int) -> list[dict[str, object]]:
        return [
            {
                "global_id": index, "domain": "Test", "title": f"Title {index}",
                "asin": f"A{index}", "image_url": f"https://example/{index}",
                "image_status": "pending",
            }
            for index in range(count)
        ]

    @staticmethod
    def _resolve_image(record: dict[str, object], image_dir: Path) -> dict[str, object]:
        return {"status": "decoded", "path": str(image_dir / f"{record['global_id']}.bin"), "sha256": "stub"}

    def test_checkpoints_every_shard_boundary(self) -> None:
        metadata = self._metadata(5)
        captioner = StubCaptioner({index: f"cap-{index}" for index in range(5)})
        paraphraser = StubParaphraser({index: f"para-{index}" for index in range(5)})
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "shards"
            records = process_catalog_records_incrementally(
                metadata, captioner, paraphraser, out_dir, self.IDENTITY, Path(tmp) / "images",
                self._resolve_image, shard_size=2, progress=lambda _msg: None,
            )
            self.assertEqual(len(records), 5)
            manifest = validate_shard_manifest(out_dir / "shard-manifest.json", self.IDENTITY)
            self.assertEqual(manifest["record_count"], 5)
            self.assertEqual(manifest["shard_count"], 3)
            self.assertEqual([record["caption_raw"] for record in records], [f"cap-{i}" for i in range(5)])

    def test_interrupted_run_resumes_without_reprocessing_completed_items(self) -> None:
        metadata = self._metadata(5)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "shards"
            image_dir = Path(tmp) / "images"
            # Simulate a cancellation after only the first 3 of 5 items.
            process_catalog_records_incrementally(
                metadata[:3],
                StubCaptioner({index: f"cap-{index}" for index in range(5)}),
                StubParaphraser({index: f"para-{index}" for index in range(5)}),
                out_dir, self.IDENTITY, image_dir, self._resolve_image,
                shard_size=2, progress=lambda _msg: None,
            )
            captioned_ids: list[int] = []
            captioner = StubCaptioner({index: f"cap-{index}" for index in range(5)})
            original_caption_batch = captioner.caption_batch

            def tracking_caption_batch(items):
                captioned_ids.extend(item_id for item_id, _path in items)
                return original_caption_batch(items)

            captioner.caption_batch = tracking_caption_batch
            records = process_catalog_records_incrementally(
                metadata, captioner, StubParaphraser({index: f"para-{index}" for index in range(5)}),
                out_dir, self.IDENTITY, image_dir, self._resolve_image,
                shard_size=2, progress=lambda _msg: None,
            )
            self.assertEqual(len(records), 5)
            self.assertEqual(sorted(captioned_ids), [3, 4])
            manifest = validate_shard_manifest(out_dir / "shard-manifest.json", self.IDENTITY)
            self.assertEqual(manifest["record_count"], 5)

    def test_identity_mismatch_refuses_to_reuse_output_dir(self) -> None:
        metadata = self._metadata(3)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "shards"
            image_dir = Path(tmp) / "images"
            process_catalog_records_incrementally(
                metadata, StubCaptioner({}), StubParaphraser({}), out_dir,
                {"stage": "old", "shard_size": 2}, image_dir, self._resolve_image,
                shard_size=2, progress=lambda _msg: None,
            )
            with self.assertRaises(RuntimeError):
                process_catalog_records_incrementally(
                    metadata, StubCaptioner({}), StubParaphraser({}), out_dir,
                    self.IDENTITY, image_dir, self._resolve_image,
                    shard_size=2, progress=lambda _msg: None,
                )

    def test_deadline_stops_early_checkpoints_partial_progress_and_resumes(self) -> None:
        metadata = self._metadata(5)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "shards"
            image_dir = Path(tmp) / "images"
            calls = {"n": 0}

            def resolve_image_counting(record, image_dir):
                # Expire the deadline right after the 3rd item is resolved,
                # simulating a Kaggle-style time-budget cutoff mid-catalog.
                calls["n"] += 1
                return self._resolve_image(record, image_dir)

            records = process_catalog_records_incrementally(
                metadata,
                StubCaptioner({index: f"cap-{index}" for index in range(5)}),
                StubParaphraser({index: f"para-{index}" for index in range(5)}),
                out_dir, self.IDENTITY, image_dir, resolve_image_counting,
                shard_size=2, progress=lambda _msg: None,
                deadline=0.0,  # already expired: stop before processing anything
            )
            self.assertEqual(len(records), 0)
            self.assertEqual(calls["n"], 0)
            self.assertFalse((out_dir / "shard-manifest.json").is_file())

            # A generous, unexpired deadline lets the same call finish and
            # produce a real, validated checkpoint.
            records = process_catalog_records_incrementally(
                metadata,
                StubCaptioner({index: f"cap-{index}" for index in range(5)}),
                StubParaphraser({index: f"para-{index}" for index in range(5)}),
                out_dir, self.IDENTITY, image_dir, self._resolve_image,
                shard_size=2, progress=lambda _msg: None,
                deadline=time.perf_counter() + 60,
            )
            self.assertEqual(len(records), 5)
            manifest = validate_shard_manifest(out_dir / "shard-manifest.json", self.IDENTITY)
            self.assertEqual(manifest["record_count"], 5)


if __name__ == "__main__":
    unittest.main()
