"""Contract tests for the Video_Games paired pilot gate and arm texts."""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from video_games_pilot import (
    PILOT_CATALOG_SIZE,
    PILOT_GLOBAL_START,
    build_paired_arm_records,
    local_item_id,
    validate_pilot_review,
)


def _catalog(status_by_local: dict[int, tuple[str, str]] | None = None) -> list[dict[str, object]]:
    status_by_local = status_by_local or {}
    records = []
    for local in range(1, PILOT_CATALOG_SIZE + 1):
        status, raw = status_by_local.get(local, ("ok", f"color: c{local}"))
        records.append({
            "domain": "Video_Games", "global_id": PILOT_GLOBAL_START + local - 1,
            "title": f"Game {local}", "image_status": "decoded",
            "caption_status": status, "caption_raw": raw,
        })
    return records


class IdMappingTests(unittest.TestCase):
    def test_block_edges_map_to_one_based_downstream_ids(self) -> None:
        self.assertEqual(local_item_id(PILOT_GLOBAL_START), 1)
        self.assertEqual(local_item_id(PILOT_GLOBAL_START + PILOT_CATALOG_SIZE - 1), PILOT_CATALOG_SIZE)

    def test_ids_outside_block_are_rejected(self) -> None:
        for global_id in (PILOT_GLOBAL_START - 1, PILOT_GLOBAL_START + PILOT_CATALOG_SIZE):
            with self.assertRaises(ValueError):
                local_item_id(global_id)


class PairedArmTests(unittest.TestCase):
    def test_mismatch_and_failed_captions_become_unavailable_without_dropping_rows(self) -> None:
        arms = build_paired_arm_records(_catalog({2: ("mismatch", "MISMATCH"), 3: ("image_failed", "")}))
        self.assertEqual(len(arms), PILOT_CATALOG_SIZE)
        self.assertEqual(arms[0]["arm_texts"]["real"], "Title: Game 1; Visual cues: color: c1")
        self.assertEqual(arms[1]["arm_texts"]["real"], "Title: Game 2; Visual cues: unavailable")
        self.assertIsNone(arms[2]["real_cue"])
        self.assertTrue(all(a["arm_texts"]["title-only"] == a["title"] for a in arms))

    def test_foreign_domain_or_reordered_catalog_fails_closed(self) -> None:
        foreign = _catalog()
        foreign[5]["domain"] = "Electronics"
        reordered = _catalog()
        reordered[0], reordered[1] = reordered[1], reordered[0]
        for records in (foreign, reordered, _catalog()[:-1]):
            with self.assertRaises(RuntimeError):
                build_paired_arm_records(records)


class ReviewGateTests(unittest.TestCase):
    def _write(self, root: Path, labels: list[str]) -> tuple[Path, Path]:
        (root / "images").mkdir()
        sheet = root / "sheet.csv"
        with sheet.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["n", "human_label"])
            writer.writeheader()
            for n, label in enumerate(labels, start=1):
                writer.writerow({"n": n, "human_label": label})
                (root / "images" / f"{n:03d}.jpg").write_bytes(b"x")
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "source_sheet_sha256_after_labels": hashlib.sha256(sheet.read_bytes()).hexdigest(),
        }), encoding="utf-8")
        return sheet, manifest

    def test_passing_sheet_reports_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sheet, manifest = self._write(Path(tmp), ["ADDS"] * 80 + ["OCR"] * 10 + ["SUSPECT"] * 10)
            self.assertEqual(validate_pilot_review(sheet, manifest)["label_counts"]["ADDS"], 80)

    def test_blank_label_or_threshold_breach_fails_closed(self) -> None:
        cases = (
            ["ADDS"] * 99 + [""],
            ["ADDS"] * 89 + ["SUSPECT"] * 11,
            ["ADDS"] * 79 + ["REDUNDANT"] * 21,
            ["ADDS"] * 39 + ["GENERIC"] * 61,
        )
        for labels in cases:
            with tempfile.TemporaryDirectory() as tmp, self.subTest(labels=labels[-1]):
                sheet, manifest = self._write(Path(tmp), labels)
                with self.assertRaises(RuntimeError):
                    validate_pilot_review(sheet, manifest)

    def test_sheet_edited_after_manifest_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sheet, manifest = self._write(Path(tmp), ["ADDS"] * 100)
            sheet.write_text(sheet.read_text(encoding="utf-8").replace("ADDS", "OCR", 1), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "does not pin"):
                validate_pilot_review(sheet, manifest)


if __name__ == "__main__":
    unittest.main()
