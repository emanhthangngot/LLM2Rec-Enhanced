"""Video_Games-only paired pilot: review gate, ID mapping, and two-arm texts.

Standard-library only so the gate and arm construction run locally and are
inlined verbatim into the Kaggle notebook.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from corpus import UNAVAILABLE, fuse_text
from crosswalk import catalog_blocks

PILOT_DOMAIN = "Video_Games"
PAIRED_ARMS: tuple[str, ...] = ("title-only", "real")
REVIEW_LABELS = frozenset({"ADDS", "GENERIC", "MISMATCH_OK", "OCR", "REDUNDANT", "SUSPECT"})
REVIEW_ROW_COUNT = 100
ADDS_MINIMUM = 40
OCR_OR_REDUNDANT_MAXIMUM = 20
SUSPECT_MAXIMUM = 10

_PILOT_BLOCK = next(block for block in catalog_blocks() if block.domain == PILOT_DOMAIN)
PILOT_GLOBAL_START = _PILOT_BLOCK.start
PILOT_CATALOG_SIZE = _PILOT_BLOCK.size


def local_item_id(global_id: int) -> int:
    """Map an AmazonMix-6 global ID to the Games_5core downstream ID (1-based).

    Verified on 114 pilot/smoke titles: local = global - 66082 + 1.
    """
    if not PILOT_GLOBAL_START <= global_id < PILOT_GLOBAL_START + PILOT_CATALOG_SIZE:
        raise ValueError(f"global ID {global_id} is outside the {PILOT_DOMAIN} block")
    return global_id - PILOT_GLOBAL_START + 1


def validate_pilot_review(sheet_path: Path, manifest_path: Path) -> dict[str, object]:
    """Fail closed unless all 100 review labels exist and pass the pilot gates."""
    with sheet_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    numbers = [int(row["n"]) for row in rows]
    if len(rows) != REVIEW_ROW_COUNT or sorted(numbers) != list(range(1, REVIEW_ROW_COUNT + 1)):
        raise RuntimeError(f"review sheet must contain rows n=1..{REVIEW_ROW_COUNT} exactly once")
    blank = [row["n"] for row in rows if not row.get("human_label", "").strip()]
    if blank:
        raise RuntimeError(f"review labels missing for rows {blank}")
    invalid = sorted({row["human_label"] for row in rows} - REVIEW_LABELS)
    if invalid:
        raise RuntimeError(f"invalid review labels: {invalid}")
    images = sheet_path.parent / "images"
    missing_images = [n for n in numbers if not (images / f"{n:03d}.jpg").is_file()]
    if missing_images:
        raise RuntimeError(f"review images missing for rows {missing_images}")
    counts = Counter(row["human_label"] for row in rows)
    failures = []
    if counts["ADDS"] < ADDS_MINIMUM:
        failures.append(f"ADDS {counts['ADDS']} < {ADDS_MINIMUM}")
    if counts["OCR"] + counts["REDUNDANT"] > OCR_OR_REDUNDANT_MAXIMUM:
        failures.append(f"OCR+REDUNDANT {counts['OCR'] + counts['REDUNDANT']} > {OCR_OR_REDUNDANT_MAXIMUM}")
    if counts["SUSPECT"] > SUSPECT_MAXIMUM:
        failures.append(f"SUSPECT {counts['SUSPECT']} > {SUSPECT_MAXIMUM}")
    if failures:
        raise RuntimeError("pilot quality gate failed: " + "; ".join(failures))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sheet_sha256 = hashlib.sha256(sheet_path.read_bytes()).hexdigest()
    if manifest.get("source_sheet_sha256_after_labels") != sheet_sha256:
        raise RuntimeError("review manifest does not pin the current review sheet")
    return {
        "sheet_sha256": sheet_sha256,
        "reviewer": manifest.get("reviewer"),
        "label_counts": dict(sorted(counts.items())),
    }


def usable_real_cue(record: Mapping[str, object]) -> str | None:
    """Return the raw caption only for successful captions; mismatches stay unavailable."""
    if record.get("caption_status") != "ok":
        return None
    cue = str(record.get("caption_raw") or "").strip()
    return cue or None


def build_paired_arm_records(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Build title-only/real texts for the complete Video_Games catalog.

    Every catalog item is kept; items without a usable caption get the
    explicit ``unavailable`` cue in the real arm, never a dropped row.
    """
    if len(records) != PILOT_CATALOG_SIZE:
        raise RuntimeError(f"expected {PILOT_CATALOG_SIZE} {PILOT_DOMAIN} records, got {len(records)}")
    output: list[dict[str, object]] = []
    for expected_local, record in enumerate(records, start=1):
        if record.get("domain") != PILOT_DOMAIN:
            raise RuntimeError(f"non-{PILOT_DOMAIN} record in paired corpus: {record.get('global_id')}")
        local_id = local_item_id(int(record["global_id"]))
        if local_id != expected_local:
            raise RuntimeError(f"records are not in catalog order at local ID {expected_local}")
        title = str(record.get("title") or "")
        if not title:
            raise RuntimeError(f"empty title for local ID {local_id}")
        cue = usable_real_cue(record)
        output.append({
            "local_item_id": local_id,
            "global_id": int(record["global_id"]),
            "asin": record.get("asin"),
            "title": title,
            "image_status": record.get("image_status"),
            "caption_status": record.get("caption_status"),
            "caption_raw": record.get("caption_raw"),
            "real_cue": cue,
            "arm_texts": {"title-only": title, "real": fuse_text(title, cue or UNAVAILABLE)},
        })
    return output
