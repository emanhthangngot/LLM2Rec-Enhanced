"""Deterministic, standard-library-only corpus construction.

Authoritative protocol: plans/260915-0955-visual-delta-fusion-pilot/plan.md
and phase-01-local-module.md. This module never imports torch, transformers,
numpy, or PIL; caption/paraphrase generation backends live in caption.py and
are loaded lazily there. Everything here is pure functions over plain Python
data so it is testable without a GPU and reproducible on Kaggle.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence

UNAVAILABLE = "unavailable"
TITLE_PREFIX = "Title: "
CUE_SEPARATOR = "; Visual cues: "
ARMS: tuple[str, ...] = ("title-only", "null", "real", "shuffle", "paraphrase")
CUE_CAPS: tuple[int, ...] = (16, 32, 64)
PRIMARY_CAP = 32
BIN_SIZE = 64
MAX_CSFT_TOKENS = 1024

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace and strip the ends. Never touches wording."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def fuse_text(title: str, cue: str) -> str:
    """Build the fused item text ``Title: T; Visual cues: Z`` (plan.md formula).

    ``title`` is used verbatim (callers decide normalization); ``cue`` falls
    back to the literal ``unavailable`` marker when empty, never to an
    invented placeholder.
    """
    if not title:
        raise ValueError("fuse_text requires a non-empty title")
    resolved_cue = cue if cue else UNAVAILABLE
    return f"{TITLE_PREFIX}{title}{CUE_SEPARATOR}{resolved_cue}"


@dataclass(frozen=True)
class CatalogRow:
    """One catalog item with everything downstream stages need, in one place.

    ``order_index`` preserves the source's original row order (title-source
    order for a single domain, or occurrence order across the mixed corpus);
    string keys elsewhere never imply this ordering.
    """

    domain: str
    source_item_id: str
    downstream_item_id: int | None
    parent_asin: str | None
    source_title: str
    order_index: int
    image_status: str = "unknown"
    caption_status: str = "unknown"
    raw_caption: str | None = None


def build_catalog_rows(
    item_titles: Mapping[str, str],
    item_asin_map: Mapping[str, Mapping[str, object]] | None = None,
    domain: str = "",
) -> list[CatalogRow]:
    """Join a downstream ``item_titles`` map with an optional ASIN map.

    ``item_titles`` keys are string downstream IDs, contract-compatible with
    the committed ``item_titles.json`` (IDs ``1..N``, no gaps). Row order
    follows ascending numeric ID, matching every existing item_texts.txt
    convention in this repository. Every title row is kept even when no ASIN
    entry exists for it (recorded, not dropped) so coverage can be reported
    honestly rather than by silent omission.
    """
    if not item_titles:
        raise ValueError("item_titles must not be empty")
    try:
        ordered_ids = sorted(item_titles, key=int)
    except ValueError as exc:
        raise ValueError("item_titles keys must be numeric downstream IDs") from exc
    numeric_ids = [int(item_id) for item_id in ordered_ids]
    expected = list(range(1, max(numeric_ids) + 1))
    if numeric_ids != expected:
        raise ValueError(
            f"item_titles for domain {domain!r} must cover exactly 1..N with no gaps; "
            f"got {len(numeric_ids)} rows spanning {numeric_ids[0]}..{numeric_ids[-1]}"
        )
    asin_by_downstream: dict[str, Mapping[str, object]] = {}
    if item_asin_map is not None:
        items = item_asin_map.get("items", item_asin_map)
        if not isinstance(items, Mapping):
            raise ValueError("item_asin_map must contain an 'items' mapping or be one itself")
        title_ids = set(ordered_ids)
        for raw_id, record in items.items():
            downstream_id = str(int(raw_id))
            if downstream_id not in title_ids:
                raise ValueError(
                    f"item_asin_map contains downstream ID {downstream_id} absent from item_titles"
                )
            if downstream_id in asin_by_downstream:
                raise ValueError(f"duplicate downstream ID {downstream_id} in item_asin_map for {domain!r}")
            if not isinstance(record, Mapping):
                raise ValueError(f"item_asin_map record {downstream_id} is not an object")
            mapped_title = record.get("title")
            if mapped_title is not None and str(mapped_title) != str(item_titles[downstream_id]):
                raise ValueError(
                    f"title mismatch for downstream ID {downstream_id}: "
                    f"item_titles={item_titles[downstream_id]!r}, map={mapped_title!r}"
                )
            if not record.get("parent_asin"):
                raise ValueError(f"item_asin_map record {downstream_id} has no parent_asin")
            asin_by_downstream[downstream_id] = record
    rows: list[CatalogRow] = []
    for order_index, item_id in enumerate(ordered_ids):
        record = asin_by_downstream.get(item_id)
        rows.append(
            CatalogRow(
                domain=domain,
                source_item_id=item_id,
                downstream_item_id=int(item_id),
                parent_asin=str(record["parent_asin"]) if record else None,
                source_title=item_titles[item_id],
                order_index=order_index,
                image_status="unmapped" if record is None else "pending",
            )
        )
    return rows


def compute_training_frequencies(train_sequences: Iterable[Sequence[int]]) -> dict[int, int]:
    """Count item occurrences across training-only interaction sequences.

    Callers must pass training-split sequences only (plan.md: "Frequency and
    strata use training interactions only"); this function has no way to
    enforce that boundary itself and trusts the caller's split.
    """
    frequencies: dict[int, int] = {}
    for sequence in train_sequences:
        for item_id in sequence:
            frequencies[item_id] = frequencies.get(item_id, 0) + 1
    return frequencies


def frequency_bin_derangement(
    available_ids: Sequence[int],
    frequencies: Mapping[int, float],
    seed: int,
    bin_size: int = BIN_SIZE,
) -> dict[int, int]:
    """Return a donor map ``item_id -> donor_item_id`` with no fixed points.

    Mirrors the frequency-bin-then-roll-by-one convention already accepted in
    this repository (``baby_sealed_manifest.py:build_derangement_for_seed``),
    reimplemented with ``random.Random`` since this workstation and this
    corpus family carry no numpy dependency. Items outside ``available_ids``
    are never donors and never receive a mapping; callers treat them as
    ``unavailable`` in every arm that consults this map.
    """
    unique_ids = list(dict.fromkeys(available_ids))
    if len(unique_ids) < 2:
        raise RuntimeError(f"derangement for seed {seed} needs at least two available items")
    ordered = sorted(unique_ids, key=lambda item_id: (frequencies.get(item_id, 0.0), item_id))
    bins = [ordered[start : start + bin_size] for start in range(0, len(ordered), bin_size)]
    if len(bins) > 1 and len(bins[-1]) < 2:
        bins[-2] = bins[-2] + bins[-1]
        bins.pop()
    rng = random.Random(seed)
    mapping: dict[int, int] = {}
    for bin_ids in bins:
        shuffled = list(bin_ids)
        rng.shuffle(shuffled)
        rolled = shuffled[1:] + shuffled[:1]
        for original, donor in zip(shuffled, rolled):
            mapping[original] = donor
    fixed_points = [item_id for item_id in unique_ids if mapping[item_id] == item_id]
    if fixed_points:
        raise RuntimeError(f"derangement for seed {seed} has fixed points: {fixed_points}")
    if sorted(mapping.values()) != sorted(mapping.keys()):
        raise RuntimeError(f"derangement for seed {seed} is not a permutation")
    return mapping


class Tokenizer(Protocol):
    """Minimal tokenizer boundary so token budgeting is testable without a GPU."""

    def encode(self, text: str) -> list[int]: ...


def truncate_to_token_cap(text: str, cap: int, tokenizer: Tokenizer) -> tuple[str, int]:
    """Truncate ``text`` to at most ``cap`` tokens at a whole-word boundary.

    Binary search over word count, assuming token count is non-decreasing in
    word count for the supplied tokenizer (true for word-piece/BPE
    tokenizers on additive text). Returns ``("", 0)`` for empty input or when
    even the first word already exceeds ``cap`` tokens.
    """
    words = text.split()
    if not words or cap <= 0:
        return "", 0
    low, high = 0, len(words)
    best_text, best_tokens = "", 0
    while low <= high:
        mid = (low + high) // 2
        candidate = " ".join(words[:mid]) if mid else ""
        token_count = len(tokenizer.encode(candidate)) if candidate else 0
        if token_count <= cap:
            best_text, best_tokens = candidate, token_count
            low = mid + 1
        else:
            high = mid - 1
    return best_text, best_tokens


@dataclass(frozen=True)
class ArmTexts:
    """The five arm strings for a single catalog item, plus donor provenance."""

    downstream_item_id: int
    texts: dict[str, str]
    shuffle_donor_id: int | None


def build_arms(
    rows: Sequence[CatalogRow],
    real_cues: Mapping[int, str],
    paraphrase_cues: Mapping[int, str],
    shuffle_donor_map: Mapping[int, int],
) -> list[ArmTexts]:
    """Build the five arms with one shared cue-availability mask."""
    if set(real_cues) != set(paraphrase_cues):
        raise ValueError("real and paraphrase availability masks must be identical")
    available_ids = {item_id for item_id, cue in real_cues.items() if cue}
    if set(real_cues) != available_ids:
        raise ValueError("real_cues must omit unavailable items rather than contain empty cues")
    if set(shuffle_donor_map) != available_ids or set(shuffle_donor_map.values()) != available_ids:
        raise ValueError("shuffle donor map must be a permutation of exactly the available IDs")
    if any(item_id == donor_id for item_id, donor_id in shuffle_donor_map.items()):
        raise ValueError("shuffle donor map contains a fixed point")
    arms: list[ArmTexts] = []
    for row in rows:
        item_id = row.downstream_item_id
        if item_id is None:
            raise ValueError(f"row {row.source_item_id!r} has no downstream_item_id")
        donor_id = shuffle_donor_map.get(item_id)
        real_cue = real_cues.get(item_id)
        shuffle_cue = real_cues.get(donor_id) if donor_id is not None else None
        paraphrase_cue = paraphrase_cues.get(item_id)
        texts = {
            "title-only": row.source_title,
            "null": fuse_text(row.source_title, UNAVAILABLE),
            "real": fuse_text(row.source_title, real_cue or UNAVAILABLE),
            "shuffle": fuse_text(row.source_title, shuffle_cue or UNAVAILABLE),
            "paraphrase": fuse_text(row.source_title, paraphrase_cue or UNAVAILABLE),
        }
        arms.append(ArmTexts(item_id, texts, donor_id))
    _assert_arm_consistency(rows, arms)
    return arms


def _assert_arm_consistency(rows: Sequence[CatalogRow], arms: Sequence[ArmTexts]) -> None:
    if len(rows) != len(arms):
        raise RuntimeError("arm count must equal row count")
    for row, arm in zip(rows, arms):
        if row.downstream_item_id != arm.downstream_item_id:
            raise RuntimeError("arm ordering drifted from row ordering")
        for name in ARMS:
            if name not in arm.texts:
                raise RuntimeError(f"item {arm.downstream_item_id} is missing arm {name!r}")
        title_segment = f"{TITLE_PREFIX}{row.source_title}"
        for name in ("null", "real", "shuffle", "paraphrase"):
            if not arm.texts[name].startswith(title_segment):
                raise RuntimeError(
                    f"item {arm.downstream_item_id} arm {name!r} does not share the title segment"
                )


def compute_common_history_suffix(
    history_ids: Sequence[int],
    target_title: str,
    arm_text_by_id: Mapping[int, str],
    tokenizer: Tokenizer,
    special_token_overhead: int = 8,
    max_tokens: int = MAX_CSFT_TOKENS,
) -> tuple[list[int], int]:
    """Keep the longest whole-item suffix of the rendered CSFT sequence."""
    if max_tokens < 0 or special_token_overhead < 0:
        raise ValueError("token budget and overhead must be non-negative")
    if len(tokenizer.encode(target_title)) + special_token_overhead > max_tokens:
        raise RuntimeError("target title plus special tokens already exceed max_tokens")
    for item_id in history_ids:
        if item_id not in arm_text_by_id:
            raise KeyError(f"history item {item_id} has no arm text")
    for start in range(len(history_ids) + 1):
        retained = list(history_ids[start:])
        rendered = " ".join([*(arm_text_by_id[item_id] for item_id in retained), target_title])
        used = len(tokenizer.encode(rendered)) + special_token_overhead
        if used <= max_tokens:
            return retained, used
    raise AssertionError("empty history must fit after target preflight")


def compute_shared_history_suffix(
    history_ids: Sequence[int],
    target_title: str,
    arm_text_by_id_variants: Mapping[str, Mapping[int, str]],
    tokenizer: Tokenizer,
    special_token_overhead: int = 8,
    max_tokens: int = MAX_CSFT_TOKENS,
) -> tuple[list[int], dict[str, int]]:
    """Choose one suffix valid for every registered arm/cap variant."""
    if not arm_text_by_id_variants:
        raise ValueError("at least one arm variant is required")
    candidates = [
        compute_common_history_suffix(
            history_ids, target_title, texts, tokenizer, special_token_overhead, max_tokens
        )[0]
        for texts in arm_text_by_id_variants.values()
    ]
    keep = min(map(len, candidates))
    suffix = list(history_ids[-keep:]) if keep else []
    budgets: dict[str, int] = {}
    for name, texts in arm_text_by_id_variants.items():
        rendered = " ".join([*(texts[item_id] for item_id in suffix), target_title])
        budgets[name] = len(tokenizer.encode(rendered)) + special_token_overhead
        if budgets[name] > max_tokens:
            raise RuntimeError(f"shared suffix exceeds budget for variant {name!r}")
    return suffix, budgets


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
def atomic_write_text(path: Path, content: str) -> str:
    """Write UTF-8 content atomically and return its digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = content.encode("utf-8")
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(payload)
    os.replace(tmp_path, path)
    return sha256_bytes(payload)

def write_arm_corpora(arms: Sequence[ArmTexts], out_dir: Path) -> dict[str, dict[str, str]]:
    """Write ``item_texts.{json,txt}`` per arm, atomically, plus a hash ledger.

    ``item_texts.json`` uses string keys (matching every existing
    ``item_titles.json``-shaped file in this repository); ``item_texts.txt``
    has one line per item in ascending downstream-ID order, mirroring
    ``info/item_titles.txt``. A source title or cue containing a literal
    newline would otherwise silently split the TXT corpus, so any such text
    is rejected here rather than laundered.
    """
    if not arms:
        raise ValueError("write_arm_corpora requires at least one item")
    ordered = sorted(arms, key=lambda arm: arm.downstream_item_id)
    hashes: dict[str, dict[str, str]] = {}
    for arm_name in ARMS:
        json_map: dict[str, str] = {}
        lines: list[str] = []
        for arm in ordered:
            text = arm.texts[arm_name]
            if "\n" in text or "\r" in text:
                raise ValueError(
                    f"item {arm.downstream_item_id} arm {arm_name!r} contains a newline; "
                    "the flat .txt corpus cannot represent it without splitting rows"
                )
            json_map[str(arm.downstream_item_id)] = text
            lines.append(text)
        arm_dir = out_dir / arm_name
        json_hash = atomic_write_text(arm_dir / "item_texts.json", json.dumps(json_map, sort_keys=True))
        txt_hash = atomic_write_text(arm_dir / "item_texts.txt", "\n".join(lines) + "\n")
        hashes[arm_name] = {"item_texts.json": json_hash, "item_texts.txt": txt_hash}
    return hashes


SHARD_SIZE = 512


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def shard_records(records: Sequence[Mapping[str, object]], shard_size: int = SHARD_SIZE) -> list[list[Mapping[str, object]]]:
    """Partition records without reordering or dropping rows."""
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    return [list(records[start : start + shard_size]) for start in range(0, len(records), shard_size)]


def write_sharded_records(
    records: Sequence[Mapping[str, object]],
    out_dir: Path,
    identity: Mapping[str, object],
    shard_size: int = SHARD_SIZE,
) -> dict[str, object]:
    """Write completed JSONL shards and an atomic, hash-pinned manifest.

    A shard is visible in the manifest only after its temporary JSONL file has
    been atomically renamed and its digest computed. Existing output is never
    reused implicitly: its identity must exactly equal ``identity``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "shard-manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("identity") != dict(identity):
            raise RuntimeError("existing shard manifest identity mismatch; refusing resume")
    shards: list[dict[str, object]] = []
    for index, chunk in enumerate(shard_records(records, shard_size)):
        payload = "".join(_canonical_json(record) + "\n" for record in chunk).encode("utf-8")
        shard_path = out_dir / f"shard-{index:05d}.jsonl"
        temp_path = shard_path.with_suffix(".jsonl.partial")
        temp_path.write_bytes(payload)
        os.replace(temp_path, shard_path)
        shards.append({
            "index": index,
            "path": shard_path.name,
            "records": len(chunk),
            "sha256": sha256_bytes(payload),
            "completion_status": "complete",
        })
    manifest = {
        "identity": dict(identity),
        "shard_size": shard_size,
        "record_count": len(records),
        "shard_count": len(shards),
        "shards": shards,
        "completion_status": "complete",
    }
    atomic_write_text(manifest_path, _canonical_json(manifest) + "\n")
    return manifest


def validate_shard_manifest(manifest_path: Path, expected_identity: Mapping[str, object]) -> dict[str, object]:
    """Validate every shard's identity, completion marker and SHA256."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("identity") != dict(expected_identity):
        raise RuntimeError("shard manifest identity mismatch")
    if manifest.get("completion_status") != "complete":
        raise RuntimeError("shard manifest is not complete")
    if manifest.get("shard_count") != len(manifest.get("shards", [])):
        raise RuntimeError("shard manifest count mismatch")
    for shard in manifest["shards"]:
        if shard.get("completion_status") != "complete":
            raise RuntimeError(f"shard {shard.get('index')} is incomplete")
        path = manifest_path.parent / str(shard["path"])
        if not path.is_file():
            raise RuntimeError(f"shard file missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != shard.get("sha256"):
            raise RuntimeError(f"shard hash mismatch: {path}")
    return manifest


def load_shard_records(
    out_dir: Path, expected_identity: Mapping[str, object] | None = None
) -> list[dict[str, object]]:
    """Read a validated shard set back into one flat, catalog-ordered list.

    Complements `write_sharded_records`: shards are always written in
    catalog order and never reordered, so concatenating each shard's JSONL
    lines in the manifest's own shard order reconstructs the original record
    order exactly. Used to resume an interrupted generation run from its
    last checkpoint instead of restarting from item 0.

    A real cross-push resume hit `JSONDecodeError` reproducibly on a shard
    whose bytes matched the manifest hash and parsed as 512 valid physical
    JSONL lines. The shard contains U+0085 (Unicode NEL) inside one JSON
    string. `str.splitlines()` treats NEL as a line boundary, so decoding the
    whole payload before splitting manufactured a 513th line and truncated
    that record even though the file was valid.

    JSONL is byte-delimited by LF because `write_sharded_records` emits
    exactly `b"\\n"` between records. Split the validated payload on that
    delimiter before decoding each record; Unicode line-separator characters
    inside JSON strings then remain data. Hashing and parsing still share the
    same single byte read.
    """
    manifest_path = out_dir / "shard-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected_identity is not None:
        if manifest.get("identity") != dict(expected_identity):
            raise RuntimeError("shard manifest identity mismatch")
        if manifest.get("completion_status") != "complete":
            raise RuntimeError("shard manifest is not complete")
        if manifest.get("shard_count") != len(manifest.get("shards", [])):
            raise RuntimeError("shard manifest count mismatch")
    records: list[dict[str, object]] = []
    for shard in manifest["shards"]:
        shard_path = out_dir / str(shard["path"])
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                payload = shard_path.read_bytes()
                if expected_identity is not None:
                    if shard.get("completion_status") != "complete":
                        raise RuntimeError(f"shard {shard.get('index')} is incomplete")
                    digest = sha256_bytes(payload)
                    if digest != shard.get("sha256"):
                        raise RuntimeError(f"shard hash mismatch: {shard_path}")
                shard_records = [
                    json.loads(line.decode("utf-8")) for line in payload.split(b"\n") if line
                ]
                break
            except (json.JSONDecodeError, RuntimeError, UnicodeDecodeError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
        else:
            raise RuntimeError(
                f"shard file {shard_path} failed to read/verify/parse after 3 attempts: "
                f"{last_error}"
            ) from last_error
        records.extend(shard_records)
    return records
