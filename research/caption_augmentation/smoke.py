"""Qwen3-VL generation smoke and checkpointed full-corpus caption pipeline.

Qwen3-VL structured captions and Qwen2.5 title-only paraphrases run only on
Kaggle; model dependencies stay lazy in caption.py.
"""
from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import io
import json
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Callable, Sequence

from caption import (
    CAPTION_MAX_NEW_TOKENS,
    CAPTION_BATCH_SIZE,
    PARAPHRASE_MAX_NEW_TOKENS,
    PARAPHRASE_MODEL_ID,
    PARAPHRASE_MODEL_REVISION,
    Captioner,
    Paraphraser,
    Qwen3VLCaptioner,
    QwenParaphraser,
    file_sha256,
    prompt_sha256,
    text_sha256,
)
from corpus import (
    CatalogRow,
    build_arms,
    compute_training_frequencies,
    frequency_bin_derangement,
    load_shard_records,
    matched_cue_maps,
    write_sharded_records,
)
from crosswalk import CATALOG_SIZE, catalog_blocks, read_item_asin_pairs
USER_AGENT = "llm2rec-caption-augmentation-smoke/1.0"
SAMPLE_TOTAL = 64
IMAGE_MAX_BYTES = 8 * 1024 * 1024
METADATA_URL_TEMPLATE = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
    "meta_categories/meta_{category}.jsonl.gz"
)


def _evenly_spaced_positions(start: int, size: int, count: int) -> list[int]:
    if count <= 0:
        return []
    if count == 1:
        return [start + size // 2]
    return [start + round(index * (size - 1) / (count - 1)) for index in range(count)]


def select_smoke_sample(total: int = SAMPLE_TOTAL) -> list[tuple[str, int]]:
    """Deterministic sample of `total` global IDs spread across all six domains.

    Evenly spaced positions within each domain's block, never "first IDs"
    (phase-02-kaggle-package.md Step 4). Earlier domains in catalog order
    absorb the remainder when `total` does not divide evenly by six.
    """
    blocks = catalog_blocks()
    domain_count = len(blocks)
    base, remainder = divmod(total, domain_count)
    counts = [base + (1 if index < remainder else 0) for index in range(domain_count)]
    sample: list[tuple[str, int]] = []
    for block, count in zip(blocks, counts):
        for global_id in _evenly_spaced_positions(block.start, block.size, count):
            sample.append((block.domain, global_id))
    return sample


def select_video_games_smoke_sample(total: int = SAMPLE_TOTAL) -> list[tuple[str, int]]:
    """Choose deterministic, evenly spaced items from the Video_Games block only."""
    block = next(block for block in catalog_blocks() if block.domain == "Video_Games")
    if total <= 0 or total > block.size:
        raise ValueError(f"total must be between 1 and {block.size}, got {total}")
    return [
        (block.domain, global_id)
        for global_id in _evenly_spaced_positions(block.start, block.size, total)
    ]


def select_quality_review_sample(per_domain: int = 50) -> list[tuple[str, int]]:
    """Deterministic sample of `per_domain` global IDs per domain, evenly
    spaced (never "first IDs"), selected before generation
    (phase-01-local-module.md Step 6: "Deterministic sample of 50 items per
    pretraining domain, selected before generation").
    """
    sample: list[tuple[str, int]] = []
    for block in catalog_blocks():
        for global_id in _evenly_spaced_positions(block.start, block.size, per_domain):
            sample.append((block.domain, global_id))
    return sample


def locate_mixed_root(search_root: Path = Path("/kaggle/input")) -> Path:
    candidates = sorted(search_root.glob("**/AmazonMix-6/5-core"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one mounted AmazonMix-6/5-core directory, found {candidates}"
        )
    return candidates[0]


def load_asin_titles(mixed_root: Path, wanted_ids: set[int]) -> dict[int, tuple[str, str]]:
    """Return `{global_id: (asin, title)}` for exactly the wanted global IDs.

    A wanted ID absent from every split's ASIN column is omitted here (the
    catalog's own 20 ASIN-less positions, see
    plans/260915-0955-visual-delta-fusion-pilot/reports/crosswalk-verification.md);
    callers must record it as `no_asin_in_catalog`, never drop it silently.
    """
    pairs, _blank = read_item_asin_pairs(list(mixed_root.glob("*/*.csv")))
    titles_path = mixed_root / "info" / "item_titles.txt"
    lines = titles_path.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    result: dict[int, tuple[str, str]] = {}
    for global_id in wanted_ids:
        asin = pairs.get(global_id)
        if asin is None:
            continue
        result[global_id] = (asin, lines[global_id])
    return result


def choose_image_url(record: dict) -> str | None:
    images = record.get("images") or []
    for image in images:
        if not isinstance(image, dict):
            continue
        for key in ("hi_res", "large", "thumb"):
            value = image.get(key)
            if isinstance(value, str) and value.startswith("http"):
                return value
    return None


def fetch_domain_image_urls(category: str, wanted_asins: set[str], cache_dir: Path) -> dict[str, str]:
    """Stream `meta_<category>.jsonl.gz`, returning `{asin: image_url}` found.

    Stops early once every wanted ASIN in this category is found; the
    metadata file order is not guaranteed, so a worst-case target near the
    file's end still requires a full stream.
    """
    if not wanted_asins:
        return {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / f"meta_{category}.jsonl.gz"
    if not archive_path.is_file() or archive_path.stat().st_size == 0:
        url = METADATA_URL_TEMPLATE.format(category=category)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        temp_path = archive_path.with_suffix(".jsonl.gz.partial")
        with urllib.request.urlopen(request, timeout=180) as response, temp_path.open("wb") as output:
            shutil.copyfileobj(response, output)
        temp_path.replace(archive_path)
    remaining = set(wanted_asins)
    found: dict[str, str] = {}
    with gzip.open(archive_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not remaining:
                break
            record = json.loads(line)
            parent_asin = record.get("parent_asin")
            if parent_asin in remaining:
                url = choose_image_url(record)
                if url:
                    found[parent_asin] = url
                remaining.discard(parent_asin)
    return found


def download_image(url: str, dest: Path) -> dict[str, object]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: str | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read(IMAGE_MAX_BYTES + 1)
            if len(payload) > IMAGE_MAX_BYTES:
                return {"status": "download_failed", "error": "exceeds_size_limit"}
            from PIL import Image

            with Image.open(io.BytesIO(payload)) as image:
                image.convert("RGB")
            dest.write_bytes(payload)
            return {
                "status": "decoded",
                "path": str(dest),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        except Exception as exc:  # noqa: BLE001 - preserve per-image failure evidence
            last_error = f"{type(exc).__name__}: {exc}"
    return {"status": "download_failed", "error": last_error}


def run_generation_pass(
    sample: list[tuple[str, int]],
    work_dir: Path,
    mixed_root: Path,
    stage: str,
    output_filename: str,
    temp_dir: Path = Path("/kaggle/temp"),
) -> dict[str, object]:
    """Shared real-generation core for both the 64-item smoke and the
    Phase 1 Step 6 quality-review sample (50 items/domain). Identical logic,
    different sample size and output name; nothing about caption/paraphrase
    generation differs between the two.
    """
    t_start = time.perf_counter()
    wanted_ids = {global_id for _domain, global_id in sample}
    asin_titles = load_asin_titles(mixed_root, wanted_ids)

    records: list[dict[str, object]] = []
    by_domain: dict[str, list[tuple[int, str, str]]] = {}
    for domain, global_id in sample:
        entry = asin_titles.get(global_id)
        if entry is None:
            records.append({
                "domain": domain, "global_id": global_id, "asin": None, "title": None,
                "image_url": None, "image_status": "no_asin_in_catalog",
            })
            continue
        asin, title = entry
        by_domain.setdefault(domain, []).append((global_id, asin, title))

    # Metadata and images are scratch data: written under /kaggle/temp, never
    # under work_dir (/kaggle/working), so they are never bundled into the
    # kernel's persisted output. Each domain's multi-GB meta_*.jsonl.gz is
    # deleted immediately after its URLs are extracted, not accumulated
    # across all six domains, to bound peak disk usage.
    cache_dir = temp_dir / f"{stage}_metadata_cache"
    image_dir = temp_dir / f"{stage}_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    t_metadata_start = time.perf_counter()
    for domain, entries in by_domain.items():
        wanted_asins = {asin for _gid, asin, _title in entries}
        url_map = fetch_domain_image_urls(domain, wanted_asins, cache_dir)
        archive_path = cache_dir / f"meta_{domain}.jsonl.gz"
        archive_path.unlink(missing_ok=True)
        for global_id, asin, title in entries:
            url = url_map.get(asin)
            records.append({
                "domain": domain, "global_id": global_id, "asin": asin, "title": title,
                "image_url": url,
                "image_status": "missing_metadata" if url is None else "pending",
            })
    metadata_seconds = time.perf_counter() - t_metadata_start

    t_download_start = time.perf_counter()
    for record in records:
        if record.get("image_status") != "pending":
            continue
        dest = image_dir / f"{record['global_id']}.bin"
        outcome = download_image(str(record["image_url"]), dest)
        record["image_status"] = outcome["status"]
        if outcome["status"] == "decoded":
            record["image_path"] = outcome["path"]
            record["image_bytes"] = outcome["bytes"]
            record["image_sha256"] = outcome["sha256"]
        else:
            record["image_error"] = outcome.get("error")
    download_seconds = time.perf_counter() - t_download_start

    caption_items = [
        (record["global_id"], str(record["title"]), Path(str(record["image_path"])))
        for record in records if record.get("image_status") == "decoded"
    ]

    t_caption_load_start = time.perf_counter()
    captioner = Qwen3VLCaptioner()
    caption_model_load_seconds = time.perf_counter() - t_caption_load_start

    t_caption_start = time.perf_counter()
    caption_results = captioner.caption_batch(caption_items)
    caption_seconds = time.perf_counter() - t_caption_start

    caption_by_id = {result.item_id: result for result in caption_results}
    for record in records:
        result = caption_by_id.get(record["global_id"])
        if result is None:
            continue
        record["caption_status"] = result.status
        record["caption_raw"] = result.raw_text
        record["caption_generated_tokens"] = result.generated_token_count
        record["caption_hit_token_cap"] = result.generated_token_count == CAPTION_MAX_NEW_TOKENS
        record["caption_title_sha256"] = text_sha256(str(record["title"]))
        record["caption_image_sha256"] = file_sha256(Path(str(record["image_path"])))
    paraphrase_items = [
        (record["global_id"], str(record["title"]))
        for record in records if record.get("title")
    ]
    t_paraphrase_load_start = time.perf_counter()
    paraphraser = QwenParaphraser()
    paraphrase_model_load_seconds = time.perf_counter() - t_paraphrase_load_start
    t_paraphrase_start = time.perf_counter()
    paraphrase_results = paraphraser.paraphrase_batch(paraphrase_items)
    paraphrase_seconds = time.perf_counter() - t_paraphrase_start
    paraphrase_by_id = {result.item_id: result for result in paraphrase_results}
    for record in records:
        result = paraphrase_by_id.get(record["global_id"])
        if result is None:
            continue
        record["paraphrase_status"] = result.status
        record["paraphrase_text"] = result.raw_text
        record["paraphrase_generated_tokens"] = result.generated_token_count
        record["paraphrase_hit_token_cap"] = result.generated_token_count == PARAPHRASE_MAX_NEW_TOKENS

    decoded_count = sum(1 for record in records if record.get("image_status") == "decoded")
    captioned_ok = sum(1 for record in records if record.get("caption_status") == "ok")
    caption_mismatch = sum(1 for record in records if record.get("caption_status") == "mismatch")
    caption_terminal = captioned_ok + caption_mismatch
    paraphrased_ok = sum(1 for record in records if record.get("paraphrase_status") == "ok")
    caption_cap_hits = sum(1 for record in records if record.get("caption_hit_token_cap"))
    paraphrase_cap_hits = sum(1 for record in records if record.get("paraphrase_hit_token_cap"))

    result: dict[str, object] = {
        "status": "PASS" if decoded_count > 0 and caption_terminal > 0 and paraphrased_ok > 0 else "FAIL",
        "stage": stage,
        "sample_size": len(sample),
        "domains_sampled": sorted(by_domain),
        "image_coverage": decoded_count / len(sample),
        "caption_ok_rate": captioned_ok / max(decoded_count, 1),
        "caption_mismatch_count": caption_mismatch,
        "caption_terminal_rate": caption_terminal / max(decoded_count, 1),
        "caption_token_cap_hit_rate": caption_cap_hits / max(decoded_count, 1),
        "paraphrase_ok_rate": paraphrased_ok / max(len(paraphrase_items), 1),
        "paraphrase_token_cap_hit_rate": paraphrase_cap_hits / max(len(paraphrase_items), 1),
        "caption_model_id": captioner.model_id,
        "caption_model_revision": captioner.model_revision,
        "processor_revision": captioner.processor_revision,
        "caption_prompt_sha256": prompt_sha256(),
        "caption_transformers_version": __import__("transformers").__version__,
        "timing_seconds": {
            "metadata_stream": metadata_seconds,
            "image_download": download_seconds,
            "caption_model_load": caption_model_load_seconds,
            "caption_generation_total": caption_seconds,
            "caption_generation_per_image": caption_seconds / max(decoded_count, 1),
            "paraphrase_model_load": paraphrase_model_load_seconds,
            "paraphrase_generation_total": paraphrase_seconds,
            "paraphrase_generation_per_item": paraphrase_seconds / max(len(paraphrase_items), 1),
            "total": time.perf_counter() - t_start,
        },
        "records": records,
    }
    (work_dir / output_filename).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    summary = {key: value for key, value in result.items() if key != "records"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return result


def run_real_generator_smoke(
    work_dir: Path, mixed_root: Path, temp_dir: Path = Path("/kaggle/temp")
) -> dict[str, object]:
    return run_generation_pass(
        select_smoke_sample(), work_dir, mixed_root, "real_generator_smoke",
        "smoke_results.json", temp_dir,
    )


def run_quality_review(
    work_dir: Path, mixed_root: Path, temp_dir: Path = Path("/kaggle/temp"), per_domain: int = 50
) -> dict[str, object]:
    """Phase 1 Step 6: 50-items-per-domain deterministic sample, real
    captions and paraphrases, written for human review. This function
    generates the review material; the unsupported-attribute/redundant/
    adds-visible-attribute/unclear labels remain human judgments per
    phase-01-local-module.md ("These are human-reviewed judgments, not
    automatic semantic truth").
    """
    return run_generation_pass(
        select_quality_review_sample(per_domain), work_dir, mixed_root, "quality_review",
        "quality_review_results.json", temp_dir,
    )


def run_paraphrase_probe(
    work_dir: Path,
    mixed_root: Path,
    model_id: str = PARAPHRASE_MODEL_ID,
    revision: str | None = PARAPHRASE_MODEL_REVISION,
    per_domain: int = 50,
    output_filename: str = "paraphrase_probe_results.json",
) -> dict[str, object]:
    """Paraphraser-only comparison probe: the same 300-item deterministic
    quality-review sample and titles as `run_quality_review`, but skips
    metadata streaming, image download, and Florence-2 captioning entirely
    (paraphrasing is title-only and does not depend on any of them). Used to
    A/B a candidate paraphraser model against the pinned default without
    paying the ~470s metadata + ~80s image + ~165s caption cost every time.

    See plans/260915-0955-visual-delta-fusion-pilot/ISSUES.md #15 for why
    this comparison is needed: the pinned Qwen2.5-0.5B-Instruct paraphraser
    hallucinates specific unsupported facts on ~18% of a real 300-item
    sample (worst domain ~56%), far outside the protocol's proposed ≤5%
    gate.
    """
    t_start = time.perf_counter()
    sample = select_quality_review_sample(per_domain)
    wanted_ids = {global_id for _domain, global_id in sample}
    asin_titles = load_asin_titles(mixed_root, wanted_ids)

    records: list[dict[str, object]] = []
    for domain, global_id in sample:
        entry = asin_titles.get(global_id)
        if entry is None:
            records.append({
                "domain": domain, "global_id": global_id, "title": None,
                "paraphrase_status": "no_asin_in_catalog",
            })
            continue
        _asin, title = entry
        records.append({"domain": domain, "global_id": global_id, "title": title})

    paraphrase_items = [
        (record["global_id"], str(record["title"]))
        for record in records if record.get("title")
    ]
    t_load_start = time.perf_counter()
    paraphraser = QwenParaphraser(model_id=model_id, revision=revision)
    model_load_seconds = time.perf_counter() - t_load_start

    t_gen_start = time.perf_counter()
    paraphrase_results = paraphraser.paraphrase_batch(paraphrase_items)
    generation_seconds = time.perf_counter() - t_gen_start

    paraphrase_by_id = {result.item_id: result for result in paraphrase_results}
    for record in records:
        result = paraphrase_by_id.get(record["global_id"])
        if result is None:
            continue
        record["paraphrase_status"] = result.status
        record["paraphrase_text"] = result.raw_text
        record["paraphrase_generated_tokens"] = result.generated_token_count
        record["paraphrase_hit_token_cap"] = result.generated_token_count == PARAPHRASE_MAX_NEW_TOKENS

    paraphrased_ok = sum(1 for record in records if record.get("paraphrase_status") == "ok")
    cap_hits = sum(1 for record in records if record.get("paraphrase_hit_token_cap"))

    result: dict[str, object] = {
        "status": "PASS" if paraphrased_ok > 0 else "FAIL",
        "stage": "paraphrase_probe",
        "model_id": model_id,
        "resolved_revision": paraphraser.resolved_revision,
        "sample_size": len(sample),
        "paraphrase_ok_rate": paraphrased_ok / max(len(paraphrase_items), 1),
        "paraphrase_token_cap_hit_rate": cap_hits / max(len(paraphrase_items), 1),
        "timing_seconds": {
            "model_load": model_load_seconds,
            "generation_total": generation_seconds,
            "generation_per_item": generation_seconds / max(len(paraphrase_items), 1),
            "total": time.perf_counter() - t_start,
        },
        "records": records,
    }
    (work_dir / output_filename).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    summary = {key: value for key, value in result.items() if key != "records"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return result


def input_catalog_fingerprint(records: Sequence[dict[str, object]]) -> str:
    """Hash stable catalog identity while allowing absent ASIN/image metadata."""
    payload = [
        (record["global_id"], record["domain"], record.get("asin"),
         record["title"], record.get("image_url"))
        for record in records
    ]
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def resolve_full_catalog_metadata(
    mixed_root: Path,
    temp_dir: Path = Path("/kaggle/temp"),
    progress: Callable[[str], None] = print,
    domains: Sequence[str] | None = None,
) -> tuple[list[str], list[dict[str, object]]]:
    """Resolve every catalog item's ASIN/title/image URL, one domain at a
    time, with real per-domain progress. Returns the full `item_titles.txt`
    list and a per-item metadata record (`global_id`, `domain`, `title`,
    `asin`, `image_url`, `image_status`) in catalog order, ready for
    `process_catalog_records_incrementally`. `domains`, when given, restricts
    the returned records to those catalog blocks (still in catalog order) and
    skips metadata streaming for every other domain.

    Split out from the old monolithic `run_full_corpus_generation` (see
    plans/260915-0955-visual-delta-fusion-pilot/ISSUES.md #23) so metadata
    resolution prints visible progress instead of running silently for
    minutes with no indication of where it is.
    """
    titles_path = mixed_root / "info" / "item_titles.txt"
    titles = titles_path.read_text(encoding="utf-8").split("\n")
    if titles and titles[-1] == "":
        titles.pop()
    if len(titles) != CATALOG_SIZE:
        raise RuntimeError(f"catalog title count {len(titles)} != {CATALOG_SIZE}")
    blocks = catalog_blocks()
    if domains is not None:
        unknown = set(domains) - {block.domain for block in blocks}
        if unknown or not domains:
            raise ValueError(f"unknown or empty domain selection: {sorted(unknown) or domains}")
        blocks = tuple(block for block in blocks if block.domain in set(domains))
    wanted_ids = {item_id for block in blocks for item_id in range(block.start, block.stop)}
    asin_titles = load_asin_titles(mixed_root, wanted_ids)
    metadata_dir = temp_dir / "full_corpus_metadata"
    records: list[dict[str, object]] = []
    for block in blocks:
        t0 = time.perf_counter()
        domain_ids = range(block.start, block.stop)
        asins = {asin_titles[item_id][0] for item_id in domain_ids if item_id in asin_titles}
        url_by_asin = fetch_domain_image_urls(block.domain, asins, metadata_dir)
        (metadata_dir / f"meta_{block.domain}.jsonl.gz").unlink(missing_ok=True)
        resolved = 0
        for global_id in domain_ids:
            entry = asin_titles.get(global_id)
            record: dict[str, object] = {
                "global_id": global_id, "domain": block.domain, "title": titles[global_id],
                "asin": None, "image_url": None,
            }
            if entry is None:
                record["image_status"] = "no_asin_in_catalog"
            else:
                asin, _title = entry
                record["asin"] = asin
                image_url = url_by_asin.get(asin)
                if image_url is None:
                    record["image_status"] = "missing_metadata"
                else:
                    record["image_url"] = image_url
                    record["image_status"] = "pending"
                    resolved += 1
            records.append(record)
        progress(
            f"metadata: {block.domain:26s} {resolved}/{block.size} image URLs resolved "
            f"in {time.perf_counter() - t0:.1f}s"
        )
    return titles, records


def process_catalog_records_incrementally(
    metadata_records: list[dict[str, object]],
    captioner: Captioner,
    paraphraser: Paraphraser | None,
    out_dir: Path,
    identity: dict[str, object],
    image_dir: Path,
    resolve_image: Callable[[dict[str, object], Path], dict[str, object]],
    shard_size: int = 512,
    progress: Callable[[str], None] = print,
    deadline: float | None = None,
    batch_size: int = 8,
) -> list[dict[str, object]]:
    """Caption catalog records and optionally paraphrase in a later GPU pass.

    Shard checkpoints are shared by both passes, allowing the frozen captioner
    and text paraphraser to be released between phases. `deadline` is checked
    at chunk boundaries so an interrupted Kaggle run persists complete chunks
    and resumes from the last validated manifest.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "shard-manifest.json"
    records: list[dict[str, object]] = []
    total = len(metadata_records)
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("identity") != dict(identity):
            raise RuntimeError(
                f"existing checkpoint at {out_dir} has a different run identity; "
                "use a fresh output directory for a new identity instead of overwriting evidence"
            )
        records = load_shard_records(out_dir, identity)
        progress(f"resuming from checkpoint: {len(records)}/{total} records already complete")
    start_index = len(records)
    if start_index >= total:
        progress(f"checkpoint already covers the full catalog: {start_index}/{total}")
        return records
    t_start = time.perf_counter()
    last_checkpoint = start_index
    for chunk_start in range(start_index, total, batch_size):
        if deadline is not None and time.perf_counter() >= deadline:
            progress(
                f"time budget reached at {chunk_start}/{total} records; stopping "
                "and checkpointing for a later resumed run"
            )
            break
        chunk_end = min(chunk_start + batch_size, total)
        chunk_records: list[dict[str, object]] = []
        image_paths: dict[int, Path] = {}
        for index in range(chunk_start, chunk_end):
            record = dict(metadata_records[index])
            if record.get("image_status") == "pending":
                outcome = resolve_image(record, image_dir)
                record["image_status"] = outcome["status"]
                if outcome["status"] == "decoded":
                    record["image_sha256"] = outcome["sha256"]
                    image_paths[record["global_id"]] = Path(str(outcome["path"]))
                else:
                    record["image_error"] = outcome.get("error")
            chunk_records.append(record)
        eligible = [
            record for record in chunk_records
            if record.get("image_status") == "decoded"
            and record.get("title")
            and record["global_id"] in image_paths
        ]
        if eligible:
            caption_results = {
                result.item_id: result
                for result in captioner.caption_batch([
                    (record["global_id"], str(record["title"]), image_paths[record["global_id"]])
                    for record in eligible
                ])
            }
            paraphrase_results = {}
            if paraphraser is not None:
                paraphrase_results = {
                    result.item_id: result
                    for result in paraphraser.paraphrase_batch(
                        [(record["global_id"], str(record["title"])) for record in eligible]
                    )
                }
            for record in eligible:
                item_id = record["global_id"]
                caption_result = caption_results.get(item_id)
                if caption_result is not None:
                    record["caption_status"] = caption_result.status
                    record["caption_generated_tokens"] = caption_result.generated_token_count
                    record["caption_title_sha256"] = text_sha256(str(record["title"]))
                    record["caption_image_sha256"] = record.get("image_sha256")
                    record["caption_raw"] = caption_result.raw_text
                paraphrase_result = paraphrase_results.get(item_id)
                if paraphrase_result is not None:
                    record["paraphrase_status"] = paraphrase_result.status
                    record["paraphrase_text"] = paraphrase_result.raw_text
        records.extend(chunk_records)
        completed = len(records)
        if completed - last_checkpoint >= shard_size or completed == total:
            write_sharded_records(records, out_dir, identity, shard_size)
            last_checkpoint = completed
            elapsed = time.perf_counter() - t_start
            rate = (completed - start_index) / max(elapsed, 1e-9)
            eta_minutes = (total - completed) / max(rate, 1e-9) / 60
            progress(
                f"checkpoint: {completed}/{total} ({completed / total:.1%}) "
                f"elapsed={elapsed:.1f}s rate={rate:.2f}/s eta={eta_minutes:.1f}min"
            )
    if len(records) > last_checkpoint:
        # Deadline hit between checkpoints: flush the partial tail so
        # nothing since the last checkpoint is lost.
        write_sharded_records(records, out_dir, identity, shard_size)
        progress(f"final checkpoint before stopping: {len(records)}/{total} records saved")
    return records


def process_catalog_paraphrases_incrementally(
    records: list[dict[str, object]],
    paraphraser: Paraphraser,
    out_dir: Path,
    identity: dict[str, object],
    shard_size: int = 512,
    progress: Callable[[str], None] = print,
    deadline: float | None = None,
    batch_size: int = 64,
) -> list[dict[str, object]]:
    """Paraphrase decoded-image records without keeping the captioner loaded."""
    manifest_path = out_dir / "shard-manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("paraphrase pass requires a persisted caption checkpoint")
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if existing.get("identity") != dict(identity):
        raise RuntimeError("existing checkpoint identity mismatch before paraphrase pass")
    pending = [
        index for index, record in enumerate(records)
        if record.get("image_status") == "decoded"
        and record.get("title")
        and "paraphrase_status" not in record
    ]
    progress(f"paraphrase pass: {len(pending)} decoded-image records pending")
    started_at = time.perf_counter()
    processed = 0
    completed_since_checkpoint = 0
    for offset in range(0, len(pending), batch_size):
        if deadline is not None and time.perf_counter() >= deadline:
            progress(f"paraphrase time budget reached after {processed}/{len(pending)} pending records")
            break
        indices = pending[offset:offset + batch_size]
        items = [(int(records[index]["global_id"]), str(records[index]["title"])) for index in indices]
        results = {result.item_id: result for result in paraphraser.paraphrase_batch(items)}
        for index in indices:
            record = records[index]
            result = results.get(int(record["global_id"]))
            if result is not None:
                record["paraphrase_status"] = result.status
                record["paraphrase_text"] = result.raw_text
                record["paraphrase_generated_tokens"] = result.generated_token_count
        processed += len(indices)
        completed_since_checkpoint += len(indices)
        if completed_since_checkpoint >= shard_size or processed == len(pending):
            write_sharded_records(records, out_dir, identity, shard_size)
            progress(
                f"paraphrase checkpoint: {processed}/{len(pending)} pending "
                f"elapsed={time.perf_counter() - started_at:.1f}s"
            )
            completed_since_checkpoint = 0
    if completed_since_checkpoint:
        write_sharded_records(records, out_dir, identity, shard_size)
        progress(f"paraphrase checkpoint saved after {processed}/{len(pending)} pending records")
    return records


def run_full_corpus_generation(
    work_dir: Path,
    mixed_root: Path,
    temp_dir: Path = Path("/kaggle/temp"),
    shard_size: int = 512,
    progress: Callable[[str], None] = print,
    time_budget_seconds: float | None = None,
) -> dict[str, object]:
    """Generate the complete six-domain catalog and persisted arm shards.

    Orchestrates `resolve_full_catalog_metadata` and
    `process_catalog_records_incrementally` (checkpointed, resumable), then
    builds the five arms once every record is captioned/paraphrased.

    `time_budget_seconds`, when set, caps the generation loop's own
    wall-clock time (see `process_catalog_records_incrementally`'s
    `deadline`). If the budget expires before the whole catalog is
    generated, this function returns an explicit `status: "partial"`
    summary and skips arm assembly — arms need the complete captioned set,
    and are correctly deferred to a later call that resumes from this run's
    checkpoint and finishes the catalog.
    """
    t_start = time.perf_counter()
    _titles, metadata_records = resolve_full_catalog_metadata(mixed_root, temp_dir, progress=progress)
    captioner = Qwen3VLCaptioner(batch_size=CAPTION_BATCH_SIZE)
    image_dir = temp_dir / "full_corpus_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    def resolve_image(record: dict[str, object], image_dir: Path) -> dict[str, object]:
        dest = image_dir / f"{record['global_id']}.bin"
        return download_image(str(record["image_url"]), dest)

    input_catalog_sha256 = input_catalog_fingerprint(metadata_records)
    identity = {
        "stage": "full_corpus_generation",
        "catalog_size": CATALOG_SIZE,
        "input_catalog_sha256": input_catalog_sha256,
        "caption_model": captioner.model_id,
        "caption_revision": captioner.model_revision,
        "processor_revision": captioner.processor_revision,
        "caption_prompt_sha256": prompt_sha256(),
        "caption_max_new_tokens": CAPTION_MAX_NEW_TOKENS,
        "caption_do_sample": False,
        "caption_batch_size": CAPTION_BATCH_SIZE,
        "paraphrase_model": PARAPHRASE_MODEL_ID,
        "paraphrase_revision": PARAPHRASE_MODEL_REVISION,
        "shuffle_seed": 7001,
        "shard_size": shard_size,
    }
    out_dir = work_dir / "full_corpus"
    deadline = t_start + time_budget_seconds if time_budget_seconds is not None else None
    records = process_catalog_records_incrementally(
        metadata_records, captioner, None, out_dir, identity, image_dir, resolve_image,
        shard_size=shard_size, progress=progress, deadline=deadline,
    )
    del captioner
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if len(records) == len(metadata_records):
        paraphraser = QwenParaphraser(model_id=PARAPHRASE_MODEL_ID, revision=PARAPHRASE_MODEL_REVISION)
        records = process_catalog_paraphrases_incrementally(
            records, paraphraser, out_dir, identity, shard_size=shard_size,
            progress=progress, deadline=deadline,
        )
        del paraphraser
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pending_paraphrases = sum(
        record.get("image_status") == "decoded" and "paraphrase_status" not in record
        for record in records
    )
    if len(records) < len(metadata_records) or pending_paraphrases:
        status = {
            "status": "partial",
            "identity": identity,
            "record_count": len(records),
            "catalog_size": len(metadata_records),
            "paraphrase_pending_count": pending_paraphrases,
            "note": "time budget expired before captioning or paraphrasing completed; "
                    "re-run the same identity to resume before arm assembly",
        }
        progress(json.dumps(status, indent=2, sort_keys=True))
        return status


    train_files = sorted((mixed_root / "train").glob("*.csv"))
    sequences: list[list[int]] = []
    for path in train_files:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    sequences.append([int(value) for value in ast.literal_eval(row["history_item_id"])])
                except (KeyError, SyntaxError, ValueError):
                    continue
    frequencies = compute_training_frequencies(sequences)
    real_cues, paraphrase_cues = matched_cue_maps(records)
    donor_map = frequency_bin_derangement(sorted(real_cues), frequencies, seed=7001)
    rows = [
        CatalogRow(
            domain=str(record["domain"]), source_item_id=str(record["global_id"]),
            downstream_item_id=record["global_id"], parent_asin=record.get("asin"),
            source_title=str(record["title"]), order_index=record["global_id"],
            image_status=str(record.get("image_status", "unknown")),
            caption_status=str(record.get("caption_status", "unknown")),
            raw_caption=record.get("caption_raw"),
        )
        for record in records
    ]
    arms = build_arms(rows, real_cues, paraphrase_cues, donor_map)
    for record, arm in zip(records, arms):
        record["arm_texts"] = arm.texts
        record["shuffle_donor_id"] = arm.shuffle_donor_id

    manifest = write_sharded_records(records, out_dir, identity, shard_size)
    manifest["available_caption_count"] = len(real_cues)
    manifest["image_decoded_count"] = sum(record.get("image_status") == "decoded" for record in records)
    manifest["paraphrase_ok_count"] = sum(record.get("paraphrase_status") == "ok" for record in records)
    (work_dir / "full_corpus_summary.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    progress(json.dumps({key: value for key, value in manifest.items() if key != "shards"}, indent=2, sort_keys=True))
    return manifest
