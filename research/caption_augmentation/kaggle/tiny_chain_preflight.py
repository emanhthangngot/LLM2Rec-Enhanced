"""Validate the V10 caption corpus before tiny downstream training.

This kernel intentionally performs no model training. It is the fail-closed input
boundary for the real CSFT -> MNTP -> SimCSE -> extraction -> SASRec smoke.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path

CATALOG_SIZE = 108_753
REQUIRED_ARMS = {"title-only", "null", "real", "shuffle", "paraphrase"}
VIDEO_GAMES_START = 66_082
VIDEO_GAMES_STOP = 75_599


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def find_corpus_root() -> Path:
    input_root = Path("/kaggle/input")
    compact = sorted(input_root.rglob("v10-corpus-manifest.json"))
    if compact:
        return compact[-1].parent
    candidates = sorted(
        path.parent for path in input_root.rglob("shard-manifest.json")
        if path.is_file() and (path.parent / "full_corpus_summary.json").is_file()
    )
    if candidates:
        return candidates[-1]
    archives = sorted(
        path for path in input_root.rglob("*")
        if path.is_file() and (
            path.name.endswith((".tar.gz", ".tgz", ".tar", ".zip"))
            or "caption-corpus" in path.name.lower()
        )
    )
    extract_root = Path("/kaggle/temp/llm2rec-caption-v10")
    for archive_path in archives:
        candidate_root = extract_root / archive_path.stem.replace(".tar", "")
        candidate_root.mkdir(parents=True, exist_ok=True)
        marker = candidate_root / ".extracted"
        if not marker.is_file():
            try:
                if archive_path.name.endswith((".tar.gz", ".tgz", ".tar")):
                    with tarfile.open(archive_path, "r:*") as archive:
                        archive.extractall(candidate_root)
                elif archive_path.name.endswith(".zip"):
                    with zipfile.ZipFile(archive_path) as archive:
                        archive.extractall(candidate_root)
            except (tarfile.TarError, zipfile.BadZipFile, OSError):
                continue
            marker.write_text("ok", encoding="utf-8")
        nested = sorted(
            path.parent for path in candidate_root.rglob("shard-manifest.json")
            if path.is_file() and (path.parent / "full_corpus_summary.json").is_file()
        )
        if nested:
            return nested[-1]
    visible = sorted(
        str(path.relative_to(input_root))
        for path in input_root.rglob("*") if path.is_file()
    )[:100]
    raise FileNotFoundError(
        "V10 corpus manifest was not found in Kaggle input; visible files="
        + json.dumps(visible)
    )

def load_records(root: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    compact_manifest_path = root / "v10-corpus-manifest.json"
    if compact_manifest_path.is_file():
        transport = json.loads(compact_manifest_path.read_text(encoding="utf-8"))
        record_path = root / str(transport["record_file"])
        if not record_path.is_file():
            candidate_names = [record_path.name]
            if record_path.name.endswith(".gz"):
                candidate_names.append(record_path.name.removesuffix(".gz"))
            matches = sorted(
                path for name in candidate_names
                for path in Path("/kaggle/input").rglob(name)
                if path.is_file()
            )
            if not matches:
                raise FileNotFoundError(
                    f"compact record file not found: {transport['record_file']}"
                )
            record_path = matches[-1]
        payload = record_path.read_bytes()
        if record_path.name.endswith(".gz"):
            if sha256_bytes(payload) != transport["record_file_sha256"]:
                raise RuntimeError("compact corpus transport file hash mismatch")
            raw_payload = gzip.decompress(payload)
        else:
            raw_payload = payload
        if sha256_bytes(raw_payload) != transport["record_file_sha256_uncompressed"]:
            raise RuntimeError("compact corpus uncompressed hash mismatch")
        records = [
            json.loads(line.decode("utf-8"))
            for line in raw_payload.split(b"\n") if line
        ]
        if len(records) != transport["record_count"]:
            raise RuntimeError("compact corpus record count mismatch")
        if [row.get("global_id") for row in records] != list(range(CATALOG_SIZE)):
            raise RuntimeError("compact corpus global_id order is not contiguous")
        return transport, records
    manifest = json.loads((root / "shard-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("completion_status") != "complete":
        raise RuntimeError("corpus manifest is not complete")
    if manifest.get("record_count") != CATALOG_SIZE:
        raise RuntimeError(f"unexpected corpus record count: {manifest.get('record_count')}")
    records: list[dict[str, object]] = []
    for shard in manifest["shards"]:
        path = root / str(shard["path"])
        payload = path.read_bytes()
        if sha256_bytes(payload) != shard["sha256"]:
            raise RuntimeError(f"shard hash mismatch: {path.name}")
        rows = [json.loads(line.decode("utf-8")) for line in payload.split(b"\n") if line]
        if len(rows) != shard["records"]:
            raise RuntimeError(f"shard record count mismatch: {path.name}")
        records.extend(rows)
    if [row.get("global_id") for row in records] != list(range(CATALOG_SIZE)):
        raise RuntimeError("global_id order is not contiguous catalog order")
    return manifest, records

def validate_games_subset(records: list[dict[str, object]], size: int = 12) -> dict[str, object]:
    games = records[VIDEO_GAMES_START:VIDEO_GAMES_STOP]
    if len(games) != VIDEO_GAMES_STOP - VIDEO_GAMES_START:
        raise RuntimeError("Video_Games block size mismatch")
    selected = games[:: max(1, len(games) // size)][:size]
    if len(selected) != size:
        raise RuntimeError("could not select the declared tiny subset")
    for row in selected:
        if row["domain"] != "Video_Games":
            raise RuntimeError("tiny subset crossed a domain boundary")
        arm_texts = row.get("arm_texts", {})
        if set(arm_texts) != REQUIRED_ARMS:
            raise RuntimeError(f"arm key mismatch for global_id={row['global_id']}")
        if arm_texts["title-only"] != row["title"]:
            raise RuntimeError(f"title target mutation for global_id={row['global_id']}")
        for arm in REQUIRED_ARMS - {"title-only"}:
            if not str(arm_texts[arm]).startswith("Title: " + str(row["title"])):
                raise RuntimeError(f"title prefix mutation for arm={arm}, global_id={row['global_id']}")
        if row.get("caption_status") != "ok" or row.get("paraphrase_status") != "ok":
            raise RuntimeError(f"tiny subset contains unavailable cue row: {row['global_id']}")
    donors = {int(row["global_id"]): int(row["shuffle_donor_id"]) for row in selected}
    if any(item == donor for item, donor in donors.items()):
        raise RuntimeError("tiny subset contains a shuffle fixed point")
    return {
        "subset_size": len(selected),
        "global_ids": [row["global_id"] for row in selected],
        "target_titles": [row["title"] for row in selected],
        "arms": sorted(REQUIRED_ARMS),
        "target_contract": "original title; arm text is history-side input only",
        "shuffle_fixed_points": 0,
    }


def main() -> None:
    root = find_corpus_root()
    manifest, records = load_records(root)
    subset = validate_games_subset(records)
    artifact = {
        "status": "PASS",
        "stage": "tiny_chain_input_preflight",
        "corpus_manifest_sha256": sha256_bytes(
            (root / ("v10-corpus-manifest.json" if (root / "v10-corpus-manifest.json").is_file() else "shard-manifest.json")).read_bytes()
        ),
        "corpus_identity": manifest.get("identity", manifest),
        "record_count": len(records),
        "subset": subset,
        "training_executed": False,
        "next_stage": "tiny_real_csft_chain",
    }
    output = Path("/kaggle/working/tiny-chain-input-preflight.json")
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
