"""Export AmazonMix-6 transition pairs for the zero-GPU caption screening.

CPU-only. Writes, per split, one gzip TSV row per interaction row:
``user_id  last_history_item_id  target_item_id  history_len``, plus the
``item_id -> item_asin`` map observed in the CSVs, so the caption-corpus
join can be cross-checked offline by ASIN. All analysis runs locally.
"""
from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working")
EXPECTED_CATALOG_SIZE = 108_753
REQUIRED_COLUMNS = {"user_id", "item_asin", "history_item_id", "item_id", "item_title"}
MAX_TITLE_MISMATCH_RATE = 0.001


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return " ".join(text.split())


def locate_root() -> Path:
    trains = sorted(p for p in INPUT.rglob("AmazonMix-6.csv") if p.parent.name == "train")
    if len(trains) != 1:
        raise FileNotFoundError(f"expected one train/AmazonMix-6.csv, found {trains}")
    return trains[0].parent.parent


def main() -> None:
    started = time.time()
    csv.field_size_limit(sys.maxsize)
    root = locate_root()
    titles_path = root / "info" / "item_titles.txt"
    titles = titles_path.read_text(encoding="utf-8").split("\n")
    if titles and titles[-1] == "":
        titles.pop()
    if len(titles) != EXPECTED_CATALOG_SIZE:
        raise RuntimeError(f"catalog size {len(titles)} != {EXPECTED_CATALOG_SIZE}")

    manifest: dict[str, object] = {
        "item_titles_sha256": sha256_file(titles_path),
        "catalog_size": len(titles),
        "splits": {},
    }
    item_asin: dict[int, str] = {}
    for split in ("train", "valid", "test"):
        source = root / split / "AmazonMix-6.csv"
        rows = mismatched_titles = self_pairs = 0
        destination = OUT / f"pairs_{split}.tsv.gz"
        with source.open(encoding="utf-8", newline="") as handle, gzip.open(destination, "wt", encoding="utf-8") as out:
            reader = csv.DictReader(handle)
            missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise RuntimeError(f"{split}: missing columns {sorted(missing)}")
            for row in reader:
                history = [int(v) for v in ast.literal_eval(row["history_item_id"])]
                if not history:
                    raise RuntimeError(f"{split}: empty history for user {row['user_id']}")
                target = int(row["item_id"])
                if not 0 <= target < len(titles) or not all(0 <= h < len(titles) for h in history):
                    raise RuntimeError(f"{split}: item id out of catalog range in user {row['user_id']}")
                if normalize(titles[target]) != normalize(row["item_title"]):
                    mismatched_titles += 1
                asin = row["item_asin"]
                if item_asin.setdefault(target, asin) != asin:
                    raise RuntimeError(f"item {target} has two ASINs: {item_asin[target]} vs {asin}")
                self_pairs += history[-1] == target
                out.write(f"{row['user_id']}\t{history[-1]}\t{target}\t{len(history)}\n")
                rows += 1
        if rows == 0:
            raise RuntimeError(f"{split}: no rows")
        if mismatched_titles / rows > MAX_TITLE_MISMATCH_RATE:
            raise RuntimeError(f"{split}: target title mismatch rate {mismatched_titles}/{rows}")
        manifest["splits"][split] = {
            "source_sha256": sha256_file(source), "rows": rows, "target_title_mismatches": mismatched_titles,
            "self_pairs": self_pairs, "output": destination.name, "output_sha256": sha256_file(destination),
        }
        print(f"{split}: rows={rows} title_mismatches={mismatched_titles} self_pairs={self_pairs}", flush=True)

    asin_path = OUT / "item_asin.tsv.gz"
    with gzip.open(asin_path, "wt", encoding="utf-8") as out:
        for item_id in sorted(item_asin):
            out.write(f"{item_id}\t{item_asin[item_id]}\n")
    manifest["item_asin"] = {"items": len(item_asin), "output_sha256": sha256_file(asin_path)}
    manifest["seconds"] = time.time() - started
    (OUT / "screening_pairs_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
