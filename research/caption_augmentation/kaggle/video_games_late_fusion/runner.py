"""Amendment 3 (screening/protocol.md): late-fusion SASRec on Video_Games.

Three arms share the pinned LLM2Rec SASRec, SASREC_ARGS and seeds 2024/2025/2026;
only the frozen item-feature matrix differs. The ``title`` arm must reproduce
the pilot per-seed Recall@10 within 0.002, or the run is marked invalid.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
SOURCE_DIR = ROOT / "LLM2Rec"
MARKERS = ROOT / "arm-markers"
START = time.time()

SOURCE_COMMIT = "73b481f710f67166ab958f4985d27b27fb410871"
SOURCE_HASHES = {
    "repeated_evaluate_with_seqrec.py": "d90a431af66f934217b6808a0d66cfba0d95e3223390920d28d016fe526c33e1",
}
DOWNSTREAM_HASHES = {
    "train_data.txt": "1300e5deec29d4bede6e32bfa1a5eade563f53a6916da95d67754ec40b9475b8",
    "val_data.txt": "484a97cfbcff2a78e92612b15863ce08c84c285c7bfb325ec1f33d3884e6de16",
    "test_data.txt": "80074ffea37928d92c35038e9d8bbf3de0cfcf01308ae4173e3c680c1bc50d1f",
    "item_titles.json": "17f501809c4159905cfa6dfa5626d3d52d0be32bc9747ee89b003662f6f5e84c",
    "data.txt": "48c7ead98abfa73c506bf7af133703f2ee7e922167d922760bb074376639a72c",
}
FEATURE_HASHES = {
    "title.npy": "270ec23167bd6e9299363efe3cd62d9c9345ad692f29c21d3828a86cc999899d",
    "fused_real.npy": "f8a4d4c5e951e903b67ea19ef9882bc28914727d84fb3182bbb0ea0df7476450",
    "fused_shuffle.npy": "25826b9496fdef7950efab20349dedb3a00f8f68b867413ab10a8f7d9e762e81",
}
ARMS = ("title", "fused_real", "fused_shuffle")
SEEDS = (2024, 2025, 2026)
SASREC_ARGS = ("--model=SASRec", "--dataset=Games_5core", "--lr=1.0e-3", "--weight_decay=1.0e-4",
               "--dropout=0.3", "--loss_type=ce")
# Pilot title-only arm, per-seed test Recall@10 (results/video_games_pilot/title-only_arm_v3).
PILOT_TITLE_RECALL10 = (0.08373034000396729, 0.08294720202684402, 0.08464399725198746)
REPRO_TOL = 0.002
PINS = {"transformers": "4.44.2"}


def log(msg: str) -> None:
    print(f"[{(time.time() - START) / 60:6.1f} min] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_one(name: str) -> Path:
    hits = sorted(INPUT.rglob(name))
    if not hits:
        for archive in sorted(INPUT.rglob("*.zip")):
            with zipfile.ZipFile(archive) as z:
                if any(Path(n).name == name for n in z.namelist()):
                    target = ROOT / "unzipped" / archive.stem
                    z.extractall(target)
        hits = sorted((ROOT / "unzipped").rglob(name)) if (ROOT / "unzipped").exists() else []
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one {name}, found {hits}")
    return hits[0]


def verify_inputs() -> tuple[Path, dict[str, Path]]:
    downstream = {}
    for name, expected in DOWNSTREAM_HASHES.items():
        path = find_one(name)
        if sha256_file(path) != expected:
            raise RuntimeError(f"hash mismatch: {name}")
        downstream[name] = path
    parents = {p.parent for p in downstream.values()}
    if len(parents) != 1:
        raise RuntimeError(f"downstream files are split across {parents}")
    features = {}
    for name, expected in FEATURE_HASHES.items():
        path = find_one(name)
        if sha256_file(path) != expected:
            raise RuntimeError(f"hash mismatch: {name}")
        features[name.removesuffix(".npy")] = path
    return parents.pop(), features


def install_pins() -> dict[str, str]:
    import importlib.metadata
    missing = [f"{k}=={v}" for k, v in PINS.items() if importlib.metadata.version(k) != v]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
    return {k: importlib.metadata.version(k) for k in (*PINS, "torch", "accelerate", "numpy")}


def prepare_source(downstream_dir: Path) -> None:
    if not SOURCE_DIR.exists():
        archive = ROOT / "llm2rec-source.tar.gz"
        archive.write_bytes(urllib.request.urlopen(
            f"https://github.com/HappyPointer/LLM2Rec/archive/{SOURCE_COMMIT}.tar.gz", timeout=120).read())
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(ROOT, filter="data")
        (ROOT / f"LLM2Rec-{SOURCE_COMMIT}").rename(SOURCE_DIR)
        archive.unlink()
    for name, expected in SOURCE_HASHES.items():
        if sha256_file(SOURCE_DIR / name) != expected:
            raise RuntimeError(f"pinned source hash mismatch: {name}")
    target = SOURCE_DIR / "data" / "Video_Games" / "5-core" / "downstream"
    if target.exists():
        shutil.rmtree(SOURCE_DIR / "data")
    target.mkdir(parents=True)
    for name in DOWNSTREAM_HASHES:
        (target / name).symlink_to(downstream_dir / name)


def parse_results(text: str) -> list[dict[str, float]]:
    experiments: list[dict[str, float]] = []
    current = None
    for line in text.splitlines():
        if re.fullmatch(r"Experiment \d+:", line.strip()):
            current = {}
            experiments.append(current)
            continue
        match = re.fullmatch(r"([a-z]+@\d+): ([0-9.eE+-]+)", line.strip())
        if current is not None and match:
            current[match.group(1)] = float(match.group(2))
    if len(experiments) != len(SEEDS) or any(not e for e in experiments):
        raise RuntimeError(f"expected {len(SEEDS)} per-seed results, parsed {experiments}")
    return experiments


def run_arm(arm: str, embedding: Path) -> dict[str, object]:
    marker = MARKERS / f"{arm}.json"
    if marker.is_file():
        return json.loads(marker.read_text(encoding="utf-8"))
    import numpy as np
    matrix = np.load(embedding)
    if matrix.shape[0] != 9518 or not np.isfinite(matrix).all():
        raise RuntimeError(f"{arm}: invalid feature matrix {matrix.shape}")
    results_root = SOURCE_DIR / "Results"
    if results_root.exists():
        shutil.rmtree(results_root)
    env = {**os.environ, "PYTHONPATH": str(SOURCE_DIR), "WANDB_DISABLED": "true", "WANDB_MODE": "disabled",
           "CUDA_VISIBLE_DEVICES": "0"}
    started = time.time()
    log(f"SASRec start: {arm} {matrix.shape}")
    with (ROOT / f"sasrec_{arm}.log").open("w", encoding="utf-8") as handle:
        subprocess.run([sys.executable, "repeated_evaluate_with_seqrec.py", *SASREC_ARGS,
                        f"--embedding={embedding}", f"--run_id=VG-late-fusion-{arm}"],
                       cwd=SOURCE_DIR, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    files = sorted(results_root.glob("Games_5core/SASRec/**/results.txt"))
    if len(files) != 1:
        raise RuntimeError(f"{arm}: expected one results.txt, found {files}")
    shutil.copyfile(files[0], ROOT / f"sasrec_{arm}_results.txt")
    result = {"arm": arm, "feature_shape": list(matrix.shape), "feature_sha256": sha256_file(embedding),
              "per_seed": [{"seed": s, **m} for s, m in zip(SEEDS, parse_results(files[0].read_text(encoding="utf-8")))],
              "seconds": time.time() - started}
    MARKERS.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def decide(arms: dict[str, dict]) -> dict[str, object]:
    r10 = {a: [s["recall@10"] for s in arms[a]["per_seed"]] for a in arms}
    repro = [abs(x - y) for x, y in zip(r10["title"], PILOT_TITLE_RECALL10)]
    wins = sum(x > y for x, y in zip(r10["fused_real"], r10["fused_shuffle"]))
    mean = {a: sum(v) / len(v) for a, v in r10.items()}
    out = {"title_repro_abs_diff": repro, "title_reproduces_pilot": max(repro) <= REPRO_TOL,
           "real_beats_shuffle_seeds": wins, "mean_recall10": mean,
           "real_over_title_ratio": mean["fused_real"] / mean["title"]}
    if not out["title_reproduces_pilot"]:
        out["verdict"] = "INVALID_BASELINE_NOT_REPRODUCED"
    elif wins == len(SEEDS) and out["real_over_title_ratio"] >= 1.02:
        out["verdict"] = "PASS_LATE_FUSION_RECOVERS_CAPTION_SIGNAL"
    else:
        out["verdict"] = "FAIL"
    return out


def main() -> None:
    downstream_dir, features = verify_inputs()
    versions = install_pins()
    prepare_source(downstream_dir)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    log(f"inputs verified; versions {versions}; device {torch.cuda.get_device_name(0)}")
    arms = {arm: run_arm(arm, features[arm]) for arm in ARMS}
    artifact = {"protocol": "research/caption_augmentation/screening/protocol.md#amendment-3",
                "source_commit": SOURCE_COMMIT, "sasrec_args": SASREC_ARGS, "versions": versions,
                "cuda_device": torch.cuda.get_device_name(0), "arms": arms, "decision": decide(arms),
                "kernel_seconds": time.time() - START}
    (ROOT / "late_fusion_artifact.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact["decision"], indent=2))


if __name__ == "__main__":
    main()
