"""Caption-augmented CSFT using the pinned LLM2Rec training source."""
from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import json
import os
import runpy
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path("/kaggle/working")
SOURCE_DIR = ROOT / "llm2rec-source"
MODEL_DIR = ROOT / "qwen2-0.5b"
OUTPUT_DIR = ROOT / "csft-output"
AUGMENTED_DIR = ROOT / "caption-amazonmix"
MAX_STEPS = 1000
SOURCE_COMMIT = "73b481f710f67166ab958f4985d27b27fb410871"
SOURCE_HASHES = {
    "run_csft.py": "4ceb531f24c01ba7c18a00376f75fe3700a0834ec7bd78d918b0b70219b8020e",
    "dataset.py": "8cc542d80816faf66aae6ac80b45fc38c00fa1435b604488c96d73a2845fc714",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_dependencies() -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fire", "datasets", "peft"], check=True)


def find_amazon_root() -> Path:
    matches = sorted(Path("/kaggle/input").rglob("AmazonMix-6"))
    if not matches:
        raise FileNotFoundError("AmazonMix-6 dataset missing")
    root = matches[0] / "5-core"
    if not (root / "train" / "AmazonMix-6.csv").is_file():
        raise FileNotFoundError(f"AmazonMix-6 train CSV missing under {root}")
    return root


def find_resume_checkpoint() -> Path | None:
    candidates = sorted({path for path in Path("/kaggle/input").rglob("checkpoint-*")})
    candidates = [path for path in candidates if path.is_dir() and (path / "trainer_state.json").is_file()]
    if not candidates:
        return None
    return candidates[-1]


def load_caption_map() -> dict[str, str]:
    manifests = sorted(Path("/kaggle/input").rglob("v10-corpus-manifest.json"))
    if not manifests:
        raise FileNotFoundError("V10 compact manifest missing")
    manifest = json.loads(manifests[-1].read_text(encoding="utf-8"))
    expected = "521849b297ae9dfce3626f60db48c0facbeaa757bfc343bff2285d3d89af84d7"
    if manifest.get("source_manifest_sha256") != expected:
        raise RuntimeError("caption corpus source manifest mismatch")
    names = [manifest["record_file"], str(manifest["record_file"]).removesuffix(".gz")]
    files = sorted(path for name in names for path in Path("/kaggle/input").rglob(name) if path.is_file())
    if not files:
        raise FileNotFoundError("V10 compact record transport missing")
    path = files[-1]
    raw = path.read_bytes()
    raw = gzip.decompress(raw) if path.name.endswith(".gz") else raw
    if sha256_bytes(raw) != manifest["record_file_sha256_uncompressed"]:
        raise RuntimeError("caption corpus transport hash mismatch")
    caption_map: dict[str, str] = {}
    ambiguous: set[str] = set()
    for line in raw.splitlines():
        record = json.loads(line)
        if record.get("caption_status") != "ok":
            continue
        title = str(record["title"])
        caption = str(record["arm_texts"]["real"])
        if title in caption_map and caption_map[title] != caption:
            ambiguous.add(title)
        else:
            caption_map[title] = caption
    for title in ambiguous:
        caption_map.pop(title, None)
    if len(caption_map) < 100_000:
        raise RuntimeError(f"caption title map unexpectedly incomplete: {len(caption_map)}")
    return caption_map


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


HISTORY_TOKEN_BUDGET = 850
"""Token budget for the *plain-title* rendering of retained history items.

``cutoff_len=1024`` minus room for the prompt template, target tokens, and
special tokens (~174 token margin, approximated; not measured against the
exact upstream prompt template). The item COUNT that fits this budget under
plain titles is then applied identically to the caption-augmented text, so
every arm sees the same set of history items — only what text represents
each item changes. Without this, caption text (longer per item than a bare
title) pushes upstream `dataset.py`'s tail truncation (`tokens[-max_len:]`)
to silently drop more history items in the `real` arm than a title-only run
would drop at the same `cutoff_len`, confounding "added visual signal" with
"lost interaction history".
"""


def common_suffix_length(titles: list[str], tokenizer, budget: int) -> int:
    """Max trailing item count whose plain-title rendering fits ``budget`` tokens."""
    total = 0
    for index in range(len(titles) - 1, -1, -1):
        piece = len(tokenizer.encode(titles[index], add_special_tokens=False)) + 2
        if total + piece > budget:
            return len(titles) - 1 - index
        total += piece
    return len(titles)


def augment_csv(source: Path, destination: Path, caption_map: dict[str, str], tokenizer) -> dict[str, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = augmented = missing = 0
    history_items_dropped_for_budget = 0
    empty_history_after_budget = 0
    with source.open(encoding="utf-8", newline="") as source_handle, destination.open("w", encoding="utf-8", newline="") as output_handle:
        reader = csv.DictReader(source_handle)
        fields = reader.fieldnames or []
        if "history_item_title" not in fields:
            raise RuntimeError("AmazonMix CSV lacks history_item_title")
        writer = csv.DictWriter(output_handle, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            total += 1
            titles = [str(value) for value in ast.literal_eval(row["history_item_title"])]
            keep = common_suffix_length(titles, tokenizer, HISTORY_TOKEN_BUDGET)
            if keep == 0 and titles:
                keep = 1  # never fully empty a real history; the single most recent item stays.
            if keep < len(titles):
                history_items_dropped_for_budget += len(titles) - keep
            titles = titles[-keep:] if keep > 0 else titles
            if not titles:
                empty_history_after_budget += 1
            rendered = []
            for title in titles:
                text = caption_map.get(title)
                if text is None:
                    missing += 1
                    rendered.append(title)
                else:
                    augmented += 1
                    rendered.append(text)
            row["history_item_title"] = repr(rendered)
            writer.writerow(row)
    if augmented == 0 or missing > augmented:
        raise RuntimeError(f"caption augmentation coverage failed: augmented={augmented}, missing={missing}")
    return {
        "rows": total,
        "augmented_history_items": augmented,
        "missing_history_items": missing,
        "history_items_dropped_for_budget": history_items_dropped_for_budget,
        "empty_history_after_budget": empty_history_after_budget,
        "history_token_budget": HISTORY_TOKEN_BUDGET,
    }


def prepare_source() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/HappyPointer/LLM2Rec/{SOURCE_COMMIT}/llm2rec/"
    for name, expected in SOURCE_HASHES.items():
        path = SOURCE_DIR / name
        if not path.is_file():
            path.write_bytes(urllib.request.urlopen(base + name, timeout=60).read())
        if sha256(path) != expected:
            raise RuntimeError(f"pinned source hash mismatch for {name}")


def download_model() -> None:
    if (MODEL_DIR / "config.json").is_file() and (MODEL_DIR / "model.safetensors").is_file():
        return
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="Qwen/Qwen2-0.5B", local_dir=str(MODEL_DIR), local_dir_use_symlinks=False,
        allow_patterns=["config.json", "generation_config.json", "merges.txt", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json"],
    )


def build_compat_source() -> Path:
    source = (SOURCE_DIR / "run_csft.py").read_text(encoding="utf-8")
    replacements = {
        "torch_dtype=torch.bfloat16,": "torch_dtype=torch.float32,\n            attn_implementation=\"sdpa\",",
        "bf16=True,": "fp16=True,",
        "max_steps=10000,": f"max_steps={MAX_STEPS},",
        'evaluation_strategy="steps",': 'eval_strategy="steps",',
        "eval_steps=2000,": "eval_steps=1000,",
        'save_strategy="steps",': 'save_strategy="steps",',
        "save_steps=2000,": "save_steps=100,",
        "save_total_limit=5,": "save_total_limit=2,",
        "load_best_model_at_end=True,": "load_best_model_at_end=False,",
        'report_to="wandb",': 'report_to="none",',
        "callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]": "callbacks = []",
        "trainer.evaluate()": 'print("CSFT_SKIP_PRETRAIN_EVAL")',
        "model.save_pretrained(output_dir)": 'print("CSFT_CHECKPOINTS_PERSISTED")',
        '    from datasets import Dataset as HFDataset\n    hf_train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})\n    hf_train_dataset = hf_train_dataset.shuffle(seed=seed)\n\n    hf_val_dataset = HFDataset.from_dict({k: [v[k] for v in val_data] for k in val_data[0].keys()})': '    hf_train_dataset = train_data\n    hf_val_dataset = val_data',
    }
    for old, new in replacements.items():
        if old not in source:
            raise AssertionError(f"compatibility patch target missing: {old}")
        source = source.replace(old, new, 1)
    path = ROOT / "run_csft_caption_compat.py"
    path.write_text(source, encoding="utf-8")
    return path


def main() -> None:
    os.environ.update({"CUDA_VISIBLE_DEVICES": "0", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "WANDB_DISABLED": "true", "TOKENIZERS_PARALLELISM": "false"})
    install_dependencies()
    prepare_source()
    sys.path.insert(0, str(SOURCE_DIR))
    download_model()
    from transformers import AutoTokenizer
    budget_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    amazon = find_amazon_root()
    caption_map = load_caption_map()
    train_file = AUGMENTED_DIR / "train.csv"
    eval_file = AUGMENTED_DIR / "valid.csv"
    coverage = augment_csv(amazon / "train" / "AmazonMix-6.csv", train_file, caption_map, budget_tokenizer)
    eval_coverage = augment_csv(amazon / "valid" / "AmazonMix-6.csv", eval_file, caption_map, budget_tokenizer)
    compat = build_compat_source()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("caption CSFT requires CUDA")
    resume_checkpoint = find_resume_checkpoint()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    sys.argv = [str(compat), "--base_model", str(MODEL_DIR), "--train_file", str(train_file), "--eval_file", str(eval_file), "--output_dir", str(OUTPUT_DIR), "--category", "AmazonMix-6", "--train_from_scratch", "False", "--use_lora", "False", "--batch_size", "128", "--micro_batch_size", "1", "--num_epochs", "1", "--learning_rate", "3e-4", "--cutoff_len", "1024"]
    if resume_checkpoint is not None:
        sys.argv.extend(["--resume_from_checkpoint", str(resume_checkpoint)])
    try:
        runpy.run_path(str(compat), run_name="__main__")
    finally:
        elapsed = time.perf_counter() - started
        peak_vram = torch.cuda.max_memory_allocated()
        checkpoints = sorted(str(path) for path in OUTPUT_DIR.glob("checkpoint-*"))
        (ROOT / "caption_csft_diagnostics.json").write_text(
            json.dumps(
                {
                    "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
                    "elapsed_seconds": elapsed,
                    "peak_vram_gib": peak_vram / 1024**3,
                    "checkpoints": checkpoints,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    checkpoints = sorted(str(path) for path in OUTPUT_DIR.glob("checkpoint-*"))
    if not checkpoints:
        raise RuntimeError("caption CSFT produced no checkpoints")
    artifact = {
        "status": "PASS", "pipeline": "llm2rec-caption-csft", "training_executed": True,
        "source_commit": SOURCE_COMMIT, "source_hashes": SOURCE_HASHES,
        "caption_source_manifest_sha256": "521849b297ae9dfce3626f60db48c0facbeaa757bfc343bff2285d3d89af84d7",
        "augmentation": {"train": coverage, "valid": eval_coverage},
        "max_steps": MAX_STEPS, "checkpoints": checkpoints,
        "micro_batch_size": 1, "gradient_accumulation_steps": 128, "effective_batch_size": 128,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "elapsed_seconds": elapsed, "peak_vram_gib": peak_vram / 1024**3,
        "cuda_device": torch.cuda.get_device_name(0),
    }
    (ROOT / "caption_csft_artifact.json").write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
