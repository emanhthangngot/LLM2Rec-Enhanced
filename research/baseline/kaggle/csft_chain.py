"""Run an approved, budgeted CSFT chain on the full AmazonMix-6 corpus."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path("/kaggle/working")
SOURCE_DIR = ROOT / "llm2rec-source"
MODEL_DIR = ROOT / "qwen2-0.5b"
OUTPUT_DIR = ROOT / "csft-output"
TARGET_STEPS = 1000
APPROVED_MAX_STEPS = 1000
SAVE_STEPS = 100
MICRO_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 64
MAX_STEPS = TARGET_STEPS
SOURCE_COMMIT = "73b481f710f67166ab958f4985d27b27fb410871"
SOURCE_HASHES = {
    "run_csft.py": "4ceb531f24c01ba7c18a00376f75fe3700a0834ec7bd78d918b0b70219b8020e",
    "dataset.py": "8cc542d80816faf66aae6ac80b45fc38c00fa1435b604488c96d73a2845fc714",
}


def ensure_dependencies() -> None:
    missing = []
    for module in ("fire", "datasets", "peft"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)


def find_amazon_root() -> Path:
    candidates = sorted(Path("/kaggle/input").rglob("AmazonMix-6"))
    if not candidates:
        raise FileNotFoundError("AmazonMix-6 was not found in Kaggle inputs")
    root = candidates[0] / "5-core"
    if not (root / "train" / "AmazonMix-6.csv").is_file():
        raise FileNotFoundError(f"AmazonMix-6 train file missing under {root}")
    return root


def unpack_input_archives() -> list[Path]:
    roots = [Path("/kaggle/input")]
    for archive in sorted(Path("/kaggle/input").rglob("data.zip")):
        destination = ROOT / "mounted-inputs" / archive.parent.name
        marker = destination / ".extracted"
        if not marker.is_file():
            destination.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(destination)
            marker.write_text("ok", encoding="utf-8")
        roots.append(destination)
    return roots


def find_resume_checkpoint() -> Path | None:
    roots = unpack_input_archives()
    candidates = sorted({path for root in roots for path in root.rglob("checkpoint-*")})
    candidates = [path for path in candidates if path.is_dir() and (path / "trainer_state.json").is_file()]
    if not candidates:
        return None
    return candidates[-1]


def prepare_source() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    base_url = f"https://raw.githubusercontent.com/HappyPointer/LLM2Rec/{SOURCE_COMMIT}/llm2rec/"
    for name, expected_hash in SOURCE_HASHES.items():
        target = SOURCE_DIR / name
        if not target.is_file():
            target.write_bytes(urllib.request.urlopen(base_url + name, timeout=60).read())
        actual_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(f"source hash mismatch for {name}: {actual_hash}")


def download_model() -> None:
    if (MODEL_DIR / "config.json").exists() and (MODEL_DIR / "model.safetensors").exists():
        return
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="Qwen/Qwen2-0.5B",
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "merges.txt",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ],
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
        "save_steps=2000,": f"save_steps={SAVE_STEPS},",
        "save_total_limit=5,": "save_total_limit=1,",
        "load_best_model_at_end=True,": "load_best_model_at_end=False,",
        "model.save_pretrained(output_dir)": 'print("CSFT_SKIP_FINAL_MODEL_SAVE")',
        'report_to="wandb",': 'report_to="none",',
        "callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]": "callbacks = []",
        "trainer.evaluate()": 'print("CSFT_SKIP_PRETRAIN_EVAL")',
        "trainer.train(resume_from_checkpoint=resume_from_checkpoint)": (
            "chain_train_started = __import__(\"time\").perf_counter()\n"
            "    trainer.train(resume_from_checkpoint=resume_from_checkpoint)\n"
            "    chain_train_elapsed = __import__(\"time\").perf_counter() - chain_train_started\n"
            "    with open(\"/kaggle/working/csft_train_metrics.json\", \"w\") as metrics_file:\n"
            "        __import__(\"json\").dump({\"training_elapsed_seconds\": chain_train_elapsed, \"log_history\": trainer.state.log_history}, metrics_file)"
        ),
    }
    for old, new in replacements.items():
        if old not in source:
            raise AssertionError(f"compatibility patch target missing: {old}")
        source = source.replace(old, new, 1)
    path = ROOT / "run_csft_compat.py"
    path.write_text(source, encoding="utf-8")
    return path


def main() -> None:
    # Kaggle may expose more than one CUDA device to the process even when the
    # kernel contract requests one T4.  The upstream Trainer then wraps the
    # model in DataParallel, whose gather step adds a large transient allocation.
    # Pin this compatibility run to one device before importing torch.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    ensure_dependencies()
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    data_root = find_amazon_root()
    prepare_source()
    sys.path.insert(0, str(SOURCE_DIR))
    download_model()
    compat_source = build_compat_source()
    resume_checkpoint = find_resume_checkpoint()
    train_file = data_root / "train" / "AmazonMix-6.csv"
    eval_file = data_root / "valid" / "AmazonMix-6.csv"

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Budgeted CSFT requires a Kaggle GPU")
    print(
        json.dumps(
            {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0),
            },
            sort_keys=True,
        )
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    sys.argv = [
        str(compat_source),
        "--base_model", str(MODEL_DIR),
        "--train_file", str(train_file),
        "--eval_file", str(eval_file),
        "--output_dir", str(OUTPUT_DIR),
        "--category", "AmazonMix-6",
        "--train_from_scratch", "False",
        "--use_lora", "False",
        "--batch_size", "128",
        "--micro_batch_size", str(MICRO_BATCH_SIZE),
        "--num_epochs", "1",
        "--learning_rate", "3e-4",
        "--cutoff_len", "1024",
    ]
    if resume_checkpoint is not None:
        sys.argv.extend(["--resume_from_checkpoint", str(resume_checkpoint)])
    try:
        runpy.run_path(str(compat_source), run_name="__main__")
    finally:
        elapsed = time.perf_counter() - started
        peak_vram = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        metrics_path = ROOT / "csft_train_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        artifact = {
            "pipeline": "llm2rec-budgeted-csft",
            "source_commit": SOURCE_COMMIT,
            "target_steps": TARGET_STEPS,
            "approved_max_steps": APPROVED_MAX_STEPS,
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": 128,
            "training_elapsed_seconds": metrics.get("training_elapsed_seconds"),
            "kernel_elapsed_seconds": elapsed,
            "peak_vram_gib": peak_vram / 1024**3,
            "log_history": metrics.get("log_history", []),
            "checkpoint_dirs": sorted(str(path) for path in OUTPUT_DIR.glob("checkpoint-*")),
        }
        (ROOT / "csft_chain_artifact.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
