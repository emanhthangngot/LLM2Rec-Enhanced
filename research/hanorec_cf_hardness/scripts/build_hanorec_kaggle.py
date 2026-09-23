"""Generate one self-contained HaNoRec correctness-smoke Kaggle script."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # research/hanorec_cf_hardness
SOURCE_DIR = ROOT
OUTPUT_DIR = ROOT / "kaggle" / "hanorec-faithful-signal-smoke"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    prep = (SOURCE_DIR / "prep.py").read_text(encoding="utf-8").replace("from __future__ import annotations\n", "")
    train = (SOURCE_DIR / "train.py").read_text(encoding="utf-8").replace("from __future__ import annotations\n", "")
    config = json.loads((SOURCE_DIR / "experiment.json").read_text(encoding="utf-8"))
    source_manifest = {
        "prep_sha256": sha256_text(prep),
        "train_sha256": sha256_text(train),
        "experiment_sha256": sha256_text(json.dumps(config, sort_keys=True)),
        "generator": "scripts/build_hanorec_kaggle.py",
    }
    config_literal = json.dumps(config, sort_keys=True)
    header = '''"""Generated corrected HaNoRec Games correctness smoke.

This file is intentionally self-contained: Kaggle kernel_sources expose output
files, not repository source. Regenerate it with scripts/build_hanorec_kaggle.py.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PINS = ["transformers==4.51.3", "peft==0.15.2", "accelerate==1.6.0", "bitsandbytes==0.45.5", "pillow==11.0.0"]
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *PINS], check=True)
import torch

if not torch.cuda.is_available():
    raise RuntimeError("correctness smoke requires a visible CUDA GPU")
print({"stage": "gpu_ready", "device": torch.cuda.get_device_name(0), "gpu_count": torch.cuda.device_count()})

'''
    footer = f'''

SOURCE_MANIFEST = {json.dumps(source_manifest, sort_keys=True)}
EXPERIMENT_CONFIG = json.loads({config_literal!r})
EXPERIMENT_CONFIG.update({{
    "train_pairs": 16, "eval_users": 4, "validation_users": 4,
    "sft_epochs": 1, "dpo_epochs": 1, "sft_batch_size": 1,
    "batch_size": 3, "eval_batch_size": 1, "max_pixels": 8192,
    "verify_scoring_batching": True,
    "sft_max_optimizer_updates": 2, "max_optimizer_updates": 2,
    "gradient_accumulation_steps": 2, "budget_seconds": 15000,
    "weights": [1.0], "smoke": True,
    "probe_candidates": [
        {{"batch_size": 1, "max_pixels": 8192}},
        {{"batch_size": 3, "max_pixels": 8192}},
        {{"batch_size": 4, "max_pixels": 8192}},
        {{"batch_size": 6, "max_pixels": 8192}},
        {{"batch_size": 12, "max_pixels": 8192}},
        {{"batch_size": 3, "max_pixels": 16384}},
        {{"batch_size": 4, "max_pixels": 16384}},
        {{"batch_size": 3, "max_pixels": 32768}},
        {{"batch_size": 6, "max_pixels": 32768}},
        {{"batch_size": 3, "max_pixels": 65536}},
        {{"batch_size": 4, "max_pixels": 262144}},
    ],
}})
INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working/hanorec_faithful_signal_smoke")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print({{"stage": "prepare_start", "source_manifest": SOURCE_MANIFEST}})
prepared = prepare(EXPERIMENT_CONFIG, INPUT_ROOT, OUTPUT_ROOT, Path("/kaggle/working"))
if len(prepared["train_item_ids"]) > 256:
    raise RuntimeError(f"smoke catalog unexpectedly contains {{len(prepared['train_item_ids'])}} items")
if int(EXPERIMENT_CONFIG["batch_size"]) < 3:
    raise RuntimeError("smoke batch_size must preserve HaNoRec's global mini-batch requirement")
(OUTPUT_ROOT / "prepared_manifest.json").write_text(json.dumps({{"provenance": prepared["provenance"], "train_count": len(prepared["train"]), "validation_count": len(prepared["validation"]), "evaluation_count": len(prepared["evaluation"]), "source_manifest": SOURCE_MANIFEST}}, indent=2, default=str), encoding="utf-8")
print({{"stage": "prepare_complete", "train_count": len(prepared["train"]), "train_item_count": len(prepared["train_item_ids"]), "validation_count": len(prepared["validation"]), "evaluation_count": len(prepared["evaluation"])}})

print({{"stage": "sft_start"}})
sft_bundle = run_sft(EXPERIMENT_CONFIG, prepared, OUTPUT_ROOT, Path("/kaggle/working"))
if sft_bundle.get("status") != "COMPLETE":
    raise RuntimeError("SFT smoke did not complete")
print({{"stage": "sft_complete", "optimizer_updates": sft_bundle["sft_optimizer_updates"]}})

print({{"stage": "dpo_start", "weight": 1.0, "image_condition": "real"}})
arm_result = run_arm(EXPERIMENT_CONFIG, OUTPUT_ROOT, 1.0, "real", OUTPUT_ROOT, Path("/kaggle/working"))
if arm_result.get("status") != "COMPLETE":
    raise RuntimeError("DPO smoke did not complete")
(OUTPUT_ROOT / "smoke_manifest.json").write_text(json.dumps({{"source_manifest": SOURCE_MANIFEST, "purpose": "correctness_only_non_signal", "must_not_be_used_as_signal": True, "sft_status": sft_bundle["status"], "arm_status": arm_result["status"], "sft_micro_batches": sft_bundle["sft_micro_batches"], "sft_optimizer_updates": sft_bundle["sft_optimizer_updates"], "dpo_micro_batches": arm_result["micro_batches"], "dpo_optimizer_updates": arm_result["optimizer_updates"], "expected_optimizer_updates": arm_result["expected_optimizer_updates"], "checkpoint": arm_result["checkpoint"], "checkpoint_reloaded": arm_result["checkpoint_reloaded"], "reference_reproducibility_check": sft_bundle["reference_reproducibility_check"], "sft_gpu_memory": sft_bundle["gpu_memory"], "dpo_gpu_memory": arm_result["gpu_memory"], "sft_stage_timings": sft_bundle["stage_timings"], "dpo_stage_timings": arm_result["stage_timings"], "dpo_elapsed_seconds": arm_result["elapsed_seconds"]}}, indent=2), encoding="utf-8")
print({{"stage": "smoke_complete", "status": "COMPLETE", "optimizer_updates": arm_result["optimizer_updates"], "mean_ndcg@10": arm_result["mean_ndcg@10"], "mean_recall@10": arm_result["mean_recall@10"]}})

print({{"stage": "probe_start"}})
probe_manifest = run_resource_probe(EXPERIMENT_CONFIG, prepared, OUTPUT_ROOT)
print({{"stage": "probe_complete", "status": probe_manifest["status"], "candidates": len(probe_manifest["candidates"])}})
'''
    runner = header + prep + "\n\n" + train + footer
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "runner.py").write_text(runner, encoding="utf-8")
    metadata = {
        "id": "trlxun/hanorec-faithful-signal-smoke",
        "title": "hanorec-faithful-signal-smoke",
        "code_file": "runner.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": ["trixuanle/llm2rec-games-5core", "trixuanle/llm2rec-amazonmix6-5core"],
        "kernel_sources": ["trixuanle/llm2rec-budgeted-games-sasrec-evaluation", "trixuanle/llm2rec-g1-preflight-and-visual-screen"],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaL4",
        "source_manifest": source_manifest,
    }
    (OUTPUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT_DIR), "source_manifest": source_manifest}, indent=2))


if __name__ == "__main__":
    main()
