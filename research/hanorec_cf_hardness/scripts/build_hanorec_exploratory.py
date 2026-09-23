"""Generate one self-contained HaNoRec exploratory 1-seed Kaggle script.

Base: version 17 calibration (prep/train sources unchanged). Differences:
- exploratory scope: seed 2024, 530 train pairs, 25/25 eval users, 1 epoch,
  batch 3 / max_pixels 4096, weights [1.0, 0.0, 0.5], real+shuffle.
- uses train_and_evaluate (matched SFT parents + all 6 DPO arms).
- no smoke caps, no 256-item catalog check, no resource probe.
- Regenerate with scripts/build_hanorec_exploratory.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # research/hanorec_cf_hardness
SOURCE_DIR = ROOT
OUTPUT_DIR = ROOT / "kaggle" / "hanorec-exploratory-1seed"


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
        "generator": "scripts/build_hanorec_exploratory.py",
    }
    config_literal = json.dumps(config, sort_keys=True)
    header = '''"""Generated HaNoRec Games exploratory 1-seed run.

Exploratory only, explicitly non-confirmatory. Regenerate with
scripts/build_hanorec_exploratory.py.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

PINS = ["transformers==4.51.3", "peft==0.15.2", "accelerate==1.6.0", "bitsandbytes==0.45.5", "pillow==11.0.0"]
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *PINS], check=True)
import torch

if not torch.cuda.is_available():
    raise RuntimeError("exploratory run requires a visible CUDA GPU")
print({"stage": "gpu_ready", "device": torch.cuda.get_device_name(0), "gpu_count": torch.cuda.device_count()})

'''
    footer = f'''
SOURCE_MANIFEST = {json.dumps(source_manifest, sort_keys=True)}
EXPERIMENT_CONFIG = json.loads({config_literal!r})
EXPERIMENT_CONFIG.update({{
    "seed": 2024,
    "train_pairs": 530, "eval_users": 25, "validation_users": 25,
    "sft_epochs": 1, "dpo_epochs": 1, "sft_batch_size": 1,
    "batch_size": 3, "eval_batch_size": 1, "max_pixels": 4096,
    "verify_scoring_batching": True,
    "gradient_accumulation_steps": 2, "budget_seconds": 30000,
    "weights": [1.0, 0.0, 0.5],
    "smoke": False,
}})
INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working/hanorec_exploratory_1seed")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print({{"stage": "prepare_start", "source_manifest": SOURCE_MANIFEST}})
prepared = prepare(EXPERIMENT_CONFIG, INPUT_ROOT, OUTPUT_ROOT, Path("/kaggle/working"))
if int(EXPERIMENT_CONFIG["batch_size"]) < 3:
    raise RuntimeError("exploratory batch_size must preserve HaNoRec's global mini-batch requirement")
print({{"stage": "prepare_complete", "train_count": len(prepared["train"]), "validation_count": len(prepared["validation"]), "evaluation_count": len(prepared["evaluation"])}})

result = train_and_evaluate(EXPERIMENT_CONFIG, prepared, OUTPUT_ROOT, Path("/kaggle/working"))
manifest = {{
    "source_manifest": SOURCE_MANIFEST,
    "purpose": "exploratory_only_non_confirmatory",
    "must_not_be_used_as_confirmation": True,
    "exploratory_scope": {{
        "seed": 2024, "train_pairs": 530, "validation_users": 25, "test_users": 25,
        "epochs": 1, "batch_size": 3, "max_pixels": 4096,
        "weights": [1.0, 0.0, 0.5], "image_conditions": ["real", "shuffle"],
    }},
    "limitations": [
        "Single seed; no cross-seed inference.",
        "One epoch; reduced eval population (25/25 users).",
        "Batch 3 at max_pixels 4096 differs from the frozen batch 4 / max_pixels 262144 recipe.",
        "max_pixels reduced from 8192 to 4096 after v1 CUDA OOM in the w1.0/real DPO arm at optimizer update 62/89 (torch.OutOfMemoryError, 14.55/14.56 GiB reserved); the v17 smoke probe measured only a single forward+backward pass on a fixed 3-pair sample and did not capture worst-case real-catalog image sizes.",
        "Cannot support the planned confirmatory or cross-seed claim.",
    ],
    "status": result["status"],
    "sft_bundles": result["sft_bundles"],
    "arms": [{{"weight": arm["weight"], "image_condition": arm["image_condition"], "status": arm["status"], "optimizer_updates": arm["optimizer_updates"], "expected_optimizer_updates": arm["expected_optimizer_updates"], "mean_ndcg@10": arm["mean_ndcg@10"], "mean_recall@10": arm["mean_recall@10"], "checkpoint": arm["checkpoint"], "checkpoint_reloaded": arm["checkpoint_reloaded"], "parent_sft_sha256": arm["parent_sft_sha256"]}} for arm in result["arms"]],
}}
(OUTPUT_ROOT / "exploratory_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
if result["status"] != "COMPLETE":
    raise RuntimeError("exploratory run did not complete")
print({{"stage": "exploratory_complete", "status": "COMPLETE", "arms": len(result["arms"])}})
'''
    runner = header + prep + "\n\n" + train + footer
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "runner.py").write_text(runner, encoding="utf-8")
    metadata = {
        "id": "trlxun/hanorec-exploratory-1seed",
        "title": "hanorec-exploratory-1seed",
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
