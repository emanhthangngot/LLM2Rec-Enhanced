"""Run the released MNTP and SimCSE stages on the budgeted CSFT checkpoint."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


ROOT = Path("/kaggle/working")
SOURCE_DIR = ROOT / "llm2rec-source"
STAGE1_DIR = ROOT / "iem-stage1"
STAGE2_DIR = ROOT / "iem-stage2"
STAGED_CHECKPOINT_DIR = ROOT / "csft-checkpoint-1000-iem"
TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
}
SOURCE_COMMIT = "73b481f710f67166ab958f4985d27b27fb410871"
SOURCE_FILES = {
    "run_mntp.py": "b846c9ab270690037d7790d3078ae1e489797c376b98654522cf4b631566538a",
    "run_unsupervised_SimCSE.py": "d23d5781fab51bd80a2b1ec45c57b2ad0c2e29bd537810ce9e5e7eac8e69c7c9",
    "dataset_utils.py": "e4f4dd3c0bd49262cc67d215ae3e6e9aa5384a86739870d9f1ad6f6f3f02d03f",
    "train_mntp_config.json": "b4033d8fa431a006a931a8e5184ce624ba6648f6c88ccc3ee4d3b0f4bad2b61e",
    "train_simcse_config.json": "007b729d6fa28db36f91930ec2d88b9a24751ae1cf6e5187fe55ee91a8bb7465",
}
REC_DATA_FILES = {
    "recdata/__init__.py": None,
    "recdata/dataset.py": "a376bee5780ab861194b1862af1e229cf97efe9cd417adb55d8fbcbd921e8f1d",
    "recdata/RecItemData.py": "eacc52e77ac09491e4f14f6b425bad18fa2c75d72ae1c71cc3f7988cb32a4538",
    "recdata/SeqRecData.py": "5881a195ac88b76f693ff590fb95e4dd8880c17b06f21a2f8eca2cea74154a16",
    "recdata/ItemTitleData.py": "db0cada8c22348dd751555053a50dd8bca175675fdb35c4b18a4bf3928290fee",
}


def install_dependencies() -> None:
    packages = [
        "llm2vec==0.2.3",
        "transformers==4.44.2",
        "evaluate",
        "datasets",
        "peft",
        "fire",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)


def patch_qwen2_bidirectional_mask() -> dict[str, str]:
    """Replace Qwen2's inherited causal mask with a padding-only SDPA mask."""
    distribution = importlib.metadata.distribution("llm2vec")
    path = Path(distribution.locate_file("llm2vec/models/bidirectional_qwen2.py"))
    if not path.is_file():
        raise ModuleNotFoundError(f"llm2vec Qwen2 implementation was not found: {path}")
    source = path.read_text(encoding="utf-8")
    original_hash = hashlib.sha256(source.encode()).hexdigest()
    marker = "        self.post_init()\n\n\nclass Qwen2BiForMNTP"
    if marker not in source:
        raise RuntimeError("llm2vec Qwen2 patch marker missing; dependency source changed")
    method = '''        self.post_init()

    def _update_causal_mask(
        self,
        attention_mask,
        input_tensor,
        cache_position,
        past_key_values,
        output_attentions,
    ):
        """Build an additive padding-only mask for truly bidirectional SDPA."""
        if past_key_values is not None and past_key_values.get_seq_length() != 0:
            raise RuntimeError("Bidirectional Qwen2 does not support cached decoding")
        if attention_mask is None:
            return None
        if attention_mask.dim() != 2:
            raise ValueError(
                f"Bidirectional Qwen2 expects a 2D padding mask, found {attention_mask.shape}"
            )
        dtype = input_tensor.dtype
        min_dtype = torch.finfo(dtype).min
        padding_mask = attention_mask[:, None, None, :].eq(0)
        padding_mask = padding_mask.expand(
            input_tensor.shape[0], 1, input_tensor.shape[1], attention_mask.shape[-1]
        )
        return torch.zeros(
            padding_mask.shape, dtype=dtype, device=input_tensor.device
        ).masked_fill(padding_mask, min_dtype)


class Qwen2BiForMNTP'''
    patched = source.replace(marker, method, 1)
    path.write_text(patched, encoding="utf-8")
    patched_hash = hashlib.sha256(patched.encode()).hexdigest()
    if original_hash == patched_hash:
        raise RuntimeError("Qwen2 bidirectional mask patch made no source change")
    return {
        "path": str(path),
        "original_sha256": original_hash,
        "patched_sha256": patched_hash,
    }


def validate_bidirectional_model(checkpoint: Path) -> dict[str, object]:
    """Prove future-token sensitivity before starting either IEM trainer."""
    import torch
    from llm2vec import LLM2Vec

    encoder = LLM2Vec.from_pretrained(
        checkpoint,
        enable_bidirectional=True,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model = encoder.model.eval().cuda()
    vocab_size = int(model.config.vocab_size)
    prefix = [min(100, vocab_size - 1), min(200, vocab_size - 1)]
    input_a = torch.tensor([prefix + [min(300, vocab_size - 1)]], device="cuda")
    input_b = torch.tensor([prefix + [min(301, vocab_size - 1)]], device="cuda")
    attention_mask = torch.ones_like(input_a)
    with torch.no_grad():
        hidden_a = model(input_ids=input_a, attention_mask=attention_mask).last_hidden_state
        hidden_b = model(input_ids=input_b, attention_mask=attention_mask).last_hidden_state
    prefix_delta = float((hidden_a[:, :2] - hidden_b[:, :2]).abs().max().item())
    layer_classes = sorted({layer.self_attn.__class__.__name__ for layer in model.layers})
    is_causal = sorted({bool(layer.self_attn.is_causal) for layer in model.layers})
    result = {
        "model_class": model.__class__.__name__,
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
        "attention_classes": layer_classes,
        "attention_is_causal": is_causal,
        "future_token_changes_prefix_hidden_state_max_abs": prefix_delta,
    }
    if (
        result["model_class"] != "Qwen2BiModel"
        or result["attention_implementation"] != "sdpa"
        or is_causal != [False]
        or prefix_delta <= 1e-6
    ):
        raise RuntimeError(f"Qwen2 bidirectional preflight failed: {result}")
    result["passed"] = True
    del model, encoder, hidden_a, hidden_b
    torch.cuda.empty_cache()
    return result


def find_checkpoint() -> Path:
    candidates = sorted(Path("/kaggle/input").rglob("checkpoint-1000"))
    candidates = [path for path in candidates if path.is_dir() and (path / "config.json").is_file()]
    if not candidates:
        raise FileNotFoundError("Budgeted CSFT checkpoint-1000 was not found in Kaggle inputs")
    return candidates[-1]


def find_amazon_root() -> Path:
    candidates = sorted(Path("/kaggle/input").rglob("AmazonMix-6"))
    if not candidates:
        raise FileNotFoundError("AmazonMix-6 was not found in Kaggle inputs")
    root = candidates[0] / "5-core"
    if not (root / "info" / "item_titles.txt").is_file():
        raise FileNotFoundError(f"item_titles.txt missing under {root}")
    return root


def find_base_tokenizer() -> Path:
    candidates = sorted(Path("/kaggle/input").rglob("qwen2-0.5b/tokenizer.json"))
    if not candidates:
        raise FileNotFoundError("The valid Qwen2 base tokenizer was not found in Kaggle inputs")
    return candidates[-1].parent


def stage_checkpoint_with_base_tokenizer(checkpoint: Path) -> tuple[Path, Path]:
    """Use CSFT weights but replace the incompatible serialized tokenizer files."""
    tokenizer_source = find_base_tokenizer()
    if STAGED_CHECKPOINT_DIR.exists():
        shutil.rmtree(STAGED_CHECKPOINT_DIR)
    STAGED_CHECKPOINT_DIR.mkdir(parents=True)
    for source in checkpoint.iterdir():
        if source.name in TOKENIZER_FILES:
            continue
        (STAGED_CHECKPOINT_DIR / source.name).symlink_to(source)
    for name in TOKENIZER_FILES:
        source = tokenizer_source / name
        if source.is_file():
            (STAGED_CHECKPOINT_DIR / name).symlink_to(source)

    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(STAGED_CHECKPOINT_DIR, use_fast=True)
    return STAGED_CHECKPOINT_DIR, tokenizer_source


def fetch_source() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    base_url = f"https://raw.githubusercontent.com/HappyPointer/LLM2Rec/{SOURCE_COMMIT}/llm2rec/"
    for name, expected_hash in SOURCE_FILES.items():
        target = SOURCE_DIR / name
        if not target.is_file():
            target.write_bytes(urllib.request.urlopen(base_url + name, timeout=60).read())
        actual_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(f"source hash mismatch for {name}: {actual_hash}")
    for name, expected_hash in REC_DATA_FILES.items():
        target = SOURCE_DIR / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            if expected_hash is None:
                target.write_text("", encoding="utf-8")
            else:
                target.write_bytes(
                    urllib.request.urlopen(base_url + name, timeout=60).read()
                )
        if expected_hash is not None:
            actual_hash = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                raise RuntimeError(f"source hash mismatch for {name}: {actual_hash}")


def prepare_configs(checkpoint: Path, titles: Path) -> tuple[Path, Path, Path]:
    staged_checkpoint, tokenizer_source = stage_checkpoint_with_base_tokenizer(checkpoint)
    mntp = json.loads((SOURCE_DIR / "train_mntp_config.json").read_text(encoding="utf-8"))
    mntp.update(
        {
            "model_name_or_path": str(staged_checkpoint),
            "dataset_name": str(titles),
            "output_dir": str(STAGE1_DIR),
            "torch_dtype": "float32",
            "attn_implementation": "sdpa",
            "eval_strategy": "steps",
            "fp16": True,
            "bf16": False,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "report_to": "none",
            "logging_steps": 1,
            "save_steps": 500,
            "save_total_limit": 1,
            "save_only_model": True,
        }
    )
    mntp.pop("evaluation_strategy", None)
    mntp_path = ROOT / "train_mntp_budgeted.json"
    mntp_path.write_text(json.dumps(mntp, indent=2), encoding="utf-8")

    simcse = json.loads((SOURCE_DIR / "train_simcse_config.json").read_text(encoding="utf-8"))
    data_root = ROOT / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    amazon_link = data_root / "AmazonMix-6"
    if amazon_link.exists() or amazon_link.is_symlink():
        if amazon_link.is_symlink() or amazon_link.is_file():
            amazon_link.unlink()
        else:
            shutil.rmtree(amazon_link)
    amazon_link.symlink_to(titles.parent.parent.parent)
    simcse.update(
        {
            "model_name_or_path": str(STAGE1_DIR / "checkpoint-1000"),
            "output_dir": str(STAGE2_DIR),
            "torch_dtype": "float32",
            "attn_implementation": "sdpa",
            "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 32,
            "gradient_accumulation_steps": 1,
            "fp16": True,
            "bf16": False,
            "report_to": "none",
            "lora_r": None,
            "save_steps": 500,
            "save_total_limit": 2,
            "save_only_model": True,
            "logging_steps": 10,
        }
    )
    simcse_path = ROOT / "train_simcse_budgeted.json"
    simcse_path.write_text(json.dumps(simcse, indent=2), encoding="utf-8")
    return mntp_path, simcse_path, tokenizer_source


def compact_stage1_for_simcse(stage1_checkpoint: Path) -> None:
    """Release duplicate MNTP outputs before SimCSE writes its checkpoints."""
    for child in STAGE1_DIR.iterdir():
        if child == stage1_checkpoint:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    for cache_dir in (
        Path.home() / ".cache" / "huggingface" / "datasets",
        Path.home() / ".cache" / "pip",
    ):
        if cache_dir.exists():
            shutil.rmtree(cache_dir)


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_DISABLED"] = "true"
    install_dependencies()
    mask_patch = patch_qwen2_bidirectional_mask()
    checkpoint = find_checkpoint()
    amazon_root = find_amazon_root()
    fetch_source()
    sys.path.insert(0, str(SOURCE_DIR))
    mntp_config, simcse_config, tokenizer_source = prepare_configs(
        checkpoint, amazon_root / "info" / "item_titles.txt"
    )
    bidirectional_preflight = validate_bidirectional_model(STAGED_CHECKPOINT_DIR)

    subprocess.run([sys.executable, str(SOURCE_DIR / "run_mntp.py"), str(mntp_config)], check=True)
    stage1_checkpoint = STAGE1_DIR / "checkpoint-1000"
    if not stage1_checkpoint.is_dir():
        raise FileNotFoundError(f"MNTP checkpoint-1000 missing: {stage1_checkpoint}")
    compact_stage1_for_simcse(stage1_checkpoint)

    subprocess.run(
        [sys.executable, str(SOURCE_DIR / "run_unsupervised_SimCSE.py"), str(simcse_config)],
        check=True,
    )
    for step in (500, 1000):
        if not (STAGE2_DIR / f"checkpoint-{step}").is_dir():
            raise FileNotFoundError(f"SimCSE checkpoint-{step} missing")

    artifact = {
        "pipeline": "llm2rec-budgeted-iem",
        "source_commit": SOURCE_COMMIT,
        "csft_checkpoint": str(checkpoint),
        "mntp_checkpoint": str(stage1_checkpoint),
        "simcse_checkpoints": [str(STAGE2_DIR / "checkpoint-500"), str(STAGE2_DIR / "checkpoint-1000")],
        "runtime_dtype": "float16_autocast",
        "parameter_dtype": "float32",
        "runtime_attention": "sdpa",
        "qwen2_bidirectional_mask_patch": mask_patch,
        "bidirectional_preflight": bidirectional_preflight,
        "mntp_effective_batch": 32,
        "simcse_batch": 32,
        "simcse_official_batch": 256,
        "simcse_batch_deviation": "32 in-batch negatives on one T4; gradient accumulation intentionally not used because it does not increase in-batch negatives.",
        "tokenizer_override": "CSFT checkpoint tokenizer replaced with the validated base Qwen2 tokenizer because the checkpoint tokenizer.json failed the fast-tokenizer ModelWrapper parser.",
        "tokenizer_source": str(tokenizer_source),
        "storage_profile": {
            "mntp_save_steps": 500,
            "simcse_save_steps": [500, 1000],
            "save_only_model": True,
            "stage1_cleanup_before_simcse": True,
        },
    }
    (ROOT / "iem_pipeline_artifact.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
