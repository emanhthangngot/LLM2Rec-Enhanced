"""Audit saved LLM2Rec SASRec ranks and bidirectional Qwen2 behavior."""

from __future__ import annotations

import ast
import gc
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
SEEDS = (2024, 2025, 2026)
STEPS = (500, 1000)
QWEN2_ORIGINAL_SHA256 = "ef0985fe8d386d5d8ebafb98968ed6381a4a69ed43d6a75af7b1f823d08e67b7"
QWEN2_PATCHED_SHA256 = "b3dfa1df58ea1eee8121d700f5f37ce42268a53828609c2db9456b5d84c89e47"
EXPECTED = {
    500: {
        "ndcg@10": 0.04975216214855512,
        "recall@10": 0.08207705120245616,
        "ndcg@20": 0.056554569552342095,
        "recall@20": 0.10907350728909175,
    },
    1000: {
        "ndcg@10": 0.050377074629068375,
        "recall@10": 0.08205529799064,
        "ndcg@20": 0.05719099069635073,
        "recall@20": 0.10911701371272405,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_one(pattern: str, predicate=lambda path: True) -> Path:
    matches = sorted(path for path in INPUT.rglob(pattern) if predicate(path))
    if not matches:
        raise FileNotFoundError(f"Kaggle input missing: {pattern}")
    return matches[-1]


def prepare_evaluation_source() -> tuple[Path, Path, Path]:
    evaluation_artifact = find_one("games_evaluation_artifact.json")
    evaluation_root = evaluation_artifact.parent
    evaluation = json.loads(evaluation_artifact.read_text(encoding="utf-8"))
    repeated = evaluation_root / "LLM2Rec" / "repeated_evaluate_with_seqrec.py"
    if not repeated.is_file() or not (repeated.parent / "seqrec" / "ckpt").is_dir():
        raise FileNotFoundError(f"evaluation source/checkpoints missing under {evaluation_root}")
    mounted_source = repeated.parent
    source = ROOT / "LLM2Rec-audit-source"
    if not source.exists():
        shutil.copytree(
            mounted_source,
            source,
            ignore=shutil.ignore_patterns(
                ".git", "Results", "ckpt", "item_info", "__pycache__", "*.pth", "*.npy"
            ),
        )
    games_root = find_one(
        "5-core", lambda path: path.is_dir() and (path / "downstream" / "test_data.txt").is_file()
    )
    for name, provenance in evaluation["dataset_source"]["files"].items():
        path = games_root / "downstream" / name
        if not path.is_file() or sha256_file(path) != provenance["sha256"]:
            raise RuntimeError(f"Games dataset provenance mismatch: {path}")
    data_link = source / "data"
    if data_link.is_symlink():
        data_link.unlink()
    if not data_link.exists():
        data_link.symlink_to(games_root.parent.parent)
    return source, games_root, evaluation_root


def metric_vectors(predictions: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    matches = predictions == labels[:, None]
    ranks = np.where(matches.any(axis=1), matches.argmax(axis=1) + 1, 0)
    return {
        "recall@10": ((ranks > 0) & (ranks <= 10)).astype(np.float32),
        "ndcg@10": np.where(
            (ranks > 0) & (ranks <= 10), 1.0 / np.log2(ranks + 1), 0.0
        ).astype(np.float32),
        "recall@20": ((ranks > 0) & (ranks <= 20)).astype(np.float32),
        "ndcg@20": np.where(ranks > 0, 1.0 / np.log2(ranks + 1), 0.0).astype(np.float32),
        "rank": ranks.astype(np.int16),
    }


def checkpoint_manifest(evaluation_root: Path) -> dict[tuple[int, int], Path]:
    checkpoint_paths = list(evaluation_root.rglob("*.pth"))
    checkpoints = {path.name: path for path in checkpoint_paths}
    if len(checkpoint_paths) != 6 or len(checkpoints) != 6:
        raise RuntimeError(
            f"expected six uniquely named SASRec checkpoints, found {len(checkpoint_paths)}"
        )
    manifest: dict[tuple[int, int], Path] = {}
    for step in STEPS:
        log = evaluation_root / f"games_sasrec_step{step}.log"
        if not log.is_file():
            raise FileNotFoundError(f"evaluation log missing: {log}")
        seed = None
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.startswith("{'model':"):
                config = ast.literal_eval(line)
                seed = int(config["rand_seed"])
                if f"step{step}_title_item_embs.npy" not in config["embedding"]:
                    raise RuntimeError(f"step {step} log references wrong embedding: {config}")
            if line.startswith("Loaded best model checkpoint from "):
                if seed not in SEEDS:
                    raise RuntimeError(f"checkpoint line has no valid seed context: {line}")
                name = Path(line.removeprefix("Loaded best model checkpoint from ")).name
                checkpoint = checkpoints.get(name)
                if checkpoint is None:
                    raise FileNotFoundError(f"logged checkpoint missing from evaluation output: {name}")
                key = (step, seed)
                if key in manifest:
                    raise RuntimeError(f"duplicate checkpoint provenance for step/seed {key}")
                manifest[key] = checkpoint
    expected_keys = {(step, seed) for step in STEPS for seed in SEEDS}
    if set(manifest) != expected_keys or len(set(manifest.values())) != 6:
        raise RuntimeError(f"incomplete or duplicate checkpoint manifest: {manifest}")
    return manifest


def audit_saved_sasrec(source: Path, evaluation_root: Path) -> dict[str, object]:
    expected_data = source / "data" / "Video_Games" / "5-core" / "downstream" / "data.txt"
    if not expected_data.is_file():
        raise FileNotFoundError(f"Games data missing through source link: {expected_data}")
    os.chdir(source)
    cwd_data = Path("data/Video_Games/5-core/downstream/data.txt")
    if not cwd_data.is_file():
        raise FileNotFoundError(f"Games data missing from evaluation cwd {Path.cwd()}: {cwd_data}")

    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(source))
    from seqrec.runner import Runner

    checkpoints = checkpoint_manifest(evaluation_root)
    embeddings = {}
    for step in STEPS:
        matches = [
            path
            for path in evaluation_root.rglob(f"*step{step}_title_item_embs.npy")
            if path.is_file()
        ]
        if len(matches) != 1:
            raise RuntimeError(f"expected one step-{step} embedding, found {len(matches)}")
        embeddings[step] = matches[0]
    configs = sorted(evaluation_root.rglob("results.txt"))
    config_by_step = {}
    for result in configs:
        match = re.search(r"step(500|1000)", str(result))
        if match:
            step = int(match.group(1))
            if step in config_by_step:
                raise RuntimeError(f"duplicate saved evaluation config for step {step}")
            config_by_step[step] = json.loads(
                (result.parent / "config.json").read_text(encoding="utf-8")
            )
    if set(config_by_step) != set(STEPS):
        raise RuntimeError(f"missing saved evaluation configs: {sorted(config_by_step)}")

    summary: dict[str, object] = {}
    for step in STEPS:
        seed_metrics = []
        for seed in SEEDS:
            config = dict(config_by_step[step])
            config.update({"embedding": str(embeddings[step]), "rand_seed": seed})
            runner = Runner(model_name="SASRec", config_dict=config)
            checkpoint = checkpoints[(step, seed)]
            runner.model.load_state_dict(torch.load(checkpoint, map_location=runner.config["device"]))
            test_loader = DataLoader(
                runner.recdata["test"], batch_size=runner.config["eval_batch_size"], shuffle=False
            )
            runner.model, test_loader = runner.accelerator.prepare(runner.model, test_loader)
            all_predictions = []
            all_labels = []
            runner.model.eval()
            for batch in test_loader:
                batch = {
                    key: value.to(runner.accelerator.device) if key != "seq_type" else value
                    for key, value in batch.items()
                }
                with torch.no_grad():
                    predictions = runner.model.predict(batch, n_return_sequences=20)
                all_predictions.append(predictions.detach().cpu().numpy())
                all_labels.append(batch["labels"].detach().cpu().numpy())
            predictions = np.concatenate(all_predictions)
            labels = np.concatenate(all_labels).reshape(-1)
            vectors = metric_vectors(predictions, labels)
            output = ROOT / f"raw_ranks_step{step}_seed{seed}.npz"
            np.savez_compressed(output, predictions=predictions, labels=labels, **vectors)
            metrics = {name: float(values.mean()) for name, values in vectors.items() if name != "rank"}
            metrics.update(
                {
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "raw_artifact": output.name,
                    "raw_sha256": sha256_file(output),
                    "users": int(labels.shape[0]),
                }
            )
            seed_metrics.append(metrics)
            runner.trainer.end()
            del runner
            gc.collect()
            torch.cuda.empty_cache()
        summary[str(step)] = {
            "per_seed": seed_metrics,
            "mean": {
                metric: float(np.mean([record[metric] for record in seed_metrics]))
                for metric in ("ndcg@10", "recall@10", "ndcg@20", "recall@20")
            },
            "population_std": {
                metric: float(np.std([record[metric] for record in seed_metrics]))
                for metric in ("ndcg@10", "recall@10", "ndcg@20", "recall@20")
            },
        }
        for metric, expected in EXPECTED[step].items():
            actual = summary[str(step)]["mean"][metric]
            if not np.isfinite(actual) or abs(actual - expected) > 1e-8:
                raise RuntimeError(
                    f"raw-rank recomputation mismatch at step {step} {metric}: {actual} != {expected}"
                )
    return summary


def patch_qwen2_bidirectional_mask() -> dict[str, str]:
    """Replace Qwen2's inherited causal mask with a padding-only SDPA mask."""
    distribution = importlib.metadata.distribution("llm2vec")
    path = Path(distribution.locate_file("llm2vec/models/bidirectional_qwen2.py"))
    if not path.is_file():
        raise ModuleNotFoundError(f"llm2vec Qwen2 implementation was not found: {path}")
    source = path.read_text(encoding="utf-8")
    original_hash = hashlib.sha256(source.encode()).hexdigest()
    if original_hash == QWEN2_PATCHED_SHA256:
        return {
            "path": str(path),
            "original_sha256": QWEN2_ORIGINAL_SHA256,
            "patched_sha256": original_hash,
            "already_patched": True,
        }
    if original_hash != QWEN2_ORIGINAL_SHA256:
        raise RuntimeError(f"unexpected pristine llm2vec Qwen2 source hash: {original_hash}")
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
    if patched_hash != QWEN2_PATCHED_SHA256:
        raise RuntimeError(f"unexpected patched llm2vec Qwen2 source hash: {patched_hash}")
    result = {
        "path": str(path),
        "original_sha256": original_hash,
        "patched_sha256": patched_hash,
    }
    return result


def audit_bidirectional_model() -> dict[str, object]:
    expected = {"llm2vec": "0.2.3", "transformers": "4.44.2"}
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", *(f"{k}=={v}" for k, v in expected.items())],
        check=True,
    )
    resolved = {name: importlib.metadata.version(name) for name in expected}
    if resolved != expected:
        raise RuntimeError(f"dependency mismatch: {resolved}")
    mask_patch = patch_qwen2_bidirectional_mask()

    import torch
    from llm2vec import LLM2Vec

    checkpoint = find_one(
        "checkpoint-1000",
        lambda path: path.is_dir()
        and "iem-stage2" in str(path)
        and (path / "model.safetensors").is_file(),
    )
    encoder = LLM2Vec.from_pretrained(
        checkpoint,
        enable_bidirectional=True,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model = encoder.model.eval().cuda()
    model_class = model.__class__.__name__
    attention_implementation = getattr(model.config, "_attn_implementation", None)
    vocab_size = int(model.config.vocab_size)
    prefix = [min(100, vocab_size - 1), min(200, vocab_size - 1)]
    input_a = torch.tensor([prefix + [min(300, vocab_size - 1)]], device="cuda")
    input_b = torch.tensor([prefix + [min(301, vocab_size - 1)]], device="cuda")
    mask = torch.ones_like(input_a)
    with torch.no_grad():
        hidden_a = model(input_ids=input_a, attention_mask=mask).last_hidden_state
        hidden_b = model(input_ids=input_b, attention_mask=mask).last_hidden_state
    prefix_delta = float((hidden_a[:, :2] - hidden_b[:, :2]).abs().max().item())
    attention_classes = sorted({layer.self_attn.__class__.__name__ for layer in model.layers})
    attention_is_causal = sorted({bool(layer.self_attn.is_causal) for layer in model.layers})
    passed = (
        model_class == "Qwen2BiModel"
        and attention_implementation == "sdpa"
        and attention_is_causal == [False]
        and prefix_delta > 1e-6
    )
    if not passed:
        raise RuntimeError(
            "bidirectional runtime proof failed: "
            f"class={model_class}, attention={attention_implementation}, "
            f"is_causal={attention_is_causal}, prefix_delta={prefix_delta}"
        )
    result = {
        "checkpoint": str(checkpoint),
        "model_class": model_class,
        "attention_implementation": attention_implementation,
        "future_token_changes_prefix_hidden_state_max_abs": prefix_delta,
        "attention_classes": attention_classes,
        "attention_is_causal": attention_is_causal,
        "passed": passed,
        "assertion": (
            "passed: future token changes prefix hidden state"
            if passed
            else "failed: Qwen2BiModel reports non-causal attention but inherited causal mask prevents prefix sensitivity"
        ),
        "dependencies": resolved,
        "qwen2_bidirectional_mask_patch": mask_patch,
    }
    del model, encoder, hidden_a, hidden_b
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    os.environ.update(
        {"CUDA_VISIBLE_DEVICES": "0", "WANDB_DISABLED": "true", "WANDB_MODE": "disabled"}
    )
    source, _, evaluation_root = prepare_evaluation_source()
    bidirectional = audit_bidirectional_model()
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    ranks = audit_saved_sasrec(source, evaluation_root)
    artifact = {
        "pipeline": "llm2rec-games-raw-rank-audit",
        "bidirectional": bidirectional,
        "raw_rank_recomputation": ranks,
    }
    (ROOT / "llm2rec_audit_artifact.json").write_text(
        json.dumps(artifact, indent=2), encoding="utf-8"
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
