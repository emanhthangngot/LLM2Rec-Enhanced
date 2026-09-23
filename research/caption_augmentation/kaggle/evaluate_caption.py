"""Extract budgeted IEM embeddings and run the released Games_5core SASRec eval."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path("/kaggle/working")
SOURCE_DIR = ROOT / "LLM2Rec"
SOURCE_COMMIT = "73b481f710f67166ab958f4985d27b27fb410871"
CONTRACT_VERSION = "2026-08-20.1"
DATASET_SLUG = "llm2rec-games-5core"
IEM_KERNEL = "trixuanle/llm2rec-caption-full-iem"
IEM_STEPS = (500, 1000)

SOURCE_HASHES = {
    "extract_llm_embedding.py": "29fd09df92736e15b521efa0d2a2f8407459ff0b571654f006a3cdab14c6eb85",
    "repeated_evaluate_with_seqrec.py": "d90a431af66f934217b6808a0d66cfba0d95e3223390920d28d016fe526c33e1",
}


def ensure_runtime_dependencies() -> dict[str, str]:
    expected = {"llm2vec": "0.2.3", "transformers": "4.44.2"}
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            *(f"{name}=={version}" for name, version in expected.items()),
        ],
        check=True,
    )
    resolved = {name: importlib.metadata.version(name) for name in expected}
    if resolved != expected:
        raise RuntimeError(f"Dependency resolution mismatch: expected {expected}, found {resolved}")
    return resolved


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
    """Prove future-token sensitivity before extracting item embeddings."""
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


def materialize_dataset(slug: str, expected_relative_root: str) -> Path:
    mounts = sorted(path for path in Path("/kaggle/input").rglob(slug) if path.is_dir())
    mount = mounts[0] if mounts else Path("/kaggle/input") / slug
    archive = next(iter(sorted(mount.rglob("data.zip"))), None) if mount.exists() else None
    search_root = mount
    if archive is not None:
        destination = ROOT / "dataset-unpacked" / slug
        if not destination.exists():
            destination.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(destination)
        search_root = destination
    candidate = search_root / expected_relative_root
    if candidate.is_dir():
        return candidate
    matches = sorted(search_root.glob(f"**/{expected_relative_root.split('/')[-2]}"))
    for match in matches:
        candidate = match / expected_relative_root.split("/")[-1]
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"{expected_relative_root} was not found in dataset {slug}")


def find_iem_checkpoint(step: int) -> Path:
    candidates = sorted(Path("/kaggle/input").rglob(f"checkpoint-{step}"))
    candidates = [path for path in candidates if (path / "config.json").is_file()]
    if not candidates:
        raise FileNotFoundError(f"IEM checkpoint-{step} was not found in Kaggle inputs")
    return candidates[-1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_provenance(games_root: Path) -> dict[str, dict[str, object]]:
    downstream = games_root / "downstream"
    required = ("train_data.txt", "val_data.txt", "test_data.txt", "item_titles.json")
    provenance: dict[str, dict[str, object]] = {}
    for name in required:
        path = downstream / name
        if not path.is_file():
            raise FileNotFoundError(f"required Games file missing: {path}")
        provenance[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return provenance


def validate_embedding_matrix(embedding: Path, games_root: Path) -> dict[str, object]:
    titles_path = games_root / "downstream" / "item_titles.json"
    titles = json.loads(titles_path.read_text(encoding="utf-8"))
    item_ids = sorted(int(item_id) for item_id in titles)
    expected_ids = list(range(1, item_ids[-1] + 1))
    if item_ids != expected_ids:
        raise RuntimeError("Games item IDs are not contiguous from 1 through the catalog maximum")

    matrix = np.load(embedding, mmap_mode="r")
    expected_rows = item_ids[-1] + 1  # row zero is the padding item
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows:
        raise RuntimeError(
            f"embedding shape mismatch: expected ({expected_rows}, dimension), found {matrix.shape}"
        )
    nan_count = int(np.isnan(matrix).sum())
    inf_count = int(np.isinf(matrix).sum())
    zero_rows = np.flatnonzero(np.all(matrix == 0, axis=1)).tolist()
    if nan_count or inf_count or zero_rows:
        raise RuntimeError(
            f"invalid embedding matrix: nan={nan_count}, inf={inf_count}, zero_rows={zero_rows}"
        )
    norms = np.linalg.norm(matrix.astype(np.float32), axis=1)
    return {
        "path": str(embedding),
        "sha256": sha256_file(embedding),
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "catalog_items": len(item_ids),
        "padding_row": 0,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "all_zero_row_count": len(zero_rows),
        "norm_min": float(norms.min()),
        "norm_mean": float(norms.mean()),
        "norm_max": float(norms.max()),
    }


def prepare_source() -> None:
    if not SOURCE_DIR.exists():
        archive_path = ROOT / "llm2rec-source.tar.gz"
        url = f"https://github.com/HappyPointer/LLM2Rec/archive/{SOURCE_COMMIT}.tar.gz"
        archive_path.write_bytes(urllib.request.urlopen(url, timeout=120).read())
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(ROOT)
        extracted = ROOT / f"LLM2Rec-{SOURCE_COMMIT}"
        extracted.rename(SOURCE_DIR)
    for name, expected_hash in SOURCE_HASHES.items():
        actual_hash = hashlib.sha256((SOURCE_DIR / name).read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(f"source hash mismatch for {name}: {actual_hash}")


def patch_extractor() -> Path:
    source = (SOURCE_DIR / "extract_llm_embedding.py").read_text(encoding="utf-8")
    replacements = {
        "print(token)": 'print("HF_TOKEN_CONFIGURED", bool(token))',
        "torch_dtype=torch.bfloat16": "torch_dtype=torch.float16",
        "use_auth_token=token,": 'use_auth_token=token,\n                attn_implementation="sdpa",',
    }
    for old, new in replacements.items():
        if old not in source:
            raise AssertionError(f"extractor patch target missing: {old}")
        source = source.replace(old, new)
    target = ROOT / "extract_llm_embedding_compat.py"
    target.write_text(source, encoding="utf-8")
    return target


def extract_embeddings(extractor: Path, games_root: Path, checkpoints: dict[int, Path]) -> dict[int, Path]:
    source_titles = games_root / "downstream" / "item_titles.json"
    if not source_titles.is_file():
        raise FileNotFoundError(f"Games item titles missing before linking: {source_titles}")
    data_link = SOURCE_DIR / "data"
    if data_link.exists() or data_link.is_symlink():
        if data_link.is_symlink() or data_link.is_file():
            data_link.unlink()
        else:
            shutil.rmtree(data_link)
    data_link.symlink_to(games_root.parent.parent)
    linked_titles = (
        data_link / "Video_Games" / "5-core" / "downstream" / "item_titles.json"
    )
    if not linked_titles.is_file():
        raise FileNotFoundError(f"Games item titles missing through source link: {linked_titles}")
    outputs: dict[int, Path] = {}
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(SOURCE_DIR), child_env.get("PYTHONPATH", "")) if path
    )
    for step, checkpoint in checkpoints.items():
        save_info = f"Qwen2-0.5B-LLM2Rec-IEM-budgeted_step{step}"
        subprocess.run(
            [
                sys.executable,
                str(extractor),
                "--dataset=Games_5core",
                f"--model_path={checkpoint}",
                "--item_prompt_type=title",
                "--bidirectional=1",
                "--batch_size=16",
                f"--save_info={save_info}",
            ],
            cwd=SOURCE_DIR,
            env=child_env,
            check=True,
        )
        output = SOURCE_DIR / "item_info" / "Games_5core" / f"{save_info}_title_item_embs.npy"
        if not output.is_file():
            raise FileNotFoundError(f"embedding output missing: {output}")
        outputs[step] = output
    return outputs


def evaluate_embedding(step: int, embedding: Path) -> list[str]:
    log_path = ROOT / f"games_sasrec_step{step}.log"
    command = [
        sys.executable,
        str(SOURCE_DIR / "repeated_evaluate_with_seqrec.py"),
        "--model=SASRec",
        "--dataset=Games_5core",
        "--lr=1.0e-3",
        "--weight_decay=1.0e-4",
        f"--embedding={embedding}",
        "--dropout=0.3",
        "--loss_type=ce",
        "--run_id=LLM2Rec-budgeted",
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=SOURCE_DIR, stdout=handle, stderr=subprocess.STDOUT, check=True)
    result_files = sorted((SOURCE_DIR / "Results").glob("Games_5core/SASRec/**/results.txt"))
    if not result_files:
        raise FileNotFoundError(f"SASRec results missing after checkpoint-{step}")
    return [str(path) for path in result_files[-1:]]


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    dependency_versions = ensure_runtime_dependencies()
    mask_patch = patch_qwen2_bidirectional_mask()
    prepare_source()
    games_root = materialize_dataset(DATASET_SLUG, "data/Video_Games/5-core")
    dataset_files = dataset_provenance(games_root)
    checkpoints = {step: find_iem_checkpoint(step) for step in IEM_STEPS}
    bidirectional_preflights = {
        step: validate_bidirectional_model(checkpoint) for step, checkpoint in checkpoints.items()
    }
    extractor = patch_extractor()
    embeddings = extract_embeddings(extractor, games_root, checkpoints)
    embedding_validations = {
        step: validate_embedding_matrix(path, games_root) for step, path in embeddings.items()
    }
    result_files = {step: evaluate_embedding(step, path) for step, path in embeddings.items()}
    artifact = {
        "pipeline": "llm2rec-budgeted-games-evaluation",
        "contract_version": CONTRACT_VERSION,
        "source_commit": SOURCE_COMMIT,
        "dataset_source": {"slug": DATASET_SLUG, "root": str(games_root), "files": dataset_files},
        "checkpoint_source": {"kernel": IEM_KERNEL, "paths": {str(step): str(path) for step, path in checkpoints.items()}},
        "checkpoints": {str(step): str(path) for step, path in checkpoints.items()},
        "embeddings": {str(step): str(path) for step, path in embeddings.items()},
        "embedding_validations": embedding_validations,
        "sasrec_result_files": result_files,
        "seeds": [2024, 2025, 2026],
        "full_catalog_scoring": True,
        "evaluation_protocol": {
            "candidate_scope": "all item IDs 1..item_num",
            "history_item_masking": False,
            "tie_breaking": "torch.topk implementation; no explicit tie-break rule in release",
            "metrics": ["Recall@10", "NDCG@10", "Recall@20", "NDCG@20"],
        },
        "domain_relation": (
            "in-domain: Games train interactions are included in AmazonMix-6; "
            "Games validation and test interactions remain downstream-only"
        ),
        "runtime_deviations": {
            "dtype": "float16",
            "attention": "sdpa",
            "hf_token": "not_required_for_public_Qwen2-0.5B",
        },
        "dependencies": dependency_versions,
        "qwen2_bidirectional_mask_patch": mask_patch,
        "bidirectional_preflights": bidirectional_preflights,
    }
    (ROOT / "games_evaluation_artifact.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
