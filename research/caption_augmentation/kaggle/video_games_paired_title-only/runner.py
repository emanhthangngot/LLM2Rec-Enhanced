"""Video_Games-only LLM2Rec chain for one paired arm (title-only or real).

One LLM chain per arm (CSFT -> MNTP -> SimCSE -> embedding extraction),
then the released SASRec evaluation over seeds 2024/2025/2026 on the full
Video_Games catalog. Both arms use identical splits, base model, optimizer
settings, history suffixes, and candidate set; only item text differs.

Resumable: every completed stage leaves a marker and model-only artifacts in
/kaggle/working. Re-push with this kernel as its own kernel_source to resume.
Pure helpers at the top are standard-library only and tested locally.
"""
from __future__ import annotations

import ast
import csv
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
from pathlib import Path
from typing import Callable, Mapping, Sequence

ARM = "title-only"
ARMS = ("title-only", "real")
CHAIN_SEED = 2024
# Set by the generator for resume pushes: fail fast instead of silently retraining CSFT.
RESUME_REQUIRED = False
SASREC_SEEDS = (2024, 2025, 2026)
CATALOG_SIZE = 9517
CUE_CAP_TOKENS = 32
CUTOFF_LEN = 1024
CSFT_MAX_STEPS = 1000
TITLE_PREFIX = "Title: "
CUE_SEPARATOR = "; Visual cues: "
UNAVAILABLE = "unavailable"

ROOT = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
SOURCE_DIR = ROOT / "LLM2Rec"
MODEL_DIR = ROOT / "qwen2-0.5b"
CSFT_DIR = ROOT / "csft-output"
STAGE1_DIR = ROOT / "iem-stage1"
STAGE2_DIR = ROOT / "iem-stage2"
MARKER_DIR = ROOT / "stage-markers"
KERNEL_START = time.time()
# Push timeout is 6 h; leave room to flush outputs.
CSFT_DEADLINE = KERNEL_START + 5.25 * 3600
POST_CSFT_MIN_SECONDS = 1.25 * 3600

BASE_MODEL_ID = "Qwen/Qwen2-0.5B"
BASE_MODEL_REVISION = "91d2aff3f957f99e4c74c962f2f408dcc88a18d8"
SOURCE_COMMIT = "73b481f710f67166ab958f4985d27b27fb410871"
SOURCE_HASHES = {
    "llm2rec/run_csft.py": "4ceb531f24c01ba7c18a00376f75fe3700a0834ec7bd78d918b0b70219b8020e",
    "llm2rec/dataset.py": "8cc542d80816faf66aae6ac80b45fc38c00fa1435b604488c96d73a2845fc714",
    "llm2rec/run_mntp.py": "b846c9ab270690037d7790d3078ae1e489797c376b98654522cf4b631566538a",
    "llm2rec/run_unsupervised_SimCSE.py": "d23d5781fab51bd80a2b1ec45c57b2ad0c2e29bd537810ce9e5e7eac8e69c7c9",
    "llm2rec/dataset_utils.py": "e4f4dd3c0bd49262cc67d215ae3e6e9aa5384a86739870d9f1ad6f6f3f02d03f",
    "llm2rec/train_mntp_config.json": "b4033d8fa431a006a931a8e5184ce624ba6648f6c88ccc3ee4d3b0f4bad2b61e",
    "llm2rec/train_simcse_config.json": "007b729d6fa28db36f91930ec2d88b9a24751ae1cf6e5187fe55ee91a8bb7465",
    "llm2rec/recdata/ItemTitleData.py": "db0cada8c22348dd751555053a50dd8bca175675fdb35c4b18a4bf3928290fee",
    "extract_llm_embedding.py": "29fd09df92736e15b521efa0d2a2f8407459ff0b571654f006a3cdab14c6eb85",
    "repeated_evaluate_with_seqrec.py": "d90a431af66f934217b6808a0d66cfba0d95e3223390920d28d016fe526c33e1",
}
# Pinned in results/video_games_pilot/video_games_dataset_contract.json.
DATASET_HASHES = {
    "train/Video_Games_5_1996-9-2023-10.csv": "87814fd96b6d593af6a26918537b4a42c26db17bcaa6a6bc824c113f3359f4ad",
    "valid/Video_Games_5_1996-9-2023-10.csv": "07393a1dc5a05c967532e80589d1c94d551060cd76f776314bd314c3499214a2",
    "test/Video_Games_5_1996-9-2023-10.csv": "f7d2195aa484f1335cc0d89c0d28633f549bfb3ec0dcd491c6a3e3243d240cbe",
    "downstream/train_data.txt": "1300e5deec29d4bede6e32bfa1a5eade563f53a6916da95d67754ec40b9475b8",
    "downstream/val_data.txt": "484a97cfbcff2a78e92612b15863ce08c84c285c7bfb325ec1f33d3884e6de16",
    "downstream/test_data.txt": "80074ffea37928d92c35038e9d8bbf3de0cfcf01308ae4173e3c680c1bc50d1f",
    "downstream/item_titles.json": "17f501809c4159905cfa6dfa5626d3d52d0be32bc9747ee89b003662f6f5e84c",
}
SASREC_ARGS = (
    "--model=SASRec", "--dataset=Games_5core", "--lr=1.0e-3", "--weight_decay=1.0e-4",
    "--dropout=0.3", "--loss_type=ce",
)


# ---------------------------------------------------------------- pure helpers

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def cap_cue(cue: str, cap: int, count_tokens: Callable[[str], int]) -> str:
    """Longest whole-word prefix of ``cue`` with at most ``cap`` tokens."""
    words = normalize_whitespace(cue).split()
    low, high, best = 0, len(words), ""
    while low <= high:
        mid = (low + high) // 2
        candidate = " ".join(words[:mid])
        if count_tokens(candidate) <= cap:
            best, low = candidate, mid + 1
        else:
            high = mid - 1
    return best


def build_item_texts(
    arm_records: Sequence[Mapping[str, object]],
    titles: Mapping[str, str],
    count_tokens: Callable[[str], int],
) -> dict[str, dict[int, str]]:
    """Return ``{arm: {local_id: text}}`` for both arms over the full catalog.

    Title-only text is the downstream title verbatim; real text fuses the
    token-capped cue, or ``unavailable`` when no usable caption exists.
    """
    if len(arm_records) != CATALOG_SIZE or len(titles) != CATALOG_SIZE:
        raise RuntimeError("paired corpus and downstream catalog must both cover 9517 items")
    texts: dict[str, dict[int, str]] = {arm: {} for arm in ARMS}
    for record in arm_records:
        local_id = int(record["local_item_id"])
        title = titles[str(local_id)]
        if normalize_whitespace(str(record["title"])) != normalize_whitespace(title):
            raise RuntimeError(f"corpus title differs from downstream title at item {local_id}")
        cue = record.get("real_cue")
        capped = cap_cue(str(cue), CUE_CAP_TOKENS, count_tokens) if cue else ""
        texts["title-only"][local_id] = title
        texts["real"][local_id] = f"{TITLE_PREFIX}{title}{CUE_SEPARATOR}{capped or UNAVAILABLE}"
    if sorted(texts["real"]) != list(range(1, CATALOG_SIZE + 1)):
        raise RuntimeError("item texts are not contiguous over 1..9517")
    return texts


def shared_history_suffix(
    real_history: Sequence[str], target: str, count_tokens: Callable[[str], int], cutoff: int = CUTOFF_LEN
) -> int:
    """Largest trailing item count whose *real-arm* CSFT sequence fits ``cutoff``.

    The real rendering is the longest, so applying the same count to the
    title-only arm keeps identical interaction history in both arms and
    prevents upstream left-truncation from cutting items mid-text.
    """
    budget = cutoff - count_tokens(target) - 2  # newline + EOS appended upstream
    if not real_history:
        return 0
    if count_tokens(", ".join(real_history)) <= budget:
        return len(real_history)
    low, high, best = 1, len(real_history) - 1, 1
    while low <= high:
        mid = (low + high) // 2
        if count_tokens(", ".join(real_history[-mid:])) <= budget:
            best, low = mid, mid + 1
        else:
            high = mid - 1
    return best


def csv_to_downstream_ids(
    csv_paths: Sequence[Path], arm_records: Sequence[Mapping[str, object]], titles: Mapping[str, str]
) -> dict[int, int]:
    """Map CSV item IDs (0-based, per the info file) to downstream IDs (1-based).

    The split CSVs and the downstream SASRec files use different permutations
    of the same 9,517 items. ASIN is the join key; a CSV item whose ASIN is
    absent from the corpus falls back to a unique title match. Every mapping
    must agree on title and the result must be a bijection.
    """
    pairs: dict[int, tuple[str, str]] = {}
    for path in csv_paths:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                ids = [int(v) for v in ast.literal_eval(row["history_item_id"])] + [int(row["item_id"])]
                asins = [str(v) for v in ast.literal_eval(row["item_asins"])] + [row["item_asin"]]
                names = [str(v) for v in ast.literal_eval(row["history_item_title"])] + [row["item_title"]]
                for item_id, asin, name in zip(ids, asins, names):
                    if pairs.setdefault(item_id, (asin, name)) != (asin, name):
                        raise RuntimeError(f"CSV item {item_id} has inconsistent ASIN/title")
    by_asin = {str(r["asin"]): int(r["local_item_id"]) for r in arm_records if r.get("asin")}
    by_title: dict[str, list[int]] = {}
    for key, name in titles.items():
        by_title.setdefault(normalize_whitespace(name), []).append(int(key))
    mapping: dict[int, int] = {}
    for item_id, (asin, name) in pairs.items():
        local = by_asin.get(asin)
        if local is None:
            candidates = by_title.get(normalize_whitespace(name), [])
            if len(candidates) != 1:
                raise RuntimeError(f"CSV item {item_id} ({asin}) has no ASIN or unique-title match")
            local = candidates[0]
        if normalize_whitespace(titles[str(local)]) != normalize_whitespace(name):
            raise RuntimeError(f"CSV item {item_id} maps to a downstream item with a different title")
        mapping[item_id] = local
    if len(mapping) != CATALOG_SIZE or sorted(mapping.values()) != list(range(1, CATALOG_SIZE + 1)):
        raise RuntimeError("CSV-to-downstream item map is not a bijection over the catalog")
    return mapping


def render_csft_csv(
    source: Path,
    destination: Path,
    arm: str,
    item_texts: Mapping[str, Mapping[int, str]],
    csv_to_local: Mapping[int, int],
    count_tokens: Callable[[str], int],
) -> dict[str, int]:
    """Rewrite history titles to arm texts with the shared suffix; keep targets."""
    stats = {"rows": 0, "history_items": 0, "history_items_dropped": 0}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8", newline="") as src, destination.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames or [])
        writer.writeheader()
        for row in reader:
            ids = [int(value) for value in ast.literal_eval(row["history_item_id"])]
            csv_titles = [str(value) for value in ast.literal_eval(row["history_item_title"])]
            if len(ids) != len(csv_titles) or not ids:
                raise RuntimeError(f"malformed history row for user {row['user_id']}")
            if any(item_id not in csv_to_local for item_id in ids):
                raise RuntimeError(f"history item outside the Video_Games catalog: {ids}")
            real = [item_texts["real"][csv_to_local[item_id]] for item_id in ids]
            keep = shared_history_suffix(real, str(row["item_title"]), count_tokens)
            rendered = csv_titles[-keep:] if arm == "title-only" else real[-keep:]
            row["history_item_title"] = repr(rendered)
            row["history_item_id"] = repr(ids[-keep:])
            writer.writerow(row)
            stats["rows"] += 1
            stats["history_items"] += keep
            stats["history_items_dropped"] += len(ids) - keep
    return stats


def parse_sasrec_results(text: str) -> list[dict[str, float]]:
    """Parse per-seed test metrics from repeated_evaluate_with_seqrec results.txt."""
    experiments: list[dict[str, float]] = []
    current: dict[str, float] | None = None
    for line in text.splitlines():
        if re.fullmatch(r"Experiment \d+:", line.strip()):
            current = {}
            experiments.append(current)
            continue
        match = re.fullmatch(r"([a-z]+@\d+): ([0-9.eE+-]+)", line.strip())
        if current is not None and match:
            current[match.group(1)] = float(match.group(2))
    if len(experiments) != len(SASREC_SEEDS) or any(not e for e in experiments):
        raise RuntimeError(f"expected {len(SASREC_SEEDS)} per-seed results, parsed {experiments}")
    return experiments


# ------------------------------------------------------------ Kaggle runtime

def log(message: str) -> None:
    print(f"[{(time.time() - KERNEL_START) / 60:7.1f} min] {message}", flush=True)


def marker(name: str) -> Path:
    return MARKER_DIR / f"{name}.json"


def write_marker(name: str, payload: Mapping[str, object]) -> None:
    MARKER_DIR.mkdir(parents=True, exist_ok=True)
    marker(name).write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def install_dependencies() -> dict[str, str]:
    import importlib.metadata

    pins = {"transformers": "4.44.2", "llm2vec": "0.2.3", "peft": "0.12.0"}
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *(f"{k}=={v}" for k, v in pins.items()),
         "datasets", "evaluate", "fire"],
        check=True,
    )
    resolved = {name: importlib.metadata.version(name) for name in pins}
    if resolved != pins:
        raise RuntimeError(f"dependency pins not honoured: {resolved}")
    return resolved


def patch_qwen2_bidirectional_mask() -> dict[str, str]:
    """Replace Qwen2's inherited causal mask with a padding-only SDPA mask."""
    import importlib.metadata

    path = Path(importlib.metadata.distribution("llm2vec").locate_file("llm2vec/models/bidirectional_qwen2.py"))
    source = path.read_text(encoding="utf-8")
    original = hashlib.sha256(source.encode()).hexdigest()
    marker_text = "        self.post_init()\n\n\nclass Qwen2BiForMNTP"
    if "def _update_causal_mask(\n        self,\n        attention_mask,\n        input_tensor," in source:
        return {"path": str(path), "status": "already-patched", "sha256": original}
    if marker_text not in source:
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
            raise ValueError(f"Bidirectional Qwen2 expects a 2D padding mask, found {attention_mask.shape}")
        dtype = input_tensor.dtype
        min_dtype = torch.finfo(dtype).min
        padding_mask = attention_mask[:, None, None, :].eq(0)
        padding_mask = padding_mask.expand(input_tensor.shape[0], 1, input_tensor.shape[1], attention_mask.shape[-1])
        return torch.zeros(padding_mask.shape, dtype=dtype, device=input_tensor.device).masked_fill(padding_mask, min_dtype)


class Qwen2BiForMNTP'''
    patched = source.replace(marker_text, method, 1)
    path.write_text(patched, encoding="utf-8")
    return {"path": str(path), "original_sha256": original, "patched_sha256": hashlib.sha256(patched.encode()).hexdigest()}


def validate_bidirectional_model(checkpoint: Path) -> dict[str, object]:
    """Prove future-token sensitivity before IEM training and extraction."""
    import torch
    from llm2vec import LLM2Vec

    encoder = LLM2Vec.from_pretrained(checkpoint, enable_bidirectional=True, torch_dtype=torch.float16, attn_implementation="sdpa")
    model = encoder.model.eval().cuda()
    input_a = torch.tensor([[100, 200, 300]], device="cuda")
    input_b = torch.tensor([[100, 200, 301]], device="cuda")
    mask = torch.ones_like(input_a)
    with torch.no_grad():
        delta = float(
            (model(input_ids=input_a, attention_mask=mask).last_hidden_state[:, :2]
             - model(input_ids=input_b, attention_mask=mask).last_hidden_state[:, :2]).abs().max().item()
        )
    result = {
        "model_class": model.__class__.__name__,
        "attention_is_causal": sorted({bool(layer.self_attn.is_causal) for layer in model.layers}),
        "future_token_changes_prefix_hidden_state_max_abs": delta,
    }
    if result["model_class"] != "Qwen2BiModel" or result["attention_is_causal"] != [False] or delta <= 1e-6:
        raise RuntimeError(f"Qwen2 bidirectional preflight failed: {result}")
    del model, encoder
    torch.cuda.empty_cache()
    return result


def prepare_source() -> None:
    if not SOURCE_DIR.exists():
        archive = ROOT / "llm2rec-source.tar.gz"
        archive.write_bytes(urllib.request.urlopen(
            f"https://github.com/HappyPointer/LLM2Rec/archive/{SOURCE_COMMIT}.tar.gz", timeout=120
        ).read())
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(ROOT)
        (ROOT / f"LLM2Rec-{SOURCE_COMMIT}").rename(SOURCE_DIR)
        archive.unlink()
    for name, expected in SOURCE_HASHES.items():
        if sha256_file(SOURCE_DIR / name) != expected:
            raise RuntimeError(f"pinned LLM2Rec source hash mismatch: {name}")


def download_base_model() -> None:
    if (MODEL_DIR / "model.safetensors").is_file():
        return
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=BASE_MODEL_ID, revision=BASE_MODEL_REVISION, local_dir=str(MODEL_DIR),
        allow_patterns=["config.json", "generation_config.json", "merges.txt", "model.safetensors",
                        "tokenizer.json", "tokenizer_config.json", "vocab.json"],
    )


def find_games_root() -> Path:
    matches = sorted(
        path.parent.parent for path in INPUT.rglob("item_titles.json")
        if path.parent.name == "downstream" and path.parent.parent.name == "5-core"
        and path.parent.parent.parent.name == "Video_Games"
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one mounted Video_Games/5-core, found {matches}")
    root = matches[0]
    for relative, expected in DATASET_HASHES.items():
        if sha256_file(root / relative) != expected:
            raise RuntimeError(f"Video_Games dataset hash mismatch: {relative}")
    return root


def load_paired_corpus() -> tuple[list[dict[str, object]], dict[str, object]]:
    summaries = sorted(INPUT.rglob("video_games_corpus_summary.json"))
    if len(summaries) != 1:
        raise FileNotFoundError(f"expected one Video_Games corpus summary, found {summaries}")
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    arms_path = summaries[0].parent / summary["arms_file"]
    if sha256_file(arms_path) != summary["arms_sha256"]:
        raise RuntimeError("paired corpus hash does not match its summary")
    records = [json.loads(line) for line in arms_path.read_text(encoding="utf-8").splitlines() if line]
    asin_count = attach_shard_asins(records, summaries[0].parent)
    return records, {"summary_path": str(summaries[0]), "arms_sha256": summary["arms_sha256"],
                     "identity": summary["identity"], "usable_real_cue_count": summary["usable_real_cue_count"],
                     "asins_from_verified_shards": asin_count}


def attach_shard_asins(records: list[dict[str, object]], corpus_root: Path) -> int:
    """Arm records predate the ``asin`` field; recover it from hash-pinned caption shards."""
    manifests = sorted(corpus_root.rglob("shard-manifest.json"))
    if len(manifests) != 1:
        raise FileNotFoundError(f"expected one shard manifest, found {manifests}")
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    if manifest.get("completion_status") != "complete":
        raise RuntimeError("caption shard manifest is not complete")
    shard_rows: dict[int, tuple[str, str]] = {}
    for shard in manifest["shards"]:
        path = manifests[0].parent / shard["path"]
        if sha256_file(path) != shard["sha256"]:
            raise RuntimeError(f"caption shard hash mismatch: {path.name}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                shard_rows[int(row["global_id"])] = (str(row["asin"]), str(row["title"]))
    asins: set[str] = set()
    for record in records:
        asin, title = shard_rows[int(record["global_id"])]
        if normalize_whitespace(title) != normalize_whitespace(str(record["title"])):
            raise RuntimeError(f"shard title differs from arm record at global_id {record['global_id']}")
        if record.get("asin") not in (None, asin):
            raise RuntimeError(f"arm record ASIN disagrees with shard at global_id {record['global_id']}")
        record["asin"] = asin
        asins.add(asin)
    if len(asins) != len(records) or len(shard_rows) != len(records):
        raise RuntimeError("shard ASINs are not a bijection over the paired corpus")
    return len(asins)


def restore_from_prior_output() -> list[str]:
    """Copy completed stage artifacts from this kernel's previous version."""
    restored = []
    for prior_marker_dir in sorted(INPUT.rglob("stage-markers")):
        prior_root = prior_marker_dir.parent
        for name in ("csft-output", "iem-stage1", "iem-stage2", "stage-markers", "arm-inputs"):
            source = prior_root / name
            target = ROOT / name
            if source.is_dir() and not target.exists():
                shutil.copytree(source, target)
                restored.append(name)
    return restored


def latest_checkpoint(directory: Path) -> Path | None:
    checkpoints = [p for p in directory.glob("checkpoint-*") if (p / "trainer_state.json").is_file()]
    return max(checkpoints, key=lambda p: int(p.name.split("-")[1]), default=None)


def build_csft_compat() -> Path:
    source = (SOURCE_DIR / "llm2rec" / "run_csft.py").read_text(encoding="utf-8")
    replacements = {
        "torch_dtype=torch.bfloat16,": 'torch_dtype=torch.float32,\n            attn_implementation="sdpa",',
        "bf16=True,": "fp16=True,",
        "max_steps=10000,": f"max_steps={CSFT_MAX_STEPS},",
        'evaluation_strategy="steps",': 'eval_strategy="steps",',
        "eval_steps=2000,": "eval_steps=1000,",
        "save_steps=2000,": "save_steps=100,",
        "save_total_limit=5,": "save_total_limit=2,",
        "load_best_model_at_end=True,": "load_best_model_at_end=False,",
        'report_to="wandb",': 'report_to="none",',
        "callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]": "callbacks = [_DeadlineCallback()]",
        "trainer.evaluate()": 'print("CSFT_SKIP_PRETRAIN_EVAL")',
        "model.save_pretrained(output_dir)": 'print("CSFT_CHECKPOINTS_PERSISTED")',
        "    from datasets import Dataset as HFDataset\n    hf_train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})\n    hf_train_dataset = hf_train_dataset.shuffle(seed=seed)\n\n    hf_val_dataset = HFDataset.from_dict({k: [v[k] for v in val_data] for k in val_data[0].keys()})": "    hf_train_dataset = train_data\n    hf_val_dataset = val_data",
    }
    for old, new in replacements.items():
        if old not in source:
            raise AssertionError(f"CSFT compatibility patch target missing: {old[:60]}")
        source = source.replace(old, new, 1)
    deadline_callback = '''
import time as _time
from transformers import TrainerCallback as _TrainerCallback


class _DeadlineCallback(_TrainerCallback):
    """Stop at a save boundary once the kernel deadline passes."""

    def on_step_end(self, args, state, control, **kwargs):
        if _time.time() >= float(os.environ["CSFT_DEADLINE"]) and state.global_step % args.save_steps == 0:
            control.should_save = True
            control.should_training_stop = True
        return control

'''
    source = source.replace("def train(", deadline_callback + "\ndef train(", 1)
    path = ROOT / "run_csft_compat.py"
    path.write_text(source, encoding="utf-8")
    return path


def run_csft(train_csv: Path, valid_csv: Path) -> dict[str, object]:
    if marker("csft").is_file():
        return json.loads(marker("csft").read_text(encoding="utf-8"))
    compat = build_csft_compat()
    resume = latest_checkpoint(CSFT_DIR)
    command = [
        sys.executable, str(compat), "--base_model", str(MODEL_DIR), "--train_file", str(train_csv),
        "--eval_file", str(valid_csv), "--output_dir", str(CSFT_DIR), "--category", "Video_Games",
        "--seed", str(CHAIN_SEED), "--train_from_scratch", "False", "--use_lora", "False",
        "--batch_size", "128", "--micro_batch_size", "1", "--num_epochs", "1",
        "--learning_rate", "3e-4", "--cutoff_len", str(CUTOFF_LEN),
    ]
    if resume is not None:
        command += ["--resume_from_checkpoint", str(resume)]
    env = {**os.environ, "CSFT_DEADLINE": str(CSFT_DEADLINE),
           "PYTHONPATH": str(SOURCE_DIR / "llm2rec")}
    started = time.time()
    log(f"CSFT start (resume={resume})")
    subprocess.run(command, check=True, env=env, cwd=ROOT)
    final = CSFT_DIR / f"checkpoint-{CSFT_MAX_STEPS}"
    elapsed = time.time() - started
    if not (final / "model.safetensors").is_file():
        log(f"CSFT stopped at deadline; latest={latest_checkpoint(CSFT_DIR)}")
        return {"status": "incomplete", "elapsed_seconds": elapsed}
    for child in CSFT_DIR.iterdir():
        if child != final:
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (final / name).unlink(missing_ok=True)
    result = {"status": "complete", "checkpoint": str(final), "elapsed_seconds": elapsed,
              "model_sha256": sha256_file(final / "model.safetensors"), "resumed_from": str(resume) if resume else None}
    write_marker("csft", result)
    return result


def stage_with_base_tokenizer(checkpoint: Path) -> Path:
    staged = ROOT / "csft-for-iem"
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir()
    tokenizer_files = {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                       "special_tokens_map.json", "added_tokens.json"}
    for source in checkpoint.iterdir():
        if source.name not in tokenizer_files:
            (staged / source.name).symlink_to(source)
    for name in tokenizer_files:
        if (MODEL_DIR / name).is_file():
            (staged / name).symlink_to(MODEL_DIR / name)
    return staged


def run_iem(item_text_file: Path) -> dict[str, object]:
    if marker("iem").is_file():
        return json.loads(marker("iem").read_text(encoding="utf-8"))
    staged = stage_with_base_tokenizer(CSFT_DIR / f"checkpoint-{CSFT_MAX_STEPS}")
    preflight = validate_bidirectional_model(staged)
    llm2rec_dir = SOURCE_DIR / "llm2rec"
    mntp = json.loads((llm2rec_dir / "train_mntp_config.json").read_text(encoding="utf-8"))
    mntp.pop("evaluation_strategy", None)
    mntp.update({
        "model_name_or_path": str(staged), "dataset_name": str(item_text_file), "output_dir": str(STAGE1_DIR),
        "torch_dtype": "float32", "attn_implementation": "sdpa", "eval_strategy": "steps", "fp16": True,
        "bf16": False, "per_device_train_batch_size": 8, "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 4, "report_to": "none", "logging_steps": 10, "save_steps": 500,
        "save_total_limit": 1, "save_only_model": True,
        # 9,517 domain texts give ~268 optimizer steps/epoch; 5 epochs let the
        # released 1000-step stop fire instead of ending at 3 default epochs.
        "num_train_epochs": 5, "seed": CHAIN_SEED,
    })
    mntp_path = ROOT / "train_mntp_video_games.json"
    mntp_path.write_text(json.dumps(mntp, indent=2), encoding="utf-8")
    started = time.time()
    log("MNTP start")
    subprocess.run([sys.executable, str(llm2rec_dir / "run_mntp.py"), str(mntp_path)], check=True, cwd=ROOT)
    stage1 = STAGE1_DIR / "checkpoint-1000"
    if not (stage1 / "config.json").is_file():
        raise FileNotFoundError(f"MNTP checkpoint-1000 missing: {stage1}")
    for child in STAGE1_DIR.iterdir():
        if child != stage1:
            shutil.rmtree(child) if child.is_dir() else child.unlink()

    simcse_data = ROOT / "arm-inputs" / "simcse-data"
    title_file = simcse_data / "AmazonMix-6" / "5-core" / "info" / "item_titles.txt"
    title_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(item_text_file, title_file)
    simcse = json.loads((llm2rec_dir / "train_simcse_config.json").read_text(encoding="utf-8"))
    simcse.update({
        "model_name_or_path": str(stage1), "dataset_file_path": str(simcse_data), "output_dir": str(STAGE2_DIR),
        "torch_dtype": "float32", "attn_implementation": "sdpa", "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 32, "gradient_accumulation_steps": 1, "fp16": True, "bf16": False,
        "report_to": "none", "lora_r": None, "save_steps": 500, "save_total_limit": 1,
        "save_only_model": True, "logging_steps": 10, "seed": CHAIN_SEED,
    })
    simcse_path = ROOT / "train_simcse_video_games.json"
    simcse_path.write_text(json.dumps(simcse, indent=2), encoding="utf-8")
    log("SimCSE start")
    subprocess.run([sys.executable, str(llm2rec_dir / "run_unsupervised_SimCSE.py"), str(simcse_path)], check=True, cwd=ROOT)
    stage2 = STAGE2_DIR / "checkpoint-1000"
    if not (stage2 / "config.json").is_file():
        raise FileNotFoundError(f"SimCSE checkpoint-1000 missing: {stage2}")
    for child in STAGE2_DIR.iterdir():
        if child != stage2:
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    shutil.rmtree(STAGE1_DIR)
    shutil.rmtree(CSFT_DIR)
    shutil.rmtree(staged)
    result = {"status": "complete", "checkpoint": str(stage2), "elapsed_seconds": time.time() - started,
              "bidirectional_preflight": preflight, "mntp_config": mntp, "simcse_config": simcse}
    write_marker("iem", result)
    return result


def run_evaluation(games_root: Path, item_titles: Mapping[int, str]) -> dict[str, object]:
    import numpy as np

    started = time.time()
    checkpoint = STAGE2_DIR / "checkpoint-1000"
    eval_root = ROOT / "eval-data" / "Video_Games" / "5-core" / "downstream"
    eval_root.mkdir(parents=True, exist_ok=True)
    for name in ("train_data.txt", "val_data.txt", "test_data.txt", "data.txt"):
        target = eval_root / name
        if not target.exists():
            target.symlink_to(games_root / "downstream" / name)
    titles_path = eval_root / "item_titles.json"
    titles_path.write_text(json.dumps({str(k): item_titles[k] for k in range(1, CATALOG_SIZE + 1)},
                                      ensure_ascii=False), encoding="utf-8")
    if ARM == "title-only":
        original = json.loads((games_root / "downstream" / "item_titles.json").read_text(encoding="utf-8"))
        if json.loads(titles_path.read_text(encoding="utf-8")) != original:
            raise RuntimeError("title-only arm must embed the unmodified downstream titles")
    data_link = SOURCE_DIR / "data"
    if data_link.is_symlink() or data_link.exists():
        data_link.unlink() if data_link.is_symlink() else shutil.rmtree(data_link)
    data_link.symlink_to(ROOT / "eval-data")

    extractor_source = (SOURCE_DIR / "extract_llm_embedding.py").read_text(encoding="utf-8")
    for old, new in {
        "print(token)": 'print("HF_TOKEN_CONFIGURED", bool(token))',
        "torch_dtype=torch.bfloat16": "torch_dtype=torch.float16",
        "use_auth_token=token,": 'use_auth_token=token,\n                attn_implementation="sdpa",',
    }.items():
        if old not in extractor_source:
            raise AssertionError(f"extractor patch target missing: {old}")
        extractor_source = extractor_source.replace(old, new)
    extractor = SOURCE_DIR / "extract_llm_embedding_compat.py"
    extractor.write_text(extractor_source, encoding="utf-8")
    preflight = validate_bidirectional_model(checkpoint)
    save_info = f"Qwen2-0.5B-VG-{ARM}"
    env = {**os.environ, "PYTHONPATH": str(SOURCE_DIR)}
    subprocess.run([sys.executable, str(extractor), "--dataset=Games_5core", f"--model_path={checkpoint}",
                    "--item_prompt_type=title", "--bidirectional=1", "--batch_size=16", f"--save_info={save_info}"],
                   cwd=SOURCE_DIR, env=env, check=True)
    embedding = SOURCE_DIR / "item_info" / "Games_5core" / f"{save_info}_title_item_embs.npy"
    matrix = np.load(embedding)
    zero_rows = np.flatnonzero(np.all(matrix == 0, axis=1)).tolist()
    if matrix.shape[0] != CATALOG_SIZE + 1 or not np.isfinite(matrix).all() or zero_rows:
        raise RuntimeError(f"invalid embedding: shape={matrix.shape}, zero_rows={zero_rows[:5]}")
    kept_embedding = ROOT / f"video_games_{ARM}_item_embs.npy"
    shutil.copyfile(embedding, kept_embedding)

    log("SASRec start")
    log_path = ROOT / f"sasrec_{ARM}.log"
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run([sys.executable, "repeated_evaluate_with_seqrec.py", *SASREC_ARGS,
                        f"--embedding={embedding}", f"--run_id=VG-paired-{ARM}"],
                       cwd=SOURCE_DIR, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    results = sorted((SOURCE_DIR / "Results").glob("Games_5core/SASRec/**/results.txt"))
    if len(results) != 1:
        raise RuntimeError(f"expected one SASRec results file, found {results}")
    per_seed = parse_sasrec_results(results[0].read_text(encoding="utf-8"))
    shutil.copyfile(results[0], ROOT / f"sasrec_{ARM}_results.txt")
    return {
        "per_seed": [{"seed": seed, **metrics} for seed, metrics in zip(SASREC_SEEDS, per_seed)],
        "embedding_sha256": sha256_file(kept_embedding), "embedding_shape": list(matrix.shape),
        "item_titles_sha256": sha256_file(titles_path), "bidirectional_preflight": preflight,
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    if ARM not in ARMS:
        raise RuntimeError(f"kernel was generated without a valid arm: {ARM!r}")
    os.environ.update({"CUDA_VISIBLE_DEVICES": "0", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                       "WANDB_DISABLED": "true", "WANDB_MODE": "disabled", "TOKENIZERS_PARALLELISM": "false"})
    restored = restore_from_prior_output()
    log(f"restored from prior output: {restored}")
    if RESUME_REQUIRED and not marker("csft").is_file():
        raise RuntimeError("resume push did not restore a completed CSFT stage; refusing to retrain")
    dependencies = install_dependencies()
    mask_patch = patch_qwen2_bidirectional_mask()
    prepare_source()
    download_base_model()
    games_root = find_games_root()
    corpus, corpus_provenance = load_paired_corpus()

    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("paired arm kernel requires CUDA")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    count_tokens = lambda text: len(tokenizer.encode(text, add_special_tokens=False))  # noqa: E731
    titles = json.loads((games_root / "downstream" / "item_titles.json").read_text(encoding="utf-8"))
    item_texts = build_item_texts(corpus, titles, count_tokens)
    inputs = ROOT / "arm-inputs"
    train_csv, valid_csv = inputs / "train.csv", inputs / "valid.csv"
    item_text_file = inputs / "item_texts.txt"
    if not marker("inputs").is_file():
        split_csvs = {split: games_root / split / "Video_Games_5_1996-9-2023-10.csv" for split in ("train", "valid", "test")}
        csv_to_local = csv_to_downstream_ids(list(split_csvs.values()), corpus, titles)
        render_stats = {
            "train": render_csft_csv(split_csvs["train"], train_csv, ARM, item_texts, csv_to_local, count_tokens),
            "valid": render_csft_csv(split_csvs["valid"], valid_csv, ARM, item_texts, csv_to_local, count_tokens),
            "csv_items_mapped_by_asin_or_unique_title": len(csv_to_local),
        }
        lines = [normalize_whitespace(item_texts[ARM][k]) for k in range(1, CATALOG_SIZE + 1)]
        item_text_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_marker("inputs", {
            "render_stats": render_stats, "train_csv_sha256": sha256_file(train_csv),
            "valid_csv_sha256": sha256_file(valid_csv), "item_texts_sha256": sha256_file(item_text_file),
            "real_texts_with_cue": sum(not t.endswith(CUE_SEPARATOR + UNAVAILABLE) for t in item_texts["real"].values()),
        })
    inputs_marker = json.loads(marker("inputs").read_text(encoding="utf-8"))
    log(f"inputs ready: {inputs_marker['render_stats']}")

    artifact: dict[str, object] = {
        "arm": ARM, "chain_seed": CHAIN_SEED, "sasrec_seeds": list(SASREC_SEEDS),
        "cuda_device": torch.cuda.get_device_name(0), "dependencies": dependencies,
        "qwen2_bidirectional_mask_patch": mask_patch, "source_commit": SOURCE_COMMIT,
        "base_model": {"id": BASE_MODEL_ID, "revision": BASE_MODEL_REVISION},
        "dataset_hashes": DATASET_HASHES, "corpus": corpus_provenance, "inputs": inputs_marker,
        "restored_stages": restored,
    }
    csft = run_csft(train_csv, valid_csv)
    artifact["csft"] = csft
    if csft["status"] != "complete" or time.time() - KERNEL_START > 6 * 3600 - POST_CSFT_MIN_SECONDS:
        artifact["status"] = "INCOMPLETE"
        log("stopping before IEM; re-push with this kernel as kernel_source to resume")
    else:
        artifact["iem"] = run_iem(item_text_file)
        artifact["evaluation"] = run_evaluation(games_root, item_texts[ARM])
        artifact["status"] = "PASS"
    artifact["kernel_seconds"] = time.time() - KERNEL_START
    (ROOT / f"video_games_paired_{ARM}_artifact.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in artifact.items() if k not in {"inputs"}}, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
