"""Generated HaNoRec Games exploratory 1-seed run.

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

"""Real data preparation for the HaNoRec x LLM2Rec CF-hardness smoke pipeline.

Every artifact loaded here is the genuine pinned LLM2Rec Games_5core release
(commit 73b481f710f67166ab958f4985d27b27fb410871): the frozen SASRec checkpoint
(seed 2024, IEM checkpoint-500 title embeddings), the real train/val/test_data.txt
sequences, item_titles.json, and the real Amazon image manifest produced by the
G1 visual preflight screen. No synthetic catalog, no invented item ids.

The SASRec reconstruction below is a verbatim functional port of
seqrec/base.py, seqrec/modules.py (TransformerEncoder_v2 subset),
seqrec/models/Embedding2.py, and seqrec/models/SASRec/_model.py from that same
pinned commit -- copied because Kaggle kernel_sources mount only output files,
never source code (rule://kaggle-mcp-experiments), so this kernel must stay
self-contained. Parameter names/shapes are kept identical to the original so
the pinned .pth state_dict loads without remapping.
"""

import copy
import hashlib
import json
import math
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------
# Verbatim-ported SASRec architecture (see module docstring for provenance).
# --------------------------------------------------------------------------

def _build_sasrec_model(config: dict[str, Any], pretrained_embeddings):
    import torch
    import torch.nn as nn

    class FeedForward(nn.Module):
        def __init__(self, hidden_size, inner_size, hidden_dropout_prob, layer_norm_eps=1e-12):
            super().__init__()
            self.dense_1 = nn.Linear(hidden_size, inner_size)
            self.dense_2 = nn.Linear(inner_size, hidden_size)
            self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
            self.dropout = nn.Dropout(hidden_dropout_prob)

        def forward(self, input_tensor):
            hidden_states = self.dense_1(input_tensor)
            hidden_states = hidden_states * 0.5 * (1.0 + torch.erf(hidden_states / math.sqrt(2.0)))
            hidden_states = self.dense_2(hidden_states)
            hidden_states = self.dropout(hidden_states)
            return self.LayerNorm(hidden_states + input_tensor)

    class MultiHeadAttention_v2(nn.Module):
        def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps=1e-12):
            super().__init__()
            if hidden_size % n_heads != 0:
                raise ValueError("hidden_size must be divisible by n_heads")
            self.num_attention_heads = n_heads
            self.attention_head_size = hidden_size // n_heads
            self.all_head_size = self.num_attention_heads * self.attention_head_size
            self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)
            self.query = nn.Linear(hidden_size, self.all_head_size)
            self.key = nn.Linear(hidden_size, self.all_head_size)
            self.value = nn.Linear(hidden_size, self.all_head_size)
            self.softmax = nn.Softmax(dim=-1)
            self.attn_dropout = nn.Dropout(attn_dropout_prob)
            self.dense = nn.Linear(hidden_size, hidden_size)
            self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
            self.out_dropout = nn.Dropout(hidden_dropout_prob)

        def _transpose(self, x):
            shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            return x.view(*shape)

        def forward(self, input_tensor, attention_mask):
            query_layer = self._transpose(self.query(input_tensor)).permute(0, 2, 1, 3)
            key_layer = self._transpose(self.key(input_tensor)).permute(0, 2, 3, 1)
            value_layer = self._transpose(self.value(input_tensor)).permute(0, 2, 1, 3)
            attention_scores = torch.matmul(query_layer, key_layer) / self.sqrt_attention_head_size
            attention_scores = attention_scores + attention_mask
            attention_probs = self.attn_dropout(self.softmax(attention_scores))
            context_layer = torch.matmul(attention_probs, value_layer).permute(0, 2, 1, 3).contiguous()
            new_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_shape)
            hidden_states = self.out_dropout(self.dense(context_layer))
            return self.LayerNorm(hidden_states + input_tensor)

    class TransformerLayer_v2(nn.Module):
        def __init__(self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob):
            super().__init__()
            self.multi_head_attention = MultiHeadAttention_v2(
                n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob
            )
            self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob)

        def forward(self, hidden_states, attention_mask):
            attention_output = self.multi_head_attention(hidden_states, attention_mask)
            return self.feed_forward(attention_output)

    class TransformerEncoder_v2(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            layer = TransformerLayer_v2(cfg["num_heads"], cfg["hidden_size"], 256, cfg["dropout"], cfg["dropout"])
            self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(cfg["layer_num"])])

        def forward(self, hidden_states, attention_mask):
            layers = []
            for layer_module in self.layer:
                hidden_states = layer_module(hidden_states, attention_mask)
                layers.append(hidden_states)
            return layers

    def get_attention_mask(item_seq):
        attention_mask = item_seq != 0
        extended = attention_mask.unsqueeze(1).unsqueeze(2)
        extended = torch.tril(extended.expand((-1, -1, item_seq.size(-1), -1)))
        return torch.where(extended, torch.tensor(0.0), torch.tensor(-10000.0))

    def gather_indexes(output, gather_index):
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        return output.gather(dim=1, index=gather_index).squeeze(1)

    class Embedding2Weight:
        __slots__ = ("data",)

        def __init__(self, data):
            self.data = data

    class Embedding2(nn.Module):
        def __init__(self, adapter, embedding):
            super().__init__()
            self.embedding = embedding
            self.adapter = adapter

        def forward(self, indices):
            return self.adapter(self.embedding(indices))

        @property
        def weight(self):
            return Embedding2Weight(self.adapter(self.embedding.weight.data))

    class SASRec(nn.Module):
        def __init__(self, cfg, pretrained_embs):
            super().__init__()
            self.config = cfg
            assert pretrained_embs.shape[0] == cfg["item_num"] + 1
            self.pretrained_item_embeddings = nn.Embedding.from_pretrained(pretrained_embs, padding_idx=0)
            self.pretrained_item_embeddings.weight.requires_grad = False
            assert cfg["adapter_dims"][-1] == -1
            mlp_dims = [pretrained_embs.shape[-1]] + list(cfg["adapter_dims"])
            mlp_dims[-1] = cfg["hidden_size"]
            self.item_embeddings_adapter = nn.Sequential()
            self.item_embeddings_adapter.add_module("linear_0", nn.Linear(mlp_dims[0], mlp_dims[1]))
            for index in range(1, len(mlp_dims) - 1):
                self.item_embeddings_adapter.add_module(f"activation_{index}", nn.ReLU())
                self.item_embeddings_adapter.add_module(f"linear_{index}", nn.Linear(mlp_dims[index], mlp_dims[index + 1]))
            self.item_embeddings = Embedding2(self.item_embeddings_adapter, self.pretrained_item_embeddings)
            self.positional_embeddings = nn.Embedding(cfg["max_seq_length"], cfg["hidden_size"])
            self.emb_dropout = nn.Dropout(cfg["dropout"])
            self.transformer_encoder = TransformerEncoder_v2(cfg)

        def get_embeddings(self, items):
            return self.item_embeddings(items)

        def get_all_embeddings(self):
            return self.item_embeddings.weight.data

        def get_representation(self, item_seqs, seq_lengths):
            inputs_emb = self.get_embeddings(item_seqs)
            inputs_emb = inputs_emb + self.positional_embeddings(torch.arange(self.config["max_seq_length"]))
            seq = self.emb_dropout(inputs_emb)
            mask = get_attention_mask(item_seqs)
            layers = self.transformer_encoder(seq, mask)
            output = layers[-1]
            return gather_indexes(output, seq_lengths - 1)

        def score_full_catalog(self, item_seqs, seq_lengths):
            state_hidden = self.get_representation(item_seqs, seq_lengths)
            test_item_emb = self.get_all_embeddings()
            select_pool = self.config["select_pool"]
            scores = torch.matmul(state_hidden, test_item_emb.transpose(0, 1))
            return scores[:, select_pool[0]:select_pool[1]]

    return SASRec(config, pretrained_embeddings)


def _read_sequences(path: Path) -> list[list[int]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [list(map(int, line.split())) for line in lines if line.strip()]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_one(root: Path, name: str) -> Path:
    matches = sorted(path for path in root.rglob(name) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"required real artifact not found under {root}: {name}")
    if len(matches) != 1:
        raise RuntimeError(
            f"ambiguous real artifact under {root}: {name}; "
            f"found {len(matches)} matches"
        )
    return matches[0]


def _verify_pin(path: Path, pins: dict[str, str]) -> None:
    expected = pins.get(path.name)
    if expected is None:
        return
    actual = _sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"artifact hash mismatch for {path}: expected {expected}, got {actual}")


def _future_items_for_prefix(history_and_target: list[int], all_rows: list[list[int]]) -> set[int]:
    """Return train-known future items for a stable sequence prefix."""
    future: set[int] = set()
    prefix_len = len(history_and_target)
    for row in all_rows:
        if len(row) > prefix_len and row[:prefix_len] == history_and_target:
            future.update(row[prefix_len:])
    return future


def _seeded_selection(
    candidates: list[tuple[int, list[int]]], count: int, seed: int, label: str
) -> list[tuple[int, list[int]]]:
    import random

    if count <= 0:
        raise ValueError(f"{label} count must be positive")
    if len(candidates) < count:
        raise RuntimeError(f"{label} has {len(candidates)} rows; need {count}")
    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(candidates)), count))
    return [candidates[index] for index in selected_indices]


def _build_frequency_matched_derangement(
    item_order: list[int], frequencies: dict[int, int], seed: int
) -> dict[int, int]:
    """Build a deterministic fixed-point-free bijection from train frequencies."""
    import random

    if len(item_order) < 2:
        raise RuntimeError("shuffle control requires at least two train items")
    groups: dict[int, list[int]] = {}
    for item_id in sorted(item_order):
        groups.setdefault(int(frequencies.get(item_id, 0)), []).append(int(item_id))
    ordered_groups = [groups[key] for key in sorted(groups)]
    merged: list[list[int]] = []
    for group in ordered_groups:
        if len(group) == 1 and merged:
            merged[-1].extend(group)
        elif len(group) == 1 and len(ordered_groups) > 1:
            ordered_groups[1].insert(0, group[0])
        else:
            merged.append(list(group))
    if merged and len(merged[-1]) == 1 and len(merged) > 1:
        merged[-2].extend(merged.pop())
    rng = random.Random(seed)
    mapping: dict[int, int] = {}
    for group in merged:
        group = sorted(group)
        if len(group) < 2:
            raise RuntimeError("frequency-stratified shuffle contains a singleton stratum")
        permutation = group[1:] + group[:1]
        if len(group) > 2:
            rng.shuffle(permutation)
            if any(source == target for source, target in zip(group, permutation)):
                permutation = group[1:] + group[:1]
        mapping.update(zip(group, permutation, strict=True))
    if set(mapping) != set(item_order) or len(set(mapping.values())) != len(item_order):
        raise RuntimeError("shuffle control is not a bijection")
    if any(item_id == mapped for item_id, mapped in mapping.items()):
        raise RuntimeError("shuffle control contains a fixed point")
    return mapping


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return _sha256_bytes(payload)


def _download_image(url: str, cache_dir: Path) -> tuple[str, str]:
    """Real download with retry-with-backoff for transient network errors.

    At real scale (thousands of items) some fraction of sequential HTTPS
    requests to Amazon's real image CDN hit transient resets/timeouts (real
    failure observed: `ConnectionResetError` on push
    `hanorec-cf-hardness-sft` v1) -- retrying a fresh connection recovers
    these without masking a genuinely broken URL (raises after exhausting
    retries, same as before for a persistent failure).
    """
    import time as _time

    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    suffix = Path(url).suffix or ".jpg"
    target = cache_dir / f"{digest}{suffix}"
    if not target.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "hanorec-cf-hardness-prep/1.0"})
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = response.read()
                if not payload:
                    raise RuntimeError(f"empty image payload for {url}")
                target.write_bytes(payload)
                last_error = None
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as error:
                last_error = error
                if attempt < 3:
                    _time.sleep(2.0 * (attempt + 1))
        if last_error is not None:
            raise RuntimeError(f"real image download failed after 4 attempts for {url}: {last_error}") from last_error
    return str(target), _sha256_file(target)




def prepare(config: dict[str, Any], input_root: Path, output_root: Path, source_root: Path) -> dict[str, Any]:
    """Prepare leakage-free train pairs and inference-only evaluation rows."""
    import numpy as np
    import torch

    output_root.mkdir(parents=True, exist_ok=True)
    pins: dict[str, str] = dict(config.get("artifact_pins", {}))
    checkpoint_path = _find_one(input_root, config["checkpoint_name"])
    embedding_path = _find_one(input_root, config["embedding_name"])
    manifest_path = _find_one(input_root, "Video_Games_image_manifest.jsonl")
    train_path = _find_one(input_root, "train_data.txt")
    val_path = _find_one(input_root, "val_data.txt")
    test_path = _find_one(input_root, "test_data.txt")
    full_data_path = _find_one(input_root, "data.txt")
    titles_path = _find_one(input_root, "item_titles.json")
    for path in (checkpoint_path, embedding_path, manifest_path, train_path, val_path, test_path, titles_path):
        _verify_pin(path, pins)
    if "data.txt" in pins:
        _verify_pin(full_data_path, pins)

    item_titles: dict[str, str] = json.loads(titles_path.read_text(encoding="utf-8"))
    manifest_by_item: dict[int, dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            manifest_by_item[int(record["item_id"])] = record
    usable_image_ids = {
        item_id
        for item_id, record in manifest_by_item.items()
        if record.get("status") == "ok" and record.get("url")
    }
    max_seq_length = int(config.get("max_seq_length", 10))
    raw_train_rows = _read_sequences(train_path)
    raw_val_rows = _read_sequences(val_path)
    raw_test_rows = _read_sequences(test_path)
    train_rows_full = [row for row in raw_train_rows if set(row) <= usable_image_ids]
    val_rows_full = [row for row in raw_val_rows if set(row) <= usable_image_ids]
    test_rows_full = [row for row in raw_test_rows if set(row) <= usable_image_ids]
    if not train_rows_full or not val_rows_full or not test_rows_full:
        raise RuntimeError("usable image manifest filtering removed an entire split")
    train_rows = [row[-(max_seq_length + 1):] for row in train_rows_full]
    val_rows = [row[-(max_seq_length + 1):] for row in val_rows_full]
    test_rows = [row[-(max_seq_length + 1):] for row in test_rows_full]

    embedding_matrix = np.load(embedding_path)
    total_item_num = int(embedding_matrix.shape[0] - 1)
    if total_item_num <= 0:
        raise RuntimeError("embedding matrix has no non-padding item rows")
    pretrained_embeddings = torch.tensor(embedding_matrix, dtype=torch.float32)
    sasrec_config = {
        "hidden_size": 128,
        "layer_num": 2,
        "num_heads": 2,
        "dropout": 0.3,
        "adapter_dims": [-1],
        "max_seq_length": max_seq_length,
        "item_num": total_item_num,
        "select_pool": [1, total_item_num + 1],
    }
    model = _build_sasrec_model(sasrec_config, pretrained_embeddings)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    def score_rows(rows: list[list[int]]) -> "np.ndarray":
        item_seqs: list[list[int]] = []
        seq_lengths: list[int] = []
        for row in rows:
            history = row[:-1]
            seq_length = len(history)
            if not 0 < seq_length <= max_seq_length:
                raise RuntimeError(f"invalid history length {seq_length} for max_seq_length={max_seq_length}")
            item_seqs.append(history + [0] * (max_seq_length - seq_length))
            seq_lengths.append(seq_length)
        with torch.no_grad():
            scores = model.score_full_catalog(
                torch.tensor(item_seqs, dtype=torch.long),
                torch.tensor(seq_lengths, dtype=torch.long),
            )
        score_array = scores.cpu().numpy()
        unusable_columns = [
            item_id - 1
            for item_id in range(1, total_item_num + 1)
            if item_id not in usable_image_ids
        ]
        if unusable_columns:
            score_array[:, unusable_columns] = -np.inf
        return score_array

    seed = int(config.get("seed", 2024))
    history_items = int(config.get("history_items", 3))
    train_pair_target = int(config.get("train_pairs", 8))
    eval_user_target = int(config.get("eval_users", 4))
    validation_user_target = int(config.get("validation_users", eval_user_target))
    top_m = int(config.get("top_m", 20))

    train_candidates = [(index, row) for index, row in enumerate(train_rows) if len(row) - 1 >= history_items]
    selected_train = _seeded_selection(train_candidates, train_pair_target, seed, "train")
    selected_train_scores = score_rows([row for _, row in selected_train])
    train_scope_rows = [row for _, row in selected_train] if config.get("smoke") else train_rows_full
    train_pairs: list[dict[str, Any]] = []
    train_frequencies: dict[int, int] = {}
    for row in train_scope_rows:
        for item in row:
            train_frequencies[item] = train_frequencies.get(item, 0) + 1
    for position, (row_index, row) in enumerate(selected_train):
        history_full = row[:-1]
        target = int(row[-1])
        history = history_full[-history_items:]
        future_excluded = _future_items_for_prefix(history_full + [target], train_rows_full)
        excluded_ids = set(history_full) | {target, 0} | future_excluded
        ranked = np.argsort(-selected_train_scores[position], kind="stable")
        negative_item = next(
            (int(rank_index) + 1 for rank_index in ranked if int(rank_index) + 1 not in excluded_ids),
            None,
        )
        if negative_item is None:
            raise RuntimeError(f"could not find a train-only negative for row {row_index}")
        train_pairs.append(
            {
                "user_id": f"train_row_{row_index}",
                "row_index": int(row_index),
                "history": [int(item) for item in history],
                "observed_history": [int(item) for item in history_full],
                "positive": target,
                "negative": negative_item,
                "cf_margin": float(selected_train_scores[position, target - 1] - selected_train_scores[position, negative_item - 1]),
                "excluded_items": sorted(int(item) for item in excluded_ids),
            }
        )

    def build_eval_rows(rows: list[list[int]], target_count: int, label: str) -> list[dict[str, Any]]:
        candidates = [(index, row) for index, row in enumerate(rows) if len(row) - 1 >= history_items]
        selected = _seeded_selection(candidates, target_count, seed + (1 if label == "validation" else 2), label)
        scores = score_rows([row for _, row in selected])
        built: list[dict[str, Any]] = []
        for position, (row_index, row) in enumerate(selected):
            top_indices = np.argsort(-scores[position], kind="stable")[:top_m]
            built.append(
                {
                    "user_id": f"{label}_row_{row_index}",
                    "row_index": int(row_index),
                    "history": [int(item) for item in row[:-1][-history_items:]],
                    "target": int(row[-1]),
                    "candidates": [int(index) + 1 for index in top_indices],
                    "retriever_scores": [float(scores[position, index]) for index in top_indices],
                    "split": label,
                }
            )
        return built

    validation_rows = build_eval_rows(val_rows, validation_user_target, "validation")
    evaluation_rows = build_eval_rows(test_rows, eval_user_target, "test")
    train_item_ids: set[int] = {int(item) for row in train_scope_rows for item in row}
    for pair in train_pairs:
        train_item_ids.update((pair["positive"], pair["negative"]))
    eval_item_ids: set[int] = set()
    for row in validation_rows + evaluation_rows:
        eval_item_ids.update(row["history"])
        eval_item_ids.add(row["target"])
        eval_item_ids.update(row["candidates"])
    all_referenced_items = train_item_ids | eval_item_ids


    def retriever_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        predictions = []
        for row in rows:
            ranked = list(row["candidates"])
            target = int(row["target"])
            rank = ranked.index(target) + 1 if target in ranked else 0
            predictions.append({
                "user_id": row["user_id"],
                "target": target,
                "rank": rank,
                "ndcg@10": (1.0 / math.log2(rank + 1)) if 0 < rank <= 10 else 0.0,
                "recall@10": 1.0 if 0 < rank <= 10 else 0.0,
                "candidate_recall@20": 1.0 if target in ranked else 0.0,
            })
        return predictions

    baselines = {
        "sasrec_validation": retriever_predictions(validation_rows),
        "sasrec_test": retriever_predictions(evaluation_rows),
    }
    image_cache_dir = output_root / "images"
    catalog: dict[str, dict[str, str]] = {}
    image_provenance: dict[str, dict[str, str]] = {}
    for item_id in sorted(all_referenced_items):
        title = item_titles.get(str(item_id))
        record = manifest_by_item.get(item_id)
        if title is None or record is None or item_id not in usable_image_ids:
            raise RuntimeError(f"real title/image manifest has no usable entry for item {item_id}")
        local_path, image_sha256 = _download_image(record["url"], image_cache_dir)
        catalog[str(item_id)] = {"title": title, "image_path": local_path}
        image_provenance[str(item_id)] = {
            "url": record["url"],
            "parent_asin": record.get("parent_asin", ""),
            "sha256": image_sha256,
            "source": "train" if item_id in train_item_ids else "evaluation",
        }
    shuffle_map = _build_frequency_matched_derangement(sorted(train_item_ids), train_frequencies, seed)
    train_contract = {
        "seed": seed,
        "train_pairs": train_pairs,
        "train_item_ids": sorted(train_item_ids),
        "train_frequencies": {str(k): int(v) for k, v in sorted(train_frequencies.items())},
        "shuffle_map": {str(k): int(v) for k, v in sorted(shuffle_map.items())},
    }
    provenance = {
        "seed": seed,
        "checkpoint": {"path": str(checkpoint_path), "sha256": _sha256_file(checkpoint_path)},
        "embedding": {"path": str(embedding_path), "sha256": _sha256_file(embedding_path)},
        "train_data": {"path": str(train_path), "sha256": _sha256_file(train_path)},
        "val_data": {"path": str(val_path), "sha256": _sha256_file(val_path)},
        "test_data": {"path": str(test_path), "sha256": _sha256_file(test_path)},
        "full_data": {"path": str(full_data_path), "sha256": _sha256_file(full_data_path)},
        "item_titles": {"path": str(titles_path), "sha256": _sha256_file(titles_path)},
        "manifest": {"path": str(manifest_path), "sha256": _sha256_file(manifest_path)},
        "total_item_num": total_item_num,
        "sasrec_config": sasrec_config,
        "train_item_ids": sorted(train_item_ids),
        "evaluation_only_item_ids": sorted(eval_item_ids - train_item_ids),
        "train_contract_sha256": _sha256_json(train_contract),
        "shuffle_map_sha256": _sha256_json(shuffle_map),
        "train_pair_construction": "seeded train rows; negative excludes observed history, target, and train-known future items",
        "eval_construction": "validation/test rows; frozen SASRec top-M candidates; targets never inserted",
        "shuffle_unseen_policy": "identity for evaluation-only items; train item mapping is complete and audited",
        "history_items": history_items,
        "image_manifest_usable_item_count": len(usable_image_ids),
        "image_manifest_unusable_item_ids": sorted(set(manifest_by_item) - usable_image_ids),
        "raw_train_row_count": len(raw_train_rows),
        "raw_validation_row_count": len(raw_val_rows),
        "raw_evaluation_row_count": len(raw_test_rows),
        "filtered_train_row_count": len(train_rows_full),
        "filtered_validation_row_count": len(val_rows_full),
        "filtered_evaluation_row_count": len(test_rows_full),
        "images_downloaded": len(image_provenance),
        "image_provenance": image_provenance,
    }
    return {
        "catalog": catalog,
        "train": train_pairs,
        "validation": validation_rows,
        "evaluation": evaluation_rows,
        "shuffle_map": shuffle_map,
        "train_item_ids": sorted(train_item_ids),
        "baselines": baselines,
        "provenance": provenance,
    }


"""Real Qwen2.5-VL SFT+DPO training with HaRS/CF hardness and NoDO noise.

Design decision (yes/no binary framing, matching upstream HaNoRec `--hit 1`):
for every real (history, positive, negative) pair from prep.py, two DPO
examples are built that SHARE the exact same prompt template (history titles
+ images, one candidate image/title, one yes/no question) and differ only in
which candidate is shown and which answer is therefore "chosen":

  - candidate = positive item -> chosen="Yes", rejected="No"
  - candidate = negative item -> chosen="No",  rejected="Yes"

This keeps chosen/rejected sharing identical prompt structure per pair while
reusing the paper's own binary next-item preference task, and it makes
reranking simple and real: for every eval candidate, run one forward pass of
the SAME prompt template and rank by score = logit("Yes") - logit("No").

HaRS semantic hardness and NoDO perturbation are ported verbatim (pure math
functions copied, not imported -- Kaggle kernel_sources mount output files
only, never source code, rule://kaggle-mcp-experiments) from the pinned
HaNoRec commit's hanorec/hars/math.py, hanorec/hars/hardness.py, and
hanorec/nodo/hooks.py.
"""

import hashlib
import json
import math
import os
import random
import time
from functools import lru_cache
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------
# Verbatim-ported HaRS math (hanorec/hars/math.py, commit 587face7).
# --------------------------------------------------------------------------

def _finite(values, name):
    checked = [float(v) for v in values]
    if not checked:
        raise ValueError(f"{name} must be non-empty")
    if not all(math.isfinite(v) for v in checked):
        raise ValueError(f"{name} must contain only finite values")
    return checked


def _stable_sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def hars_softmax(values):
    checked = _finite(values, "values")
    offset = max(checked)
    weights = [math.exp(v - offset) for v in checked]
    total = sum(weights)
    return [w / total for w in weights]


def hars_probability_distance(chosen, rejected):
    chosen_v = _finite(chosen, "chosen")
    rejected_v = _finite(rejected, "rejected")
    if len(chosen_v) != len(rejected_v):
        raise ValueError("chosen and rejected must have equal length")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(chosen_v, rejected_v)))


def hars_normalize_hardness(deltas):
    checked = _finite(deltas, "deltas")
    if any(d < 0 for d in checked):
        raise ValueError("deltas must be non-negative")
    denominator = _stable_sigmoid(sum(checked) / len(checked))
    return [_stable_sigmoid(d) / denominator for d in checked]


def cf_normalize_hardness(margins):
    """Frozen Phase 1 formula: lambda_cf = sigmoid(m) / sigmoid(mean(m)).

    Unlike hars_normalize_hardness, margins are signed (a retriever score
    difference), so this is intentionally not clamped to (0, 1); see
    plans/260918-1114-hanorec-cf-hardness/phase-01-local-spec-and-audit.md#2.
    """
    checked = _finite(margins, "margins")
    denominator = _stable_sigmoid(sum(checked) / len(checked))
    return [_stable_sigmoid(m) / denominator for m in checked]


def responsiveness(reward_gaps):
    """Torch-free port of hanorec/hars/math.py::model_responsiveness (Eq. 8)."""
    checked = _finite(reward_gaps, "reward_gaps")
    eps = 1e-8
    mean_gap = sum(checked) / len(checked)
    if abs(mean_gap) > eps:
        scale = mean_gap
    else:
        scale = max(sum(abs(v) for v in checked) / len(checked), eps)
    normalized = [v / scale for v in checked]
    trimmed = sorted(normalized)[1:-1] if len(normalized) > 2 else normalized
    trimmed_mean = sum(trimmed) / len(trimmed)
    normalized_mean = sum(normalized) / len(normalized)
    return _stable_sigmoid(trimmed_mean) / _stable_sigmoid(normalized_mean)


# --------------------------------------------------------------------------
# Prompt construction shared by SFT, DPO, and reranking.
# --------------------------------------------------------------------------

_QUESTION = "Based on the user's history, will they like this candidate item next? Answer Yes or No."


@lru_cache(maxsize=512)
def _resize_image(image_path: str, max_pixels: int):
    from PIL import Image

    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    width, height = image.size
    if width * height > max_pixels:
        scale = (max_pixels / (width * height)) ** 0.5
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
    return image


def _build_messages(catalog: dict[str, dict[str, str]], history: list[int], candidate: int, max_pixels: int, shuffle_map: dict[int, int] | None):
    def item_image(item_id: int):
        source_id = item_id if shuffle_map is None else shuffle_map.get(item_id, item_id)
        return _resize_image(catalog[str(source_id)]["image_path"], max_pixels)

    content: list[dict[str, Any]] = [{"type": "text", "text": "User history:"}]
    for history_item in history:
        content.append({"type": "image", "image": item_image(history_item)})
        content.append({"type": "text", "text": catalog[str(history_item)]["title"]})
    content.append({"type": "text", "text": "Candidate item:"})
    content.append({"type": "image", "image": item_image(candidate)})
    content.append({"type": "text", "text": catalog[str(candidate)]["title"]})
    content.append({"type": "text", "text": _QUESTION})
    return [{"role": "user", "content": content}]


def _semantic_embedding(model, processor, torch_module, title: str, image):
    tokenizer = processor.tokenizer
    inputs = tokenizer(title, return_tensors="pt", truncation=True)
    device = next(model.parameters()).device
    with torch_module.no_grad():
        text_states = model.model.embed_tokens(inputs["input_ids"].to(device))
        mask = inputs["attention_mask"].to(device).unsqueeze(-1).to(text_states.dtype)
        text_vector = (text_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

        image_inputs = processor.image_processor(images=[image], return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(device)
        grid_thw = image_inputs["image_grid_thw"].to(device)
        visual_dtype = next(model.visual.parameters()).dtype
        pixel_values = pixel_values.to(visual_dtype)
        visual_features = model.visual(pixel_values, grid_thw)
        visual_vector = visual_features.mean(dim=0, keepdim=True).to(text_vector.dtype)

    text_np = text_vector[0].float().cpu().numpy()
    visual_np = visual_vector[0].float().cpu().numpy()
    return text_np, visual_np


def _fuse_and_topk(text_matrix, visual_matrix, item_order: list[int], k: int):
    import numpy as np

    def normalize_rows(matrix):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)

    fused = normalize_rows(normalize_rows(text_matrix) + normalize_rows(visual_matrix))
    similarity = fused @ fused.T
    neighbors: dict[int, dict[str, list[float]]] = {}
    for row_index, item_id in enumerate(item_order):
        row = similarity[row_index].copy()
        row[row_index] = -np.inf
        top_k = min(k, len(item_order) - 1)
        selected = np.argpartition(-row, top_k - 1)[:top_k]
        selected = sorted(selected.tolist(), key=lambda index: (-float(row[index]), item_order[index]))
        neighbors[item_id] = {
            "item_ids": [item_order[index] for index in selected],
            "scores": [float(row[index]) for index in selected],
        }
    return neighbors


# --------------------------------------------------------------------------
# NoDO: real transient LoRA perturbation hooks (hanorec/nodo/hooks.py port).
def _iter_active_lora_pairs(model, adapter_names: set[str] | None = None):
    for module_name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if lora_a is None or lora_b is None:
            continue
        active = getattr(module, "active_adapters", None) or getattr(module, "active_adapter", None)
        if isinstance(active, str):
            active = [active]
        for adapter_name in active or []:
            if adapter_names is not None and adapter_name not in adapter_names:
                continue
            if adapter_name in lora_a and adapter_name in lora_b:
                yield module_name, adapter_name, lora_a[adapter_name], lora_b[adapter_name]


class _NodoPerturbation:
    def __init__(self, model, sigma: float, torch_module, functional_module, generator=None):
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        self._torch = torch_module
        self._functional = functional_module
        self._generator = generator
        self._sigma = float(sigma)
        self._pairs = list(_iter_active_lora_pairs(model, {"dpo"}))
        if not self._pairs:
            raise RuntimeError("No active DPO LoRA A/B module pair")
        self._handles: list[Any] = []

    def __enter__(self):
        torch_module = self._torch

        def make_hook(noise):
            def hook(_module, inputs, output):
                if not inputs:
                    raise RuntimeError("NoDO hook received no module input")
                return output + self._functional.linear(inputs[0], noise, None)

            return hook

        for _module_name, _adapter_name, lora_a, lora_b in self._pairs:
            kwargs = {"generator": self._generator} if self._generator is not None else {}
            noise_a = torch_module.randn(lora_a.weight.shape, device=lora_a.weight.device, dtype=lora_a.weight.dtype, **kwargs) * float(self._sigma)
            noise_b = torch_module.randn(lora_b.weight.shape, device=lora_b.weight.device, dtype=lora_b.weight.dtype, **kwargs) * float(self._sigma)
            self._handles.append(lora_a.register_forward_hook(make_hook(noise_a)))
            self._handles.append(lora_b.register_forward_hook(make_hook(noise_b)))
        return self

    def __exit__(self, *exc_info):
        for handle in reversed(self._handles):
            handle.remove()
        self._handles = []
        return False


def perturb_lora_weights(model, sigma: float, torch_module, functional_module, generator=None):
    return _NodoPerturbation(model, sigma, torch_module, functional_module, generator)


# --------------------------------------------------------------------------
# Shared model-call helpers (module-level, not closures, so both the
# single-push train_and_evaluate() and the resumable run_sft()/run_arm()
# split below can reuse identical logic without duplicating it).
# --------------------------------------------------------------------------

def _build_inputs(processor, policy_model, catalog, history: list[int], candidate: int, max_pixels: int, shuffle_map: dict[int, int] | None):
    messages = _build_messages(catalog, history, candidate, max_pixels, shuffle_map)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = [content["image"] for content in messages[0]["content"] if content["type"] == "image"]
    return processor(text=[text], images=images, return_tensors="pt").to(policy_model.device)


def _answer_logprobs(model, model_inputs, functional_module, yes_token_id: int, no_token_id: int) -> tuple:
    outputs = model(**model_inputs)
    logits = outputs.logits[:, -1, :]
    log_probs = functional_module.log_softmax(logits.float(), dim=-1)
    return log_probs[0, yes_token_id], log_probs[0, no_token_id]


def _single_token_id(tokenizer, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise RuntimeError(f"protocol requires {text!r} to encode to one token, got {token_ids!r}")
    return int(token_ids[0])
def _batch_answer_logprobs(model, model_inputs, functional_module, yes_token_id: int, no_token_id: int) -> list[tuple]:
    outputs = model(**model_inputs)
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is None:
        positions = [-1] * outputs.logits.shape[0]
    else:
        positions = [int(value) - 1 for value in attention_mask.sum(dim=1).tolist()]
    logits = outputs.logits[
        range(outputs.logits.shape[0]),
        positions,
        :,
    ]
    log_probs = functional_module.log_softmax(logits.float(), dim=-1)
    return [
        (log_probs[index, yes_token_id], log_probs[index, no_token_id])
        for index in range(log_probs.shape[0])
    ]


def _build_batch_inputs(processor, policy_model, catalog, examples, max_pixels, shuffle_map):
    messages = [
        _build_messages(catalog, example["history"], example["candidate"], max_pixels, shuffle_map)
        for example in examples
    ]
    texts = [
        processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        for message in messages
    ]
    images = [
        [content["image"] for content in message[0]["content"] if content["type"] == "image"]
        for message in messages
    ]
    return processor(text=texts, images=images, return_tensors="pt", padding=True).to(policy_model.device)


def _set_active_adapters(policy_model, adapters: str | list[str]) -> None:
    if not hasattr(policy_model, "set_adapter"):
        raise RuntimeError("PEFT model does not expose set_adapter")
    if isinstance(adapters, str):
        policy_model.set_adapter(adapters)
        return
    names = list(adapters)
    if not names:
        raise ValueError("at least one adapter must be active")
    # PEFT 0.15.2's PeftModel wrapper accepts one adapter name because
    # active_peft_config indexes peft_config with this scalar. Its underlying
    # LoraModel accepts the ordered list that composes SFT + DPO. Keep the
    # wrapper on the final (trainable) adapter while activating both below it.
    if len(names) == 1:
        policy_model.set_adapter(names[0])
        return
    base_model = getattr(policy_model, "base_model", None)
    if base_model is None or not hasattr(base_model, "set_adapter"):
        raise RuntimeError("PEFT model does not expose multi-adapter activation")
    base_model.set_adapter(names)
    policy_model.active_adapter = names[-1]


def _set_only_trainable_adapter(policy_model, adapter_name: str) -> None:
    trainable = []
    for name, parameter in policy_model.named_parameters():
        parameter.requires_grad = f".{adapter_name}." in name or name.endswith(f".{adapter_name}")
        if parameter.requires_grad:
            trainable.append(name)
    if not trainable:
        raise RuntimeError(f"{adapter_name} adapter has no trainable parameters")
def _restore_active_adapters(policy_model, adapters: str | list[str]) -> None:
    _set_active_adapters(policy_model, adapters)
    names = [adapters] if isinstance(adapters, str) else list(adapters)
    if "dpo" in names:
        _set_only_trainable_adapter(policy_model, "dpo")


def _reference_answer_logprobs(policy_model, model_inputs, torch_module, functional_module, yes_token_id: int, no_token_id: int) -> tuple:
    was_training = policy_model.training
    active = list(getattr(policy_model, "active_adapters", ["dpo"]))
    policy_model.eval()
    try:
        _set_active_adapters(policy_model, "sft")
        with torch_module.no_grad():
            values = _answer_logprobs(policy_model, model_inputs, functional_module, yes_token_id, no_token_id)
    finally:
        _restore_active_adapters(policy_model, active)
        if was_training:
            policy_model.train()
    return values

def _build_dpo_examples(train_pairs, yes_token_id: int, no_token_id: int, shuffle_map: dict[int, int] | None):
    examples = []
    for pair in train_pairs:
        examples.append(
            {
                "history": pair["history"],
                "candidate": pair["positive"],
                "chosen_token": yes_token_id,
                "rejected_token": no_token_id,
                "shuffle_map": shuffle_map,
            }
        )
        examples.append(
            {
                "history": pair["history"],
                "candidate": pair["negative"],
                "chosen_token": no_token_id,
                "rejected_token": yes_token_id,
                "shuffle_map": shuffle_map,
            }
        )
    return examples


def _load_base_model(config: dict[str, Any]):
    """Load the pinned model in the protocol precision without silent quantization."""
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if config.get("precision", "bf16") != "bf16":
        raise ValueError("only the protocol's bf16 precision is accepted")
    if not torch.cuda.is_available():
        raise RuntimeError("HaNoRec training requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("protocol requires native bf16 CUDA support; use an L4/A100-class GPU")
    dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(config["model_id"], revision=config["model_revision"])
    policy_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        device_map={"": 0},
        torch_dtype=dtype,
    )
    policy_model.eval()
    return processor, policy_model
def _lora_config():
    from peft import LoraConfig

    return LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )


def _wrap_sft_lora(policy_model):
    from peft import get_peft_model

    return get_peft_model(policy_model, _lora_config(), adapter_name="sft")


def _add_dpo_lora(policy_model):
    policy_model.add_adapter("dpo", _lora_config())
    _set_active_adapters(policy_model, ["sft", "dpo"])
    _set_only_trainable_adapter(policy_model, "dpo")
    return policy_model


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=_json_default), encoding="utf-8")
    temporary.replace(path)


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _seed_everything(seed: int, torch_module) -> None:
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def _reset_cuda_peak_memory(torch_module) -> None:
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA memory telemetry requires a visible GPU")
    torch_module.cuda.synchronize()
    torch_module.cuda.reset_peak_memory_stats()


def _cuda_peak_memory(torch_module) -> dict[str, Any]:
    torch_module.cuda.synchronize()
    device = torch_module.cuda.current_device()
    total = int(torch_module.cuda.get_device_properties(device).total_memory)
    allocated = int(torch_module.cuda.max_memory_allocated(device))
    reserved = int(torch_module.cuda.max_memory_reserved(device))
    return {
        "device": str(torch_module.cuda.get_device_name(device)),
        "total_bytes": total,
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "peak_allocated_fraction": allocated / total,
        "peak_reserved_fraction": reserved / total,
    }


def _chunked(values: list[Any], size: int, drop_last: bool) -> list[list[Any]]:
    if size <= 0:
        raise ValueError("batch_size must be positive")
    chunks = [values[start:start + size] for start in range(0, len(values), size)]
    return [chunk for chunk in chunks if len(chunk) == size or not drop_last]


def _accumulation_group_size(position: int, total_micro_batches: int, accumulation_steps: int) -> int:
    if accumulation_steps <= 0 or not 0 <= position < total_micro_batches:
        raise ValueError("invalid accumulation position or step count")
    group_start = (position // accumulation_steps) * accumulation_steps
    return min(accumulation_steps, total_micro_batches - group_start)


def _load_adapter_state(policy_model, state: dict[str, Any], adapter_name: str) -> None:
    current = dict(policy_model.named_parameters())
    expected = {
        name for name in current
        if f".{adapter_name}." in name or name.endswith(f".{adapter_name}")
    }
    if set(state) != expected:
        raise RuntimeError(f"{adapter_name} adapter tensor inventory mismatch")
    for name in expected:
        current[name].data.copy_(state[name].to(current[name].device, dtype=current[name].dtype))


def _adapter_state(policy_model, adapter_name: str) -> dict[str, Any]:
    return {
        name: value.detach().cpu().clone()
        for name, value in policy_model.named_parameters()
        if f".{adapter_name}." in name or name.endswith(f".{adapter_name}")
    }


def _preference_terms(
    examples, processor, policy_model, catalog, max_pixels, yes_token_id, no_token_id,
    torch_module, functional_module, *, shuffle_map, reference,
):
    inputs = _build_batch_inputs(processor, policy_model, catalog, examples, max_pixels, shuffle_map)
    if reference:
        was_training = policy_model.training
        active = list(getattr(policy_model, "active_adapters", ["dpo"]))
        policy_model.eval()
        try:
            _set_active_adapters(policy_model, "sft")
            with torch_module.no_grad():
                values = _batch_answer_logprobs(
                    policy_model, inputs, functional_module, yes_token_id, no_token_id
                )
        finally:
            _restore_active_adapters(policy_model, active)
            if was_training:
                policy_model.train()
    else:
        values = _batch_answer_logprobs(
            policy_model, inputs, functional_module, yes_token_id, no_token_id
        )
    terms = []
    for example, (yes_value, no_value) in zip(examples, values, strict=True):
        chosen = yes_value if example["chosen_token"] == yes_token_id else no_value
        rejected = no_value if example["chosen_token"] == yes_token_id else yes_value
        terms.append((chosen, rejected))
    return terms


def _score_rows(
    rows, processor, policy_model, catalog, max_pixels, yes_token_id, no_token_id,
    torch_module, functional_module, shuffle_map, eval_batch_size: int = 1,
):
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive")
    predictions = []
    was_training = policy_model.training
    policy_model.eval()
    with torch_module.no_grad():
        for eval_row in rows:
            candidate_scores = []
            examples = [
                {"history": eval_row["history"], "candidate": candidate}
                for candidate in eval_row["candidates"]
            ]
            for batch in _chunked(examples, eval_batch_size, False):
                inputs = _build_batch_inputs(
                    processor, policy_model, catalog, batch, max_pixels, shuffle_map
                )
                values = _batch_answer_logprobs(
                    policy_model, inputs, functional_module, yes_token_id, no_token_id
                )
                candidate_scores.extend(
                    float((yes_value - no_value).cpu())
                    for yes_value, no_value in values
                )
            ranked = sorted(
                zip(eval_row["candidates"], candidate_scores, strict=True),
                key=lambda pair: (-pair[1], int(pair[0])),
            )
            ranked_ids = [item for item, _score in ranked]
            target = int(eval_row["target"])
            rank = ranked_ids.index(target) + 1 if target in ranked_ids else 0
            predictions.append(
                {
                    "user_id": eval_row["user_id"],
                    "target": target,
                    "ranked_candidates": ranked_ids,
                    "rank": rank,
                    "ndcg@10": (1.0 / math.log2(rank + 1)) if 0 < rank <= 10 else 0.0,
                    "recall@10": 1.0 if 0 < rank <= 10 else 0.0,
                    "candidate_recall@20": 1.0 if target in eval_row["candidates"] else 0.0,
                }
            )
    if was_training:
        policy_model.train()
    return predictions


def _run_dpo_arm(
    config: dict[str, Any],
    processor,
    policy_model,
    catalog,
    train_pairs,
    validation_rows,
    evaluation_rows,
    lambda_sem_values,
    lambda_cf_values,
    item_order,
    weight: float,
    image_condition: str,
    yes_token_id: int,
    no_token_id: int,
    sft_state: dict,
    remaining,
    log_stage,
):
    import torch
    import torch.nn.functional as functional

    max_pixels = int(config.get("max_pixels", 262144))
    batch_size = int(config.get("batch_size", 4))
    accumulation_steps = int(config.get("gradient_accumulation_steps", 8))
    epochs = int(config.get("dpo_epochs", config.get("num_epochs", 5)))
    max_updates = config.get("max_optimizer_updates")
    max_updates = None if max_updates is None else int(max_updates)
    noise_sigma = float(config.get("noise_sigma", 0.05))
    beta0 = float(config.get("beta0", 0.1))
    drop_last = bool(config.get("drop_last", True))
    _load_adapter_state(policy_model, sft_state, "sft")
    _set_active_adapters(policy_model, ["sft", "dpo"])
    _set_only_trainable_adapter(policy_model, "dpo")
    pre_training_snapshot = {
        name: value.detach().clone()
        for name, value in policy_model.named_parameters()
        if value.requires_grad
    }
    examples = _build_dpo_examples(train_pairs, yes_token_id, no_token_id, None)
    arm_shuffle_map = None if image_condition == "real" else {
        int(key): int(value) for key, value in config["shuffle_map"].items()
    }
    lambda_combined = [
        float(lambda_sem_values[index]) ** float(weight)
        * float(lambda_cf_values[index]) ** (1.0 - float(weight))
        for index in range(len(train_pairs))
    ]
    hardness_per_example = [value for value in lambda_combined for _ in range(2)]
    batches = _chunked(examples, batch_size, drop_last)
    if not batches or len(batches[0]) < 3:
        raise RuntimeError("HaNoRec Eq. (8) requires a global mini-batch of at least 3 examples")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy_model.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]),
    )
    updates_per_epoch = max(1, math.ceil(len(batches) / accumulation_steps))
    scheduled_updates = epochs * updates_per_epoch
    expected_updates = min(
        scheduled_updates,
        max_updates if max_updates is not None else scheduled_updates,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, scheduled_updates))
    noise_generator = torch.Generator(device=next(policy_model.parameters()).device)
    noise_generator.manual_seed(int(config.get("seed", 2024)) + 991)
    step_losses = []
    optimizer_updates = 0
    micro_batches = 0
    sft_snapshot = _adapter_state(policy_model, "sft")
    reference_probe_terms = _preference_terms(
        batches[0], processor, policy_model, catalog, max_pixels, yes_token_id, no_token_id,
        torch, functional, shuffle_map=arm_shuffle_map, reference=True,
    )
    reference_probe = [float(value[0].detach().cpu()) for value in reference_probe_terms]
    reference_cache = {
        0: [
            (chosen.detach(), rejected.detach())
            for chosen, rejected in reference_probe_terms
        ]
    }
    for epoch in range(epochs):
        order = list(range(len(batches)))
        random.Random(int(config.get("seed", 2024)) + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        pending_examples = 0
        update_start = None
        for position, batch_index in enumerate(order):
            if remaining() <= 0 or (max_updates is not None and optimizer_updates >= max_updates):
                break
            if pending == 0:
                update_start = time.monotonic()
            batch = batches[batch_index]
            batch_positions = list(range(batch_index * batch_size, batch_index * batch_size + len(batch)))
            group_size = _accumulation_group_size(position, len(order), accumulation_steps)
            if batch_index not in reference_cache:
                reference_cache[batch_index] = [
                    (chosen.detach(), rejected.detach())
                    for chosen, rejected in _preference_terms(
                        batch, processor, policy_model, catalog, max_pixels,
                        yes_token_id, no_token_id, torch, functional,
                        shuffle_map=arm_shuffle_map, reference=True,
                    )
                ]
            reference_terms = reference_cache[batch_index]
            with perturb_lora_weights(
                policy_model, noise_sigma, torch, functional, noise_generator
            ):
                policy_terms = _preference_terms(
                    batch, processor, policy_model, catalog, max_pixels, yes_token_id, no_token_id,
                    torch, functional, shuffle_map=arm_shuffle_map, reference=False,
                )
                reward_gaps = [
                    beta0 * (
                        policy_terms[index][0].detach() - reference_terms[index][0]
                        - policy_terms[index][1].detach() + reference_terms[index][1]
                    )
                    for index in range(len(batch))
                ]
                if len(reward_gaps) < 3:
                    raise RuntimeError("global DPO batch has fewer than three examples")
                responsiveness_value = responsiveness(
                    [float(value.detach().cpu()) for value in reward_gaps]
                )
                betas = [
                    max(1e-6, beta0 * responsiveness_value * hardness_per_example[batch_positions[index]])
                    for index in range(len(batch))
                ]
                batch_loss = sum(
                    -functional.logsigmoid(
                        betas[index] * (
                            policy_terms[index][0] - reference_terms[index][0]
                            - policy_terms[index][1] + reference_terms[index][1]
                        )
                    )
                    for index in range(len(batch))
                ) / len(batch)
            if not torch.isfinite(batch_loss):
                raise RuntimeError("HaNoRec DPO loss is non-finite")
            (batch_loss / group_size).backward()
            step_losses.append(float(batch_loss.detach().cpu()))
            pending += 1
            pending_examples += len(batch)
            micro_batches += 1
            end_of_epoch = position == len(order) - 1
            if pending == accumulation_steps or end_of_epoch:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in policy_model.parameters() if parameter.requires_grad],
                    float(config.get("max_grad_norm", 1.0)),
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_updates += 1
                if update_start is None:
                    raise RuntimeError("DPO update timer was not initialized")
                log_stage(
                    f"dpo_update_w{weight}_{image_condition}",
                    time.monotonic() - update_start,
                    units=pending_examples,
                )
                pending = 0
                pending_examples = 0
                update_start = None
        if remaining() <= 0 or (max_updates is not None and optimizer_updates >= max_updates):
            break
    if optimizer_updates != expected_updates:
        raise RuntimeError(f"DPO optimizer update count mismatch: {optimizer_updates} != {expected_updates}")
    if max_updates is None and micro_batches != len(batches) * epochs:
        raise RuntimeError("DPO did not execute the expected micro-batch count")
    changed = any(
        not torch.equal(value, pre_training_snapshot[name])
        for name, value in policy_model.named_parameters()
        if value.requires_grad
    )
    if any(
        not torch.equal(value, sft_snapshot[name])
        for name, value in _adapter_state(policy_model, "sft").items()
    ):
        raise RuntimeError("SFT reference adapter changed during DPO")
    reference_after = _preference_terms(
        batches[0], processor, policy_model, catalog, max_pixels, yes_token_id, no_token_id,
        torch, functional, shuffle_map=arm_shuffle_map, reference=True,
    )
    if any(
        not math.isclose(reference_probe[index], float(value[0].detach().cpu()), abs_tol=1e-5)
        for index, value in enumerate(reference_after)
    ):
        raise RuntimeError("SFT reference changed after DPO updates")
    if not changed:
        raise RuntimeError("no DPO parameter changed after training")
    eval_batch_size = int(config.get("eval_batch_size", 1))
    validation_start = time.monotonic()
    validation_predictions = _score_rows(
        validation_rows, processor, policy_model, catalog, max_pixels,
        yes_token_id, no_token_id, torch, functional, arm_shuffle_map, eval_batch_size,
    )
    log_stage(
        f"validation_scoring_w{weight}_{image_condition}",
        time.monotonic() - validation_start,
        units=sum(len(row["candidates"]) for row in validation_rows),
    )
    evaluation_start = time.monotonic()
    predictions = _score_rows(
        evaluation_rows, processor, policy_model, catalog, max_pixels,
        yes_token_id, no_token_id, torch, functional, arm_shuffle_map, eval_batch_size,
    )
    log_stage(
        f"test_scoring_w{weight}_{image_condition}",
        time.monotonic() - evaluation_start,
        units=sum(len(row["candidates"]) for row in evaluation_rows),
    )
    scoring_batching_check = None
    if config.get("verify_scoring_batching") and eval_batch_size > 1:
        single_row_predictions = _score_rows(
            evaluation_rows[:1], processor, policy_model, catalog, max_pixels,
            yes_token_id, no_token_id, torch, functional, arm_shuffle_map, 1,
        )
        batched_row = predictions[0]
        single_row = single_row_predictions[0]
        scoring_batching_check = {
            "rows_checked": 1,
            "ranked_candidates_match": (
                batched_row["ranked_candidates"] == single_row["ranked_candidates"]
            ),
            "rank_match": batched_row["rank"] == single_row["rank"],
        }
        if not all(scoring_batching_check.values()):
            batched_scores = [
                pair for pair in zip(
                    single_row["ranked_candidates"],
                    batched_row["ranked_candidates"],
                    strict=True,
                ) if pair[0] != pair[1]
            ]
            scoring_batching_check["first_disagreements"] = batched_scores[:10]
            raise RuntimeError("batched candidate scoring is not rank-identical to unbatched")
    policy_model.train()
    return {
        "weight": weight,
        "image_condition": image_condition,
        "status": "COMPLETE",
        "seed": int(config.get("seed", 2024)),
        "scoring_batching_check": scoring_batching_check,
        "validation_predictions": validation_predictions,
        "predictions": predictions,
        "micro_batches": micro_batches,
        "optimizer_updates": optimizer_updates,
        "expected_optimizer_updates": expected_updates,
        "validation_mean_ndcg@10": sum(item["ndcg@10"] for item in validation_predictions) / len(validation_predictions),
        "losses": step_losses,
        "gpu_memory": _cuda_peak_memory(torch),
        "checkpoint_state": _adapter_state(policy_model, "dpo"),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_rng_state": random.getstate(),
        "noise_generator_state": noise_generator.get_state(),
        "train_item_order": [int(item) for item in item_order],
        "mean_ndcg@10": sum(item["ndcg@10"] for item in predictions) / len(predictions),
        "mean_recall@10": sum(item["recall@10"] for item in predictions) / len(predictions),
        "mean_candidate_recall@20": sum(item["candidate_recall@20"] for item in predictions) / len(predictions),
    }


def _compute_semantic_hardness(
    policy_model, processor, catalog, train_pairs, max_pixels: int, log_stage, train_item_ids,
    shuffle_map: dict[int, int] | None = None,
):
    """Compute HaRS hardness from the condition-specific train-only catalog."""
    import numpy as np
    import torch

    embedding_start = time.monotonic()
    item_order = sorted(int(item_id) for item_id in train_item_ids)
    if len(item_order) < 2:
        raise RuntimeError("train-only hardness catalog must contain at least two items")
    top_k = min(10, len(item_order) - 1)
    text_vectors = []
    visual_vectors = []
    for item_id in item_order:
        record = catalog[str(item_id)]
        source_id = item_id if shuffle_map is None else shuffle_map.get(item_id)
        if source_id is None:
            raise RuntimeError(f"shuffle map is missing train item {item_id}")
        source_record = catalog[str(source_id)]
        image = _resize_image(source_record["image_path"], max_pixels)
        text_vector, visual_vector = _semantic_embedding(policy_model, processor, torch, record["title"], image)
        text_vectors.append(text_vector)
        visual_vectors.append(visual_vector)
    log_stage("semantic_embedding", time.monotonic() - embedding_start, units=len(item_order))
    neighbors = _fuse_and_topk(np.stack(text_vectors), np.stack(visual_vectors), item_order, top_k)

    def pair_semantic_delta(positive_id: int, negative_id: int) -> float:
        positive_scores = neighbors[positive_id]["scores"]
        negative_scores = neighbors[negative_id]["scores"]
        return hars_probability_distance(hars_softmax(positive_scores), hars_softmax(negative_scores))

    semantic_deltas = [pair_semantic_delta(pair["positive"], pair["negative"]) for pair in train_pairs]
    lambda_sem_values = hars_normalize_hardness(semantic_deltas)
    cf_margins = [pair["cf_margin"] for pair in train_pairs]
    lambda_cf_values = cf_normalize_hardness(cf_margins)
    return lambda_sem_values, lambda_cf_values, item_order
# --------------------------------------------------------------------------

def run_sft(config: dict[str, Any], prepared: dict[str, Any], output_root: Path, hanorec_root: Path) -> dict[str, Any]:
    """Train the task-specific adapter and persist a strict portable parent."""
    import torch
    import torch.nn.functional as functional

    output_root.mkdir(parents=True, exist_ok=True)
    stage_timings: list[dict[str, Any]] = []

    def log_stage(name: str, seconds: float, units: int | None = None) -> None:
        payload: dict[str, Any] = {"stage": name, "seconds": float(seconds)}
        if units is not None:
            payload["units"] = int(units)
            payload["seconds_per_unit"] = float(seconds) / max(1, units)
        stage_timings.append(payload)
        print(json.dumps({"timing": payload}))

    start_time = time.monotonic()
    deadline = start_time + float(config.get("budget_seconds", 6600))
    remaining = lambda: deadline - time.monotonic()
    seed = int(config.get("seed", 2024))
    _seed_everything(seed, torch)
    _reset_cuda_peak_memory(torch)
    catalog = prepared["catalog"]
    train_pairs = prepared["train"]
    validation_rows = prepared.get("validation", [])
    evaluation_rows = prepared["evaluation"]
    train_item_ids = [int(item) for item in prepared["train_item_ids"]]
    image_condition = str(config.get("sft_image_condition", "real"))
    if image_condition not in {"real", "shuffle"}:
        raise ValueError(f"unsupported SFT image condition: {image_condition}")
    sft_shuffle_map = None if image_condition == "real" else {
        int(key): int(value) for key, value in prepared["shuffle_map"].items()
    }
    max_pixels = int(config.get("max_pixels", 262144))

    model_load_start = time.monotonic()
    processor, policy_model = _load_base_model(config)
    log_stage("model_load", time.monotonic() - model_load_start)
    yes_token_id = _single_token_id(processor.tokenizer, "Yes")
    no_token_id = _single_token_id(processor.tokenizer, "No")
    lambda_sem_values, lambda_cf_values, item_order = _compute_semantic_hardness(
        policy_model, processor, catalog, train_pairs, max_pixels, log_stage, train_item_ids,
        sft_shuffle_map,
    )
    policy_model = _wrap_sft_lora(policy_model)
    _set_active_adapters(policy_model, "sft")
    batch_size = int(config.get("sft_batch_size", config.get("batch_size", 4)))
    accumulation_steps = int(config.get("sft_gradient_accumulation_steps", config.get("gradient_accumulation_steps", 8)))
    epochs = int(config.get("sft_epochs", config.get("num_epochs", 5)))
    max_updates = config.get("sft_max_optimizer_updates", config.get("max_optimizer_updates"))
    max_updates = None if max_updates is None else int(max_updates)
    batches = _chunked(train_pairs, batch_size, bool(config.get("drop_last", True)))
    if not batches:
        raise RuntimeError("SFT has no complete mini-batches")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy_model.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]),
    )
    updates_per_epoch = max(1, math.ceil(len(batches) / accumulation_steps))
    scheduled_updates = epochs * updates_per_epoch
    expected_updates = min(
        scheduled_updates,
        max_updates if max_updates is not None else scheduled_updates,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, scheduled_updates))
    policy_model.train()
    losses: list[float] = []
    updates = 0
    micro_batches = 0
    for epoch in range(epochs):
        order = list(range(len(batches)))
        random.Random(seed + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        pending_examples = 0
        update_start = None
        for position, batch_index in enumerate(order):
            if remaining() <= 0 or (max_updates is not None and updates >= max_updates):
                break
            if pending == 0:
                update_start = time.monotonic()
            batch = batches[batch_index]
            sft_examples = [
                {"history": pair["history"], "candidate": pair["positive"]}
                for pair in batch
            ]
            inputs = _build_batch_inputs(
                processor, policy_model, catalog, sft_examples, max_pixels, sft_shuffle_map
            )
            sft_values = _batch_answer_logprobs(
                policy_model, inputs, functional, yes_token_id, no_token_id
            )
            batch_loss = -sum(yes_value for yes_value, _no_value in sft_values) / len(batch)
            if not torch.isfinite(batch_loss):
                raise RuntimeError("SFT loss is non-finite")
            group_size = _accumulation_group_size(position, len(order), accumulation_steps)
            (batch_loss / group_size).backward()
            losses.append(float(batch_loss.detach().cpu()))
            pending += 1
            pending_examples += len(batch)
            micro_batches += 1
            if pending == accumulation_steps or position == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in policy_model.parameters() if parameter.requires_grad],
                    float(config.get("max_grad_norm", 1.0)),
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                if update_start is None:
                    raise RuntimeError("SFT update timer was not initialized")
                log_stage("sft_update", time.monotonic() - update_start, units=pending_examples)
                pending = 0
                pending_examples = 0
                update_start = None
        if remaining() <= 0 or (max_updates is not None and updates >= max_updates):
            break
    if updates != expected_updates:
        raise RuntimeError(f"SFT optimizer update count mismatch: {updates} != {expected_updates}")
    sft_state = _adapter_state(policy_model, "sft")
    torch.save(
        {
            "adapter": sft_state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "completed_epochs": epochs if max_updates is None else None,
            "micro_batches": micro_batches,
            "optimizer_updates": updates,
            "seed": seed,
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        output_root / "sft_checkpoint.pt",
    )
    torch.save(sft_state, output_root / "sft_lora_state.pt")
    probe = _build_inputs(
        processor, policy_model, catalog, train_pairs[0]["history"], train_pairs[0]["positive"],
        max_pixels, sft_shuffle_map,
    )
    first = float(_reference_answer_logprobs(
        policy_model, probe, torch, functional, yes_token_id, no_token_id
    )[0].cpu())
    second = float(_reference_answer_logprobs(
        policy_model, probe, torch, functional, yes_token_id, no_token_id
    )[0].cpu())
    if not math.isclose(first, second, abs_tol=1e-5):
        raise RuntimeError("SFT reference is not deterministic")
    eval_batch_size = int(config.get("eval_batch_size", 1))
    validation_start = time.monotonic()
    sft_validation_predictions = _score_rows(
        validation_rows, processor, policy_model, catalog, max_pixels,
        yes_token_id, no_token_id, torch, functional, sft_shuffle_map, eval_batch_size,
    )
    log_stage(
        f"sft_validation_scoring_{image_condition}",
        time.monotonic() - validation_start,
        units=sum(len(row["candidates"]) for row in validation_rows),
    )
    evaluation_start = time.monotonic()
    sft_test_predictions = _score_rows(
        evaluation_rows, processor, policy_model, catalog, max_pixels,
        yes_token_id, no_token_id, torch, functional, sft_shuffle_map, eval_batch_size,
    )
    log_stage(
        f"sft_test_scoring_{image_condition}",
        time.monotonic() - evaluation_start,
        units=sum(len(row["candidates"]) for row in evaluation_rows),
    )
    portable_catalog = {
        item_id: {**record, "image_path": os.path.relpath(record["image_path"], output_root)}
        for item_id, record in catalog.items()
    }
    bundle = {
        "train": train_pairs,
        "validation": validation_rows,
        "evaluation": evaluation_rows,
        "catalog": portable_catalog,
        "item_order": item_order,
        "train_item_ids": train_item_ids,
        "image_condition": image_condition,
        "shuffle_map": {str(key): int(value) for key, value in prepared["shuffle_map"].items()},
        "condition_map_sha256": hashlib.sha256(
            json.dumps({str(key): int(value) for key, value in prepared["shuffle_map"].items()}, sort_keys=True).encode()
        ).hexdigest(),
        "lambda_sem": lambda_sem_values,
        "lambda_cf": lambda_cf_values,
        "hardness_input_sha256": hashlib.sha256(
            json.dumps({"train": train_pairs, "item_order": item_order, "image_condition": image_condition}, sort_keys=True).encode()
        ).hexdigest(),
        "sft_validation_predictions": sft_validation_predictions,
        "sft_test_predictions": sft_test_predictions,
        "sft_losses": losses,
        "baselines": prepared.get("baselines", {}),
        "sft_micro_batches": micro_batches,
        "sft_optimizer_updates": updates,
        "sft_state_sha256": hashlib.sha256((output_root / "sft_lora_state.pt").read_bytes()).hexdigest(),
        "stage_timings": stage_timings,
        "gpu_memory": _cuda_peak_memory(torch),
        "reference_reproducibility_check": {"first": first, "second": second},
        "provenance": prepared["provenance"],
        "status": "COMPLETE",
    }
    _atomic_json_write(output_root / "sft_bundle.json", bundle)
    return bundle


def run_arm(
    config: dict[str, Any],
    sft_dir: Path,
    weight: float,
    image_condition: str,
    output_root: Path,
    hanorec_root: Path,
) -> dict[str, Any]:
    """Resume a frozen SFT parent and train exactly one fresh DPO arm."""
    import torch

    output_root.mkdir(parents=True, exist_ok=True)
    stage_timings: list[dict[str, Any]] = []

    def log_stage(name: str, seconds: float, units: int | None = None) -> None:
        payload = {"stage": name, "seconds": float(seconds)}
        if units is not None:
            payload.update({"units": int(units), "seconds_per_unit": float(seconds) / max(1, units)})
        stage_timings.append(payload)
        print(json.dumps({"timing": payload}))

    start_time = time.monotonic()
    deadline = start_time + float(config.get("budget_seconds", 6600))
    remaining = lambda: deadline - time.monotonic()
    _seed_everything(int(config.get("seed", 2024)), torch)
    _reset_cuda_peak_memory(torch)
    bundle = json.loads((sft_dir / "sft_bundle.json").read_text(encoding="utf-8"))
    if bundle.get("status") != "COMPLETE":
        raise RuntimeError("cannot resume from incomplete SFT bundle")
    if bundle.get("image_condition") != image_condition:
        raise RuntimeError("arm image condition does not match its SFT parent")
    sft_state_path = sft_dir / "sft_lora_state.pt"
    parent_hash = hashlib.sha256(sft_state_path.read_bytes()).hexdigest()
    if parent_hash != bundle.get("sft_state_sha256"):
        raise RuntimeError("SFT parent hash mismatch before model construction")
    train_pairs = bundle["train"]
    evaluation_rows = bundle["evaluation"]
    catalog = {
        item_id: {**record, "image_path": str(sft_dir / record["image_path"])}
        for item_id, record in bundle["catalog"].items()
    }
    item_order = [int(item) for item in bundle["item_order"]]
    config = {**config, "shuffle_map": bundle["shuffle_map"]}
    model_load_start = time.monotonic()
    processor, policy_model = _load_base_model(config)
    policy_model = _wrap_sft_lora(policy_model)
    policy_model = _add_dpo_lora(policy_model)
    log_stage("model_load", time.monotonic() - model_load_start)
    yes_token_id = _single_token_id(processor.tokenizer, "Yes")
    no_token_id = _single_token_id(processor.tokenizer, "No")
    sft_state = torch.load(
        sft_state_path,
        map_location=next(policy_model.parameters()).device,
        weights_only=True,
    )
    arm_result = _run_dpo_arm(
        config, processor, policy_model, catalog, train_pairs, bundle["validation"], evaluation_rows,
        bundle["lambda_sem"], bundle["lambda_cf"], item_order, weight, image_condition,
        yes_token_id, no_token_id, sft_state, remaining, log_stage,
    )
    checkpoint_state = arm_result.pop("checkpoint_state")
    optimizer_state = arm_result.pop("optimizer_state")
    scheduler_state = arm_result.pop("scheduler_state")
    torch_rng_state = arm_result.pop("torch_rng_state")
    cuda_rng_state = arm_result.pop("cuda_rng_state")
    python_rng_state = arm_result.pop("python_rng_state")
    noise_generator_state = arm_result.pop("noise_generator_state")
    checkpoint_path = output_root / f"arm_w{weight}_{image_condition}.pt"
    torch.save(
        {
            "adapter": checkpoint_state,
            "optimizer": optimizer_state,
            "scheduler": scheduler_state,
            "torch_rng_state": torch_rng_state,
            "seed": arm_result["seed"],
            "cuda_rng_state": cuda_rng_state,
            "parent_sft_sha256": parent_hash,
            "weight": weight,
            "python_rng_state": python_rng_state,
            "noise_generator_state": noise_generator_state,
            "image_condition": image_condition,
            "optimizer_updates": arm_result["optimizer_updates"],
            "status": arm_result["status"],
        },
        checkpoint_path,
    )
    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if reloaded.get("parent_sft_sha256") != parent_hash:
        raise RuntimeError("reloaded checkpoint parent hash mismatch")
    if set(reloaded.get("adapter", {})) != set(checkpoint_state):
        raise RuntimeError("reloaded DPO adapter inventory mismatch")
    if any(not torch.equal(reloaded["adapter"][name], checkpoint_state[name]) for name in checkpoint_state):
        raise RuntimeError("reloaded DPO adapter tensor mismatch")
    arm_result.update(
        {
            "checkpoint": str(checkpoint_path.relative_to(output_root)),
            "stage_timings": stage_timings,
            "elapsed_seconds": time.monotonic() - start_time,
            "parent_sft_sha256": parent_hash,
            "checkpoint_reloaded": True,
        }
    )
    if arm_result["status"] != "COMPLETE":
        raise RuntimeError("incomplete arm cannot be emitted as COMPLETE")
    _atomic_json_write(output_root / "arm_result.json", arm_result)
    return arm_result


def run_resource_probe(config: dict[str, Any], prepared: dict[str, Any], output_root: Path) -> dict[str, Any]:
    """Measure real forward+backward memory/time across batch and image budgets.

    This is a planning instrument, never a scientific arm: it performs no
    optimizer step and emits no predictions.
    """
    import torch

    candidates = config.get("probe_candidates")
    if not candidates:
        raise ValueError("resource probe requires probe_candidates")
    output_root.mkdir(parents=True, exist_ok=True)
    _seed_everything(int(config.get("seed", 2024)), torch)
    processor, policy_model = _load_base_model(config)
    policy_model = _wrap_sft_lora(policy_model)
    _set_active_adapters(policy_model, "sft")
    policy_model.train()
    catalog = prepared["catalog"]
    train_pairs = prepared["train"]
    if not train_pairs:
        raise RuntimeError("resource probe requires at least one training pair")
    results = []
    for candidate in candidates:
        batch_size = int(candidate["batch_size"])
        max_pixels = int(candidate["max_pixels"])
        record: dict[str, Any] = {
            "batch_size": batch_size,
            "max_pixels": max_pixels,
            "images_per_example": 1 + int(config.get("history_items", 3)),
        }
        record["images_per_batch"] = record["images_per_example"] * batch_size
        record["pixel_budget_per_batch"] = record["images_per_batch"] * max_pixels
        examples = [
            {"history": pair["history"], "candidate": pair["positive"]}
            for pair in train_pairs[:batch_size]
        ]
        if len(examples) < batch_size:
            record.update({"ok": False, "error": "insufficient_train_pairs"})
            results.append(record)
            continue
        policy_model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        _reset_cuda_peak_memory(torch)
        inputs = None
        outputs = None
        loss = None
        try:
            inputs = _build_batch_inputs(
                processor, policy_model, catalog, examples, max_pixels, None
            )
            torch.cuda.synchronize()
            started = time.monotonic()
            outputs = policy_model(**inputs)
            loss = outputs.logits[:, -1, :].float().mean()
            loss.backward()
            torch.cuda.synchronize()
            record.update(
                {
                    "ok": True,
                    "forward_backward_seconds": time.monotonic() - started,
                    "gpu_memory": _cuda_peak_memory(torch),
                }
            )
        except torch.cuda.OutOfMemoryError:
            record.update({"ok": False, "error": "cuda_out_of_memory"})
        finally:
            del inputs, outputs, loss
            policy_model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        results.append(record)
    manifest = {
        "purpose": "resource_planning_only_non_signal",
        "must_not_be_used_as_signal": True,
        "seed": int(config.get("seed", 2024)),
        "candidates": results,
        "status": "COMPLETE",
    }
    _atomic_json_write(output_root / "resource_probe.json", manifest)
    return manifest


# --------------------------------------------------------------------------
# Single-push entry point (kept for runs small enough to fit one real
# Kaggle session -- the technical smoke test and the calibration run both
# used this path; see phase-02-full-dataset-scaling.md for when to switch
# to run_sft()/run_arm() instead).
# --------------------------------------------------------------------------

def train_and_evaluate(config: dict[str, Any], prepared: dict[str, Any], output_root: Path, hanorec_root: Path) -> dict[str, Any]:
    """Run matched real/shuffle SFT parents and all configured DPO arms."""
    sft_bundles: dict[str, dict[str, Any]] = {}
    sft_dirs: dict[str, Path] = {}
    for image_condition in ("real", "shuffle"):
        sft_dir = output_root / f"sft_{image_condition}"
        condition_config = {**config, "sft_image_condition": image_condition}
        sft_bundles[image_condition] = run_sft(condition_config, prepared, sft_dir, hanorec_root)
        sft_dirs[image_condition] = sft_dir
    arms = []
    for weight in config["weights"]:
        for image_condition in ("real", "shuffle"):
            arm_dir = output_root / "arms" / f"w{weight}_{image_condition}"
            arms.append(
                run_arm(config, sft_dirs[image_condition], float(weight), image_condition, arm_dir, hanorec_root)
            )
    result = {
        "sft_bundles": {key: bundle["status"] for key, bundle in sft_bundles.items()},
        "arms": arms,
        "reference_reproducibility_check": {
            key: bundle["reference_reproducibility_check"] for key, bundle in sft_bundles.items()
        },
        "status": "COMPLETE" if all(arm["status"] == "COMPLETE" for arm in arms) else "PARTIAL",
    }
    _atomic_json_write(output_root / "hanorec_cf_hardness_result.json", result)
    return result

SOURCE_MANIFEST = {"experiment_sha256": "ed8db58116d6244f9b48967fd45002be401adf9f2e494cc0ab1ed20bd8ea4c2a", "generator": "scripts/build_hanorec_exploratory.py", "prep_sha256": "2ef1ed889ccd4e0d4e2fd552d07a74a5ae05529eabf247be33f04b41abac1931", "train_sha256": "d4c6034aad8ac4b7c4135a4a9425ec16c77b66b4f48d291a2a3d5f0c8f9dc9be"}
EXPERIMENT_CONFIG = json.loads('{"artifact_pins": {"LLM2Rec-budgeted-|kaggle|working|LLM2Rec|repeated_evaluate_with_seq-Sep-09-2026_11-12-13-c93f12.pth": "fb0d1d3916f10b5b6f0d9f597e7b2ac8437f445c15629ebf3099333b72806a59", "Qwen2-0.5B-LLM2Rec-IEM-budgeted_step500_title_item_embs.npy": "89f9d3dd17f9ec49537f910e39c9761878299f7d24abdcf805ab0db91433c0f0", "Video_Games_image_manifest.jsonl": "5d9c586d2d05c6ff208c7a0f945a441a4fc60ef0d18ec8aacc269fbf67a42d04", "data.txt": "48c7ead98abfa73c506bf7af133703f2ee7e922167d922760bb074376639a72c", "games_sasrec_step500.log": "a60181a26114072dd5442fd9ffe4b338505010d8b321d9e31ca1646b341ff154", "item_titles.json": "17f501809c4159905cfa6dfa5626d3d52d0be32bc9747ee89b003662f6f5e84c", "test_data.txt": "80074ffea37928d92c35038e9d8bbf3de0cfcf01308ae4173e3c680c1bc50d1f", "train_data.txt": "1300e5deec29d4bede6e32bfa1a5eade563f53a6916da95d67754ec40b9475b8", "val_data.txt": "484a97cfbcff2a78e92612b15863ce08c84c285c7bfb325ec1f33d3884e6de16"}, "batch_size": 4, "beta0": 0.1, "beta_floor": 1e-06, "budget_seconds": 6600, "checkpoint_name": "LLM2Rec-budgeted-|kaggle|working|LLM2Rec|repeated_evaluate_with_seq-Sep-09-2026_11-12-13-c93f12.pth", "dpo_epochs": 5, "drop_last": true, "embedding_name": "Qwen2-0.5B-LLM2Rec-IEM-budgeted_step500_title_item_embs.npy", "eval_users": 265, "gradient_accumulation_steps": 8, "history_items": 3, "learning_rate": 0.0001, "max_pixels": 262144, "model_id": "Qwen/Qwen2.5-VL-3B-Instruct", "model_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3", "noise_sigma": 0.05, "precision": "bf16", "purpose": "corrected Games adaptation; full signal requires Phase 3 quota gate", "seed": 2024, "sft_epochs": 5, "top_m": 20, "train_pairs": 530, "upstream": {"hanorec": {"commit": "587face74524e4553b5a7aa295fe962004682382", "sha256": "48c52e5625cd4e25e80600fed9bed8aad8906556d267b87d52890b49bf33d6a6", "url": "https://codeload.github.com/wangyu0627/HaNoRec/zip/587face74524e4553b5a7aa295fe962004682382"}, "llm2rec": {"commit": "73b481f710f67166ab958f4985d27b27fb410871", "sha256": "6bcee7b09ec9ad5fa0107a083c7b8060abbead23e1264a49480906de638c1cc8", "url": "https://codeload.github.com/HappyPointer/LLM2Rec/zip/73b481f710f67166ab958f4985d27b27fb410871"}}, "validation_users": 265, "weights": [1.0, 0.0, 0.5]}')
EXPERIMENT_CONFIG.update({
    "seed": 2024,
    "train_pairs": 530, "eval_users": 25, "validation_users": 25,
    "sft_epochs": 1, "dpo_epochs": 1, "sft_batch_size": 1,
    "batch_size": 3, "eval_batch_size": 1, "max_pixels": 4096,
    "verify_scoring_batching": True,
    "gradient_accumulation_steps": 2, "budget_seconds": 30000,
    "weights": [1.0, 0.0, 0.5],
    "smoke": False,
})
INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working/hanorec_exploratory_1seed")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print({"stage": "prepare_start", "source_manifest": SOURCE_MANIFEST})
prepared = prepare(EXPERIMENT_CONFIG, INPUT_ROOT, OUTPUT_ROOT, Path("/kaggle/working"))
if int(EXPERIMENT_CONFIG["batch_size"]) < 3:
    raise RuntimeError("exploratory batch_size must preserve HaNoRec's global mini-batch requirement")
print({"stage": "prepare_complete", "train_count": len(prepared["train"]), "validation_count": len(prepared["validation"]), "evaluation_count": len(prepared["evaluation"])})

result = train_and_evaluate(EXPERIMENT_CONFIG, prepared, OUTPUT_ROOT, Path("/kaggle/working"))
manifest = {
    "source_manifest": SOURCE_MANIFEST,
    "purpose": "exploratory_only_non_confirmatory",
    "must_not_be_used_as_confirmation": True,
    "exploratory_scope": {
        "seed": 2024, "train_pairs": 530, "validation_users": 25, "test_users": 25,
        "epochs": 1, "batch_size": 3, "max_pixels": 4096,
        "weights": [1.0, 0.0, 0.5], "image_conditions": ["real", "shuffle"],
    },
    "limitations": [
        "Single seed; no cross-seed inference.",
        "One epoch; reduced eval population (25/25 users).",
        "Batch 3 at max_pixels 4096 differs from the frozen batch 4 / max_pixels 262144 recipe.",
        "max_pixels reduced from 8192 to 4096 after v1 CUDA OOM in the w1.0/real DPO arm at optimizer update 62/89 (torch.OutOfMemoryError, 14.55/14.56 GiB reserved); the v17 smoke probe measured only a single forward+backward pass on a fixed 3-pair sample and did not capture worst-case real-catalog image sizes.",
        "Cannot support the planned confirmatory or cross-seed claim.",
    ],
    "status": result["status"],
    "sft_bundles": result["sft_bundles"],
    "arms": [{"weight": arm["weight"], "image_condition": arm["image_condition"], "status": arm["status"], "optimizer_updates": arm["optimizer_updates"], "expected_optimizer_updates": arm["expected_optimizer_updates"], "mean_ndcg@10": arm["mean_ndcg@10"], "mean_recall@10": arm["mean_recall@10"], "checkpoint": arm["checkpoint"], "checkpoint_reloaded": arm["checkpoint_reloaded"], "parent_sft_sha256": arm["parent_sft_sha256"]} for arm in result["arms"]],
}
(OUTPUT_ROOT / "exploratory_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
if result["status"] != "COMPLETE":
    raise RuntimeError("exploratory run did not complete")
print({"stage": "exploratory_complete", "status": "COMPLETE", "arms": len(result["arms"])})
