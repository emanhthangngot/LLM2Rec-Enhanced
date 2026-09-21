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
from __future__ import annotations

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
