"""18-cell pre-registered visual screening lattice for LLM2Rec.

Evaluates parameter-free visual recommendations against multiple shuffle controls
on validation users at zero recommender training GPU cost.

Grid dimensions:
- History Aggregation (3): linear_recency, uniform_mean, last_5
- Representation (3): centered_pca64, centered_pca128, centered_raw512
- Metric (2): cosine, learned_diagonal
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


def aggregate_history(
    sequence: np.ndarray,
    catalog_visual: np.ndarray,
    availability: np.ndarray,
    method: str = "linear_recency",
    padding_idx: int = 0,
) -> np.ndarray:
    valid_mask = (sequence != padding_idx) & (availability[sequence] == 1)
    valid_items = sequence[valid_mask]
    if len(valid_items) == 0:
        return np.zeros(catalog_visual.shape[1], dtype=np.float32)

    features = catalog_visual[valid_items]
    if method == "linear_recency":
        positions = np.where(valid_mask)[0] + 1
        weights = positions.astype(np.float32)[:, None]
        weighted = (features * weights).sum(axis=0)
        norm = np.linalg.norm(weighted)
        return weighted / norm if norm > 1e-12 else weighted
    elif method == "uniform_mean":
        mean = features.mean(axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 1e-12 else mean
    elif method == "last_5":
        recent = features[-5:]
        mean = recent.mean(axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 1e-12 else mean
    else:
        raise ValueError(f"unknown aggregation method: {method}")


def score_catalog(
    user_profile: np.ndarray,
    catalog_visual: np.ndarray,
    metric: str = "cosine",
    diag_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    if np.linalg.norm(user_profile) == 0:
        return np.zeros(catalog_visual.shape[0], dtype=np.float32)

    if metric == "cosine":
        scores = catalog_visual @ user_profile
    elif metric == "learned_diagonal":
        if diag_weights is None:
            diag_weights = np.ones(catalog_visual.shape[1], dtype=np.float32)
        scores = (catalog_visual * diag_weights) @ (user_profile * diag_weights)
    else:
        raise ValueError(f"unknown metric: {metric}")

    # Standardize over catalog (excluding padding index 0)
    candidates = scores[1:]
    mean = candidates.mean()
    std = max(float(candidates.std()), 1e-6)
    standardized = (scores - mean) / std
    standardized[0] = -1e9  # Mask padding
    return standardized.astype(np.float32)


def evaluate_visual_screen_cell(
    val_sequences: List[np.ndarray],
    val_targets: List[int],
    catalog_visual: np.ndarray,
    availability: np.ndarray,
    aggregation: str,
    metric: str,
    k_list: Tuple[int, ...] = (5, 10, 20),
) -> Dict[str, float]:
    recalls = {k: 0 for k in k_list}
    ndcgs = {k: 0.0 for k in k_list}
    total_users = len(val_targets)

    for seq, target in zip(val_sequences, val_targets):
        profile = aggregate_history(seq, catalog_visual, availability, method=aggregation)
        scores = score_catalog(profile, catalog_visual, metric=metric)
        top_indices = np.argpartition(-scores, max(k_list))[: max(k_list)]
        sorted_top = top_indices[np.argsort(-scores[top_indices])]

        for k in k_list:
            top_k = sorted_top[:k]
            if target in top_k:
                recalls[k] += 1
                rank = int(np.where(top_k == target)[0][0])
                ndcgs[k] += 1.0 / math.log2(rank + 2)

    return {
        **{f"recall@{k}": recalls[k] / total_users for k in k_list},
        **{f"ndcg@{k}": ndcgs[k] / total_users for k in k_list},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a completed 18-cell G1 visual screen")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--dataset")
    args = parser.parse_args()
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    datasets = artifact["datasets"]
    dataset = args.dataset
    if dataset is None:
        if len(datasets) != 1:
            parser.error("--dataset is required when the artifact contains multiple datasets")
        dataset = next(iter(datasets))
    if dataset not in datasets:
        parser.error(f"dataset {dataset!r} is absent from the artifact")
    screen = datasets[dataset]["screen"]
    if len(screen["cells"]) != 18:
        raise RuntimeError(f"expected 18 registered cells, found {len(screen['cells'])}")
    summary = {
        "dataset": dataset,
        "cells": len(screen["cells"]),
        "selected_on_half_a": screen["selected_on_half_a"],
        "confirmation_half_b": screen["confirmation_half_b"],
        "confirmation_half_b_text_weak": screen["confirmation_half_b_text_weak"],
        "screen_passed": screen["screen_passed"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
