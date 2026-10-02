"""Validation-only paired statistics for the repaired v11 gate (standard library only).

Metrics use the full event cohort: targets absent from top-20 contribute zero.
When real-user cluster IDs are supplied, uncertainty resamples whole users.
The mean remains event-weighted; training-seed uncertainty is not estimated here.
"""
from __future__ import annotations

import itertools
import math
import random
from typing import Any, Iterable

METRICS = ("ndcg@10", "recall@10")
MIN_MEANINGFUL_NDCG_DELTA = 0.002  # Phase 1 frozen threshold (plans/260920-*/phase-01-start.md)


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row["user_id"]), int(row["target"]))


def full_cohort_values(cohort: list[dict[str, Any]], predictions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand only non-informative missing events; preserve real-user clusters for resampling."""
    by_key = {}
    for row in predictions:
        key = _key(row)
        if key in by_key:
            raise ValueError("duplicate prediction event")
        by_key[key] = row
    cohort_keys = [_key(row) for row in cohort]
    if len(set(cohort_keys)) != len(cohort_keys):
        raise ValueError("duplicate cohort event")
    if set(by_key) - set(cohort_keys):
        raise ValueError("predictions outside the validation cohort")
    values = []
    for row in cohort:
        prediction = by_key.get(_key(row))
        if prediction is None and float(row["candidate_recall@20"]) != 0.0:
            raise ValueError("informative event has no prediction")
        if prediction is not None and "cluster_id" in row and prediction.get("cluster_id") != row["cluster_id"]:
            raise ValueError("prediction user cluster differs from cohort")
        value = {metric: float(prediction[metric]) if prediction else 0.0
                 for metric in (*METRICS, "candidate_recall@20")}
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in value.values()):
            raise ValueError("metric must be finite and between zero and one")
        if "cluster_id" in row:
            value["cluster_id"] = row["cluster_id"]
        values.append(value)
    return values


def summarize(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        raise ValueError("empty cohort")
    return {metric: sum(v[metric] for v in values) / len(values) for metric in (*METRICS, "candidate_recall@20")}


def paired_bootstrap(
    a: list[dict[str, Any]],
    b: list[dict[str, Any]],
    metric: str = "ndcg@10",
    n: int = 20000,
    seed: int = 20261001,
) -> dict[str, Any]:
    """Event-weighted mean delta with user-cluster bootstrap CI and a null sign-flip test.

    Sign flips act on whole-user paired deltas. This tests within-user label exchangeability,
    not training-seed uncertainty. Exact enumeration is used for <=16 nonzero clusters.
    Unclustered synthetic/legacy callers are explicitly labelled event-level.
    """
    if len(a) != len(b) or not a or n < 100:
        raise ValueError("paired cohorts must be equal-length, non-empty; n must be >=100")
    clustered = any("cluster_id" in row for row in a + b)
    if clustered and any("cluster_id" not in row for row in a + b):
        raise ValueError("real-user clusters must be supplied for all paired events")
    deltas, groups = [], {}
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        if clustered and x["cluster_id"] != y["cluster_id"]:
            raise ValueError("paired user clusters do not match")
        delta = float(x[metric]) - float(y[metric])
        if not math.isfinite(delta):
            raise ValueError("paired delta is not finite")
        deltas.append(delta)
        group = x["cluster_id"] if clustered else i
        total, count = groups.get(group, (0.0, 0))
        groups[group] = (total + delta, count + 1)
    blocks = list(groups.values())
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sampled = [blocks[rng.randrange(len(blocks))] for _ in blocks]
        means.append(sum(total for total, _ in sampled) / sum(count for _, count in sampled))
    means.sort()
    active = [total for total, _ in blocks if total != 0.0]
    observed = abs(sum(deltas))
    if len(active) <= 16:
        null_totals = (sum(sign * total for sign, total in zip(signs, active))
                       for signs in itertools.product((-1, 1), repeat=len(active)))
        p_value = sum(abs(total) >= observed - 1e-12 for total in null_totals) / (2 ** len(active))
        p_method = "exact paired user-cluster sign flip" if clustered else "exact paired event sign flip"
    else:
        null_rng = random.Random(seed + 1)
        exceed = sum(
            abs(sum(total if null_rng.randrange(2) else -total for total in active)) >= observed - 1e-12
            for _ in range(n)
        )
        p_value = (exceed + 1) / (n + 1)
        p_method = "Monte-Carlo paired user-cluster sign flip" if clustered else "Monte-Carlo paired event sign flip"
    return {
        "metric": metric,
        "events": len(deltas),
        "independent_clusters": len(blocks),
        "resampling_unit": "source_user" if clustered else "event (identity not supplied)",
        "mean_delta": sum(deltas) / len(deltas),
        "ci95": [means[int(0.025 * n)], means[min(n - 1, int(0.975 * n))]],
        "p_two_sided": p_value,
        "p_method": p_method,
        "changed_events": sum(delta != 0 for delta in deltas),
        "positive_events": sum(delta > 0 for delta in deltas),
        "negative_events": sum(delta < 0 for delta in deltas),
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        adjusted[name] = running
    return adjusted


def retriever_verdict(contrast: dict[str, Any]) -> str:
    """Classify reranker-minus-retriever NDCG@10 without hiding degradation."""
    low, high = contrast["ci95"]
    if low >= MIN_MEANINGFUL_NDCG_DELTA:
        if contrast.get("p_two_sided") is None or contrast["p_two_sided"] > 0.05:
            return "POSITIVE_CI_NULL_NOT_REJECTED"
        return "BEATS_RETRIEVER"
    if low > 0:
        return "GAIN_BELOW_MEANINGFUL" if high < MIN_MEANINGFUL_NDCG_DELTA else "POSITIVE_GAIN_MEANINGFULNESS_UNCERTAIN"
    if high < 0:
        return "WORSE_THAN_RETRIEVER"
    if high < MIN_MEANINGFUL_NDCG_DELTA:
        return "NO_MEANINGFUL_GAIN"
    return "INCONCLUSIVE"
