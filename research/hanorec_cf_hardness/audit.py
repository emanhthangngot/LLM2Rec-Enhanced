"""Fail-closed audits for the corrected HaNoRec Games study."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def protocol_errors(protocol: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    task = protocol.get("task", {})
    recipe = protocol.get("recipe", {})
    source = protocol.get("source_authority", {})
    if not task.get("primary_contrasts"):
        errors.append("primary contrasts are missing")
    if float(task.get("meaningful_ndcg_delta", 0)) <= 0:
        errors.append("meaningful NDCG threshold must be positive")
    if source.get("hanorec", {}).get("commit") != "587face74524e4553b5a7aa295fe962004682382":
        errors.append("HaNoRec source commit is not pinned")
    if source.get("llm2rec", {}).get("commit") != "73b481f710f67166ab958f4985d27b27fb410871":
        errors.append("LLM2Rec source commit is not pinned")
    for stage in ("sft", "dpo"):
        config = recipe.get(stage, {})
        for field in ("batch_size", "gradient_accumulation_steps", "epochs", "max_grad_norm"):
            if field not in config:
                errors.append(f"{stage}.{field} is missing")
        if stage == "sft" and config.get("precision") != "bf16":
            errors.append("SFT precision must remain bf16")
    if recipe.get("dpo", {}).get("reference") != "frozen post-SFT adapter, no DPO adapter":
        errors.append("DPO reference contract is not frozen")
    if protocol.get("seeds") != [2024, 2025, 2026]:
        errors.append("training seeds must be exactly 2024, 2025, 2026")
    if sorted(protocol.get("weights", [])) != [0.0, 0.5, 1.0]:
        errors.append("weights must contain 0, 0.5, and 1")
    if "invariants" not in protocol or len(protocol["invariants"]) < 8:
        errors.append("protocol invariants are incomplete")
    smoke = protocol.get("smoke", {})
    if not smoke.get("must_not_be_used_as_signal", False):
        errors.append("smoke must be explicitly non-signal")
    return errors


def audit_protocol(path: Path) -> dict[str, Any]:
    protocol = _load(path)
    errors = protocol_errors(protocol)
    result = {"status": "PASS" if not errors else "FAIL", "errors": errors, "protocol": str(path)}
    print(json.dumps(result, indent=2))
    return result


def _find(root: Path, name: str) -> list[Path]:
    return sorted(path for path in root.rglob(name) if path.is_file())


def audit_artifacts(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    required = ["sft_bundle.json", "sft_lora_state.pt"]
    missing = [name for name in required if not _find(root, name)]
    arm_results = _find(root, "arm_result.json")
    errors = [f"missing {name}" for name in missing]
    expected = {
        (float(weight), condition)
        for weight in protocol["weights"]
        for condition in protocol["controls"]["image_conditions"]
    }
    smoke_manifest = root / "smoke_manifest.json"
    if smoke_manifest.exists():
        smoke = _load(smoke_manifest)
        if smoke.get("must_not_be_used_as_signal") is not True:
            errors.append("smoke manifest is not explicitly non-signal")
        if smoke.get("purpose") != "correctness_only_non_signal":
            errors.append("smoke purpose is not correctness-only")
        expected = {(1.0, "real")}
        for label in ("sft_gpu_memory", "dpo_gpu_memory"):
            memory = smoke.get(label, {})
            total = int(memory.get("total_bytes", 0))
            allocated = int(memory.get("peak_allocated_bytes", 0))
            reserved = int(memory.get("peak_reserved_bytes", 0))
            if not (0 < allocated <= reserved <= total):
                errors.append(f"invalid {label} telemetry")
        for label, prefix in (("sft_stage_timings", "sft_update"), ("dpo_stage_timings", "dpo_update_")):
            timings = smoke.get(label, [])
            updates = [
                row for row in timings
                if str(row.get("stage", "")).startswith(prefix)
            ]
            if not updates or any(float(row.get("seconds", 0)) <= 0 for row in updates):
                errors.append(f"invalid {label} optimizer timing")
    observed: set[tuple[float, str]] = set()
    for path in arm_results:
        result = _load(path)
        key = (float(result.get("weight", float("nan"))), str(result.get("image_condition", "")))
        if key in observed:
            errors.append(f"duplicate arm: {key}")
        observed.add(key)
        if result.get("status") != "COMPLETE":
            errors.append(f"incomplete arm: {path}")
        predictions = result.get("predictions", [])
        if not predictions:
            errors.append(f"arm has no predictions: {path}")
        for row in predictions:
            for metric in ("ndcg@10", "recall@10", "candidate_recall@20"):
                if not math.isfinite(float(row.get(metric, float("nan")))):
                    errors.append(f"non-finite {metric}: {path}")
        checkpoint = result.get("checkpoint")
        checkpoint_path = (
            path.parent / checkpoint
            if checkpoint and not Path(checkpoint).is_absolute()
            else Path(checkpoint) if checkpoint else None
        )
        if checkpoint_path is None or not checkpoint_path.exists():
            errors.append(f"missing checkpoint: {path}")
        if not result.get("parent_sft_sha256"):
            errors.append(f"missing parent hash: {path}")
    if observed != expected:
        errors.append(f"arm matrix mismatch: expected {sorted(expected)}, observed {sorted(observed)}")
    output = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "sft_bundle_count": len(_find(root, "sft_bundle.json")),
        "arm_result_count": len(arm_results),
        "expected_arm_count": len(expected),
    }
    print(json.dumps(output, indent=2))
    return output


def _paired_mean(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], metric: str) -> float:
    by_key = {(row["user_id"], row["target"]): row for row in rows_b}
    deltas = []
    for row in rows_a:
        other = by_key.get((row["user_id"], row["target"]))
        if other is None:
            raise ValueError("paired prediction keys differ")
        deltas.append(float(row[metric]) - float(other[metric]))
    if not deltas:
        raise ValueError("no paired predictions")
    return sum(deltas) / len(deltas)

def _paired_deltas(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], metric: str) -> list[float]:
    by_key = {(row["user_id"], row["target"]): row for row in rows_b}
    deltas = []
    for row in rows_a:
        other = by_key.get((row["user_id"], row["target"]))
        if other is None:
            raise ValueError("paired prediction keys differ")
        deltas.append(float(row[metric]) - float(other[metric]))
    if not deltas:
        raise ValueError("no paired predictions")
    return deltas


def _bootstrap(deltas: list[float], resamples: int, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    means = [
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(resamples)
    ]
    means.sort()
    low = means[max(0, int(0.025 * resamples) - 1)]
    high = means[min(resamples - 1, int(0.975 * resamples))]
    p = min(1.0, 2.0 * min(
        sum(value <= 0.0 for value in means) / resamples,
        sum(value >= 0.0 for value in means) / resamples,
    ))
    return {"mean": sum(deltas) / len(deltas), "ci_low": low, "ci_high": high, "p_value": p}


def analyze(root: Path, bootstrap_resamples: int, seed: int) -> dict[str, Any]:
    if bootstrap_resamples < 1000:
        raise ValueError("bootstrap_resamples must be at least 1000")
    arms: dict[tuple[float, str], dict[str, Any]] = {}
    for path in _find(root, "arm_result.json"):
        result = _load(path)
        key = (float(result["weight"]), str(result["image_condition"]))
        if key in arms:
            raise RuntimeError(f"duplicate arm {key}")
        arms[key] = result
    required = [
        (0.0, "real"), (0.5, "real"), (1.0, "real"),
        (0.0, "shuffle"), (0.5, "shuffle"), (1.0, "shuffle"),
    ]
    missing = [key for key in required if key not in arms]
    if missing:
        raise RuntimeError(f"missing final arms: {missing}")
    contrasts = []
    for index, weight in enumerate((0.0, 0.5)):
        real = arms[(weight, "real")]["predictions"]
        baseline = arms[(1.0, "real")]["predictions"]
        ndcg = _bootstrap(_paired_deltas(real, baseline, "ndcg@10"), bootstrap_resamples, seed + index)
        recall = _bootstrap(_paired_deltas(real, baseline, "recall@10"), bootstrap_resamples, seed + 100 + index)
        contrasts.append({"weight": weight, "contrast": "w-vs-w1", "ndcg@10": ndcg, "recall@10": recall})
    p_values = sorted((item["ndcg@10"]["p_value"], index) for index, item in enumerate(contrasts))
    holm = {}
    total = len(p_values)
    for rank, (p_value, index) in enumerate(p_values):
        holm[str(contrasts[index]["weight"])] = min(1.0, (total - rank) * p_value)
    output = {
        "status": "COMPLETE",
        "bootstrap_resamples": bootstrap_resamples,
        "seed": seed,
        "holm_adjusted_ndcg_p": holm,
        "contrasts": contrasts,
    }
    print(json.dumps(output, indent=2))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    protocol_parser = subparsers.add_parser("protocol")
    protocol_parser.add_argument("--protocol", type=Path, required=True)
    artifacts_parser = subparsers.add_parser("artifacts")
    artifacts_parser.add_argument("--protocol", type=Path, required=True)
    artifacts_parser.add_argument("--root", type=Path, required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--protocol", type=Path, required=True)
    analyze_parser.add_argument("--root", type=Path, required=True)
    analyze_parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    analyze_parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    if args.command == "protocol":
        result = audit_protocol(args.protocol)
    elif args.command == "artifacts":
        protocol_result = audit_protocol(args.protocol)
        if protocol_result["status"] != "PASS":
            return 1
        result = audit_artifacts(args.root, _load(args.protocol))
    else:
        protocol_result = audit_protocol(args.protocol)
        if protocol_result["status"] != "PASS":
            return 1
        result = analyze(args.root, args.bootstrap_resamples, args.seed)
    return 0 if result["status"] in {"PASS", "COMPLETE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
