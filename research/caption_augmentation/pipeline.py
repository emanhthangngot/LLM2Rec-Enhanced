"""Kaggle entrypoint for the caption-augmentation experiment.

The preflight stage is intentionally complete and dependency-light: it proves
that the generated package, protocol and embedded source are coherent before
any model download or GPU allocation. Training stages are enabled only after
Phase 2 runtime pins and a real-data smoke have been recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_ARMS = ("title-only", "null", "real", "shuffle", "paraphrase")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def registered_matrix_runs(config: dict[str, object]) -> list[dict[str, object]]:
    profiles = config.get("stage_placement_profiles")
    definitions = config.get("profile_definitions")
    seeds = config.get("seeds")
    if not isinstance(profiles, list) or not isinstance(definitions, dict):
        raise RuntimeError("protocol must declare profiles and profile_definitions")
    if set(profiles) != set(definitions) or len(profiles) != 10:
        raise RuntimeError("registered profiles and profile_definitions must match exactly (10)")
    if not isinstance(seeds, dict) or seeds.get("chain_seeds") != [2024, 2025, 2026]:
        raise RuntimeError("protocol chain seeds must be exactly 2024/2025/2026")
    for name in profiles:
        definition = definitions[name]
        if not isinstance(definition, dict):
            raise RuntimeError(f"profile {name!r} must have an object definition")
        if definition.get("csft_cue_arm") not in REQUIRED_ARMS:
            raise RuntimeError(f"profile {name!r} has an invalid CSFT cue arm")
        if definition.get("representation_cue_arm") not in REQUIRED_ARMS:
            raise RuntimeError(f"profile {name!r} has an invalid representation cue arm")
        if definition.get("history_policy") not in ("common-suffix", "native-full-history"):
            raise RuntimeError(f"profile {name!r} has an invalid history policy")
        cap = definition.get("cue_cap_tokens")
        if cap not in (None, 16, 32, 64):
            raise RuntimeError(f"profile {name!r} has an invalid cue cap")
    native = definitions.get("native-title", {})
    if (
        native.get("csft_cue_arm") != "real"
        or native.get("representation_cue_arm") != "real"
        or native.get("history_policy") != "native-full-history"
        or native.get("cue_cap_tokens") is not None
    ):
        raise RuntimeError("native-title must use real cues throughout without history truncation")
    return [
        {
            "profile": profile,
            "chain_seed": seed,
            "definition": definitions[profile],
        }
        for profile in profiles
        for seed in seeds["chain_seeds"]
    ]


def run_preflight(work_dir: Path) -> dict[str, object]:
    config_path = work_dir / "experiment.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"experiment protocol missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if tuple(config.get("arms", ())) != REQUIRED_ARMS:
        raise RuntimeError(f"protocol arms mismatch: {config.get('arms')!r}")
    for key in ("primary_cue_cap_tokens", "max_csft_tokens", "seeds"):
        if key not in config:
            raise RuntimeError(f"protocol missing required key: {key}")
    matrix_runs = registered_matrix_runs(config)
    source_files = sorted(work_dir.glob("*.py"))
    if not source_files:
        raise RuntimeError("embedded source package is empty")
    source_hashes = {path.name: sha256_file(path) for path in source_files}
    result = {
        "status": "PASS",
        "stage": "preflight",
        "protocol_sha256": sha256_file(config_path),
        "source_hashes": source_hashes,
        "arms": list(REQUIRED_ARMS),
        "seeds": config["seeds"],
        "registered_matrix_run_count": len(matrix_runs),
        "registered_matrix_runs": matrix_runs,
        "gpu_required_for_next_stage": True,
        "next_stage": "real_generator_smoke",
    }
    (work_dir / "preflight.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("preflight", "real_generator_smoke", "quality_review", "paraphrase_probe", "full_corpus_generation", "full_matrix"),
        default="preflight",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("/kaggle/working/visual_delta_fusion"))
    parser.add_argument("--paraphrase-model-id", default=None)
    parser.add_argument("--paraphrase-model-revision", default=None)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "preflight":
        run_preflight(args.work_dir)
    elif args.stage in ("real_generator_smoke", "quality_review", "paraphrase_probe", "full_corpus_generation"):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from caption import PARAPHRASE_MODEL_ID, PARAPHRASE_MODEL_REVISION, ensure_runtime_dependencies
        from smoke import (
            locate_mixed_root, run_full_corpus_generation, run_paraphrase_probe,
            run_quality_review, run_real_generator_smoke,
        )

        ensure_runtime_dependencies()
        mixed_root = locate_mixed_root()
        if args.stage == "real_generator_smoke":
            run_real_generator_smoke(args.work_dir, mixed_root)
        elif args.stage == "quality_review":
            run_quality_review(args.work_dir, mixed_root)
        elif args.stage == "paraphrase_probe":
            run_paraphrase_probe(
                args.work_dir, mixed_root,
                model_id=args.paraphrase_model_id or PARAPHRASE_MODEL_ID,
                revision=args.paraphrase_model_revision or PARAPHRASE_MODEL_REVISION,
            )
        else:
            run_full_corpus_generation(args.work_dir, mixed_root)
    else:
        raise RuntimeError(
            f"stage {args.stage!r} is not registered for execution: complete quality_review first"
        )


if __name__ == "__main__":
    main()
