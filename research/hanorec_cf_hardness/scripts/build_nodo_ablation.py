"""Generate a prospective sigma-only NoDO ablation using frozen v11 core sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from pathlib import Path

from build_hanorec_notebook import (
    ROOT, KAGGLE_USER, SFT_SLUG, PROBE_SLUG, SETUP, WRITE_SOURCES, RUN_TESTS,
    PARENT_LOAD, code, md, embedded_sources,
)

DEFAULT_PROTOCOL = ROOT.parents[1] / "plans/261002-0142-v11-eight-hour-audit/reports/nodo-ablation-protocol.json"


def validate_protocol(spec):
    arms = spec["arms"]
    if len(arms) != 2 or len({arm["arm_id"] for arm in arms}) != 2:
        raise ValueError("the paired experiment requires exactly two distinct arms")
    if {float(arm["noise_sigma"]) for arm in arms} != {0.0, 0.05}:
        raise ValueError("NoDO ablation must pair sigma0.0 and sigma0.05")
    if spec["weight"] != 1.0 or spec["image_condition"] != "real" or spec["hardness_control"] is not None:
        raise ValueError("hold semantic weight1, real images and hardness control fixed")
    if spec["controlled_config"]["dpo_epochs"] != 1 or spec["test_scoring"]:
        raise ValueError("one matched epoch, validation only")
    if not 60 <= spec["job_timeout_seconds"] <= 9000:
        raise ValueError("paired job cap must be <=2.5 hours")
    if not spec["registered_before_training"]:
        raise ValueError("prospective protocol required")
    return spec


REGISTER = '''
    # Parent and completed no-training probe are actual hash-pinned prerequisites.
    assert hashlib.sha256(manifests[0].read_bytes()).hexdigest() == ABLATION_SPEC["parent_manifest_sha256"]
    assert PARENT["sft_state_sha256"] == ABLATION_SPEC["parent_sft_sha256"]
    assert SOURCE_SHA256 == ABLATION_SPEC["source_sha256"]
    probes = sorted(INPUT.rglob("probe_result.json"))
    assert len(probes) == 1, "expected one completed clean/noisy probe"
    assert hashlib.sha256(probes[0].read_bytes()).hexdigest() == ABLATION_SPEC["probe_result_sha256"]
    prerequisite = json.loads(probes[0].read_text())
    assert prerequisite["status"] == "COMPLETE"
    assert prerequisite["parent_manifest_sha256"] == ABLATION_SPEC["parent_manifest_sha256"]
    assert prerequisite["source_sha256"] == SOURCE_SHA256
    assert BUNDLE["evaluation"] == [] and BUNDLE["sft_test_predictions"] == []
    assert len(COHORT) == ABLATION_SPEC["validation_events"]
    assert len({row["cluster_id"] for row in COHORT}) == ABLATION_SPEC["true_validation_users"]
    assert len(BUNDLE["validation"]) == ABLATION_SPEC["informative_events"]
    for key, value in ABLATION_SPEC["controlled_config"].items():
        assert CONFIG[key] == value, f"scientific setting drifted: {key}"
    OUT_ROOT = Path("/kaggle/working/v11_nodo_ablation")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "study_protocol.json").write_text(json.dumps(ABLATION_SPEC, indent=2))
    ARM_RESULTS = {}
    RECEIPTS = {}
    EXECUTIONS = {}
    ARM_SECONDS = []
    SESSION_LIMIT_S = ABLATION_SPEC["job_timeout_seconds"] - 180
    def digest_file(path):
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()

    def scientific_config(config):
        return {key: value for key, value in config.items() if key not in ("noise_sigma", "budget_seconds")}
'''

RUN_ONE = '''
    def run_one(arm_id):
        declared = next(arm for arm in ABLATION_SPEC["arms"] if arm["arm_id"] == arm_id)
        if arm_id in ARM_RESULTS:
            raise RuntimeError("duplicate arm execution refused")
        elapsed = time.monotonic() - NOTEBOOK_T0
        remaining_arms = len(ABLATION_SPEC["arms"]) - len(ARM_RESULTS)
        full_arm_allowance = (SESSION_LIMIT_S - elapsed) / remaining_arms
        training_budget = full_arm_allowance - ABLATION_SPEC["post_training_reserve_seconds"]
        eta = max(ARM_SECONDS) * 1.15 if ARM_SECONDS else ABLATION_SPEC["estimated_full_arm_seconds"]
        if training_budget <= 0 or eta > full_arm_allowance:
            ARM_RESULTS[arm_id] = {"status": "SKIPPED_SESSION_BUDGET", "reason": "insufficient fair-share full-arm allowance"}
            return ARM_RESULTS[arm_id]
        arm_dir = OUT_ROOT / arm_id
        arm_dir.mkdir(parents=True, exist_ok=False)
        config = {**CONFIG, **ABLATION_SPEC["controlled_config"], "noise_sigma": float(declared["noise_sigma"]),
                  "hardness_control": None,
                  "budget_seconds": training_budget}
        for previous in EXECUTIONS.values():
            assert scientific_config(config) == scientific_config(previous), "non-sigma scientific difference"
        EXECUTIONS[arm_id] = config
        execution = {"arm_id": arm_id, "weight": 1.0, "image_condition": "real", "config": config,
                     "parent_sft_sha256": ABLATION_SPEC["parent_sft_sha256"], "source_sha256": SOURCE_SHA256}
        execution_path = arm_dir / "execution.json"
        execution_path.write_text(json.dumps(execution, indent=2))
        assert json.loads(execution_path.read_text())["config"]["noise_sigma"] == float(declared["noise_sigma"])
        t0 = time.monotonic()
        try:
            result = T.run_arm(config, SFT_DIR, 1.0, "real", arm_dir, SOURCE_DIR)
        except RuntimeError as exc:
            elapsed_arm = time.monotonic() - t0
            if elapsed_arm < config["budget_seconds"] or not str(exc).startswith("DPO optimizer update count mismatch"):
                raise
            failure = {"arm_id": arm_id, "noise_sigma": declared["noise_sigma"],
                       "status": "INCOMPLETE_BUDGET", "reason": str(exc),
                       "execution_sha256": digest_file(execution_path), "elapsed_seconds": elapsed_arm}
            (arm_dir / "budget_failure.json").write_text(json.dumps(failure, indent=2))
            ARM_RESULTS[arm_id] = failure
            print(failure)
            return failure
        ARM_SECONDS.append(time.monotonic() - t0)
        assert result["status"] == "COMPLETE" and result["predictions"] == []
        assert result["optimizer_updates"] == result["expected_optimizer_updates"] == ABLATION_SPEC["expected_optimizer_updates_per_arm"]
        expected_micro_batches = len(BUNDLE["train"]) * 2 // config["batch_size"] * config["dpo_epochs"]
        assert result["micro_batches"] == expected_micro_batches
        assert len(result["validation_predictions"]) == len(BUNDLE["validation"])
        assert result["parent_sft_sha256"] == ABLATION_SPEC["parent_sft_sha256"]
        assert result["checkpoint_reloaded"] and result["init_parity_max_abs_logit"] <= 0.25
        from audit import audit_artifacts
        single_arm = json.loads((SOURCE_DIR / "run-protocol.json").read_text())
        single_arm.update({"execution_matrix": [{"weight": 1.0, "image_condition": "real", "hardness_control": None}],
                           "evaluation_split": "validation"})
        audit = audit_artifacts(arm_dir, single_arm, parent_root=SFT_DIR)
        assert audit["status"] == "PASS", audit["errors"]
        receipt = {"arm_id": arm_id, "noise_sigma": declared["noise_sigma"], "status": "COMPLETE",
                   "execution_sha256": digest_file(execution_path), "result_sha256": digest_file(arm_dir / "arm_result.json"),
                   "checkpoint_sha256": digest_file(arm_dir / result["checkpoint"]), "artifact_audit": audit,
                   "elapsed_seconds": ARM_SECONDS[-1]}
        (arm_dir / "receipt.json").write_text(json.dumps(receipt, indent=2))
        ARM_RESULTS[arm_id] = result
        RECEIPTS[arm_id] = receipt
        print({"arm": arm_id, "sigma": declared["noise_sigma"], "updates": result["optimizer_updates"],
               "parity": result["init_parity_max_abs_logit"], "minutes": ARM_SECONDS[-1] / 60})
        return result
'''

SUMMARY = '''
    cohort_keys = {(r["user_id"], int(r["target"])): r for r in BUNDLE["validation"]}
    values = {"SFT-only": A.full_cohort_values(COHORT, BUNDLE["sft_validation_predictions"])}
    complete = {}
    for arm in ABLATION_SPEC["arms"]:
        tag = arm["arm_id"]
        result = ARM_RESULTS[tag]
        if result["status"] != "COMPLETE":
            continue
        assert EXECUTIONS[tag]["noise_sigma"] == arm["noise_sigma"]
        seen = set()
        for prediction in result["validation_predictions"]:
            key = prediction["user_id"], int(prediction["target"])
            assert key not in seen; seen.add(key)
            assert set(prediction["ranked_candidates"]) == set(cohort_keys[key]["candidates"])
            assert prediction["cluster_id"] == cohort_keys[key]["cluster_id"]
        assert seen == set(cohort_keys)
        values[tag] = A.full_cohort_values(COHORT, result["validation_predictions"])
        complete[tag] = result
    contrast = None
    exploratory = {}
    on = next(a["arm_id"] for a in ABLATION_SPEC["arms"] if a["noise_sigma"] == 0.05)
    off = next(a["arm_id"] for a in ABLATION_SPEC["arms"] if a["noise_sigma"] == 0.0)
    primary_metric = ABLATION_SPEC["primary_contrast"]["metric"]
    if on in complete and off in complete:
        contrast = A.paired_bootstrap(values[off], values[on], primary_metric)
        exploratory = {metric: A.paired_bootstrap(values[off], values[on], metric)
                       for metric in A.METRICS if metric != primary_metric}
    threshold = ABLATION_SPEC["primary_contrast"]["meaningful_delta"]
    verdict = "NOT_COMPUTED_INCOMPLETE"
    if contrast is not None:
        low, high = contrast["ci95"]
        if low >= threshold and contrast["p_two_sided"] <= 0.05:
            verdict = "OFF_MEANINGFULLY_BETTER_THAN_ON"
        elif high <= -threshold and contrast["p_two_sided"] <= 0.05:
            verdict = "ON_MEANINGFULLY_BETTER_THAN_OFF"
        elif -threshold < low and high < threshold:
            verdict = "CI_WITHIN_MEANINGFUL_BAND_CONDITIONAL_MODELS"
        else:
            verdict = "INCONCLUSIVE"
    supporting = {tag: {metric: A.paired_bootstrap(vals, values["SFT-only"], metric) for metric in A.METRICS}
                  for tag, vals in values.items() if tag != "SFT-only"}
    summary = {"status": "COMPLETE" if len(complete) == 2 else "PARTIAL_SESSION_BUDGET",
               "must_not_be_used_as_confirmation": True, "experiment_id": ABLATION_SPEC["experiment_id"],
               "protocol": ABLATION_SPEC, "source_sha256": SOURCE_SHA256, "gpu": GPU,
               "metrics": {tag: A.summarize(vals) for tag, vals in values.items()},
               "primary_off_minus_on": contrast, "primary_family_size": 1,
               "exploratory_off_minus_on": exploratory,
               "primary_verdict": verdict, "meaningful_delta": threshold,
               "supporting_vs_sft": supporting, "supporting_family": "exploratory, not multiplicity-adjusted",
               "receipts": RECEIPTS, "elapsed_minutes": (time.monotonic() - NOTEBOOK_T0) / 60,
               "job_cap_seconds": ABLATION_SPEC["job_timeout_seconds"]}
    (OUT_ROOT / "ablation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"status": summary["status"], "metrics": summary["metrics"], "off_minus_on": contrast}, indent=1))
'''


def build(protocol_path=DEFAULT_PROTOCOL):
    spec = validate_protocol(json.loads(Path(protocol_path).read_text()))
    blobs, hashes = embedded_sources()
    if hashes != spec["source_sha256"]:
        raise ValueError("core sources differ from prospective protocol/eligible SFT")
    owner, slug = spec["kernel"].split("/")
    if owner != KAGGLE_USER:
        raise ValueError("wrong Kaggle owner")
    output = ROOT / "kaggle" / slug
    output.mkdir(parents=True, exist_ok=True)
    cells = [md("""
        # v11 matched semantic-only NoDO ablation
        Frozen eligible SFT parent; two fresh DPO adapters, one paired training seed, one epoch.
        Only sigma varies scientifically (0.05 vs0). No test scoring or claim that NoDO-off reproduces HaNoRec.
        Primary: clean-evaluation NDCG@10 off-minus-on, true-user cluster inference. SFT-only/Recall are supporting.
        Hard session cap2.5h; failures count against the approved cumulative8h budget.
    """), md("## 1. Pinned environment and GPU"), code(SETUP),
        md("## 2. Self-contained hash-verified frozen core sources"),
        code(f"EMBEDDED_SOURCES = {json.dumps(blobs)}\nSOURCE_SHA256 = {json.dumps(hashes)}\n" + textwrap.dedent(WRITE_SOURCES).strip()),
        md("## 3. Existing regressions"), code(RUN_TESTS),
        md("## 4. Verify completed parent and register the matched protocol"), code(PARENT_LOAD),
        code(f"ABLATION_SPEC = json.loads({json.dumps(spec, sort_keys=True)!r})\n" + textwrap.dedent(REGISTER).strip()),
        md("## 5. Per-arm execution (sigma0 is algebraic NoDO-off; core code is unchanged)"), code(RUN_ONE)]
    for arm in spec["arms"]:
        cells += [md(f"### {arm['arm_id']} — sigma={arm['noise_sigma']}"), code(f"run_one({arm['arm_id']!r})")]
    cells += [md("## 6. Audit pairing and evaluate clean inference"), code(SUMMARY)]
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                             "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    (output / "notebook.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
    metadata = {"id": spec["kernel"], "title": slug, "code_file": "notebook.ipynb", "language": "python", "kernel_type": "notebook",
                "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "keywords": [],
                "dataset_sources": [], "kernel_sources": [f"{KAGGLE_USER}/{SFT_SLUG}", f"{KAGGLE_USER}/{PROBE_SLUG}"],
                "competition_sources": [], "model_sources": []}
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    print(json.dumps({"output": str(build(args.protocol))}))
