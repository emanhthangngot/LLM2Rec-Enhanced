"""Generate user-corrected v11 notebooks (valid .ipynb, one block per stage).

Three dependency-gated jobs; at most eight additional GPU-session hours including failures:
  --job sft   corrected labels -> fresh SFT -> validation/saturation gate (cap 3h)
  --job probe frozen eligible parent -> clean/noisy gaps, no training (cap 0.5h)
  --job arms  selected matched follow-up -> paired statistics (cap 4.5h)

Regenerate with:
  python scripts/build_hanorec_notebook.py --job sft
  python scripts/build_hanorec_notebook.py --job probe
  python scripts/build_hanorec_notebook.py --job arms --arms w1.0 w0.0 w0.5 w0.0:mean
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # research/hanorec_cf_hardness
KAGGLE_USER = "trlxun"
SFT_SLUG = "hanorec-v11-user-corrected-sft"
ARMS_SLUG = "hanorec-v11-user-corrected-arms"
PROBE_SLUG = "hanorec-v11-user-corrected-probe"
EMBEDDED = [
    "prep.py", "train.py", "analysis.py", "experiment.json",
    "audit.py", "run-protocol.json", "test_hanorec_fidelity.py", "test_hanorec_protocol.py", "test_hanorec_analysis.py",
    "test_hanorec_identity.py",
]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip().splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
        "source": textwrap.dedent(text).strip().splitlines(True),
    }


def embedded_sources() -> tuple[dict[str, str], dict[str, str]]:
    blobs, hashes = {}, {}
    for name in EMBEDDED:
        raw = (ROOT / name).read_bytes()
        blobs[name] = base64.b64encode(raw).decode("ascii")
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return blobs, hashes


SETUP = '''
    import hashlib, json, os, subprocess, sys, time
    from pathlib import Path

    NOTEBOOK_T0 = time.monotonic()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                    "transformers==4.51.3", "peft==0.15.2", "accelerate==1.6.0", "pillow==11.0.0"], check=True)
    import torch, transformers, peft

    assert torch.cuda.is_available(), "GPU required"
    GPU = {
        "name": torch.cuda.get_device_name(0),
        "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
    }
    # torch.cuda.is_bf16_supported() is True on T4 by emulation, so it proves nothing. Kaggle's kernel API only
    # offers T4/P100 (NvidiaL4 is silently served as a T4), so bf16 is emulated: recorded as an adaptation.
    GPU["bf16_native"] = GPU["capability"][0] >= 8
    print(GPU)
    assert GPU["name"] and torch.cuda.is_bf16_supported(), "bf16 (even emulated) is required by the protocol"
'''

WRITE_SOURCES = '''
    import base64
    SOURCE_DIR = Path("/kaggle/working/v11_source")
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    for name, blob in EMBEDDED_SOURCES.items():
        data = base64.b64decode(blob)
        assert hashlib.sha256(data).hexdigest() == SOURCE_SHA256[name], f"embedded source corrupted: {name}"
        (SOURCE_DIR / name).write_bytes(data)
    sys.path.insert(0, str(SOURCE_DIR))
    print({name: SOURCE_SHA256[name][:12] for name in SOURCE_SHA256})
'''

RUN_TESTS = '''
    # CPU regression tests for every repaired behavior (SFT task, dropout, LoRA scope, tie order, statistics).
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(SOURCE_DIR), "-p", "test_hanorec_*.py"],
                   cwd=SOURCE_DIR, check=True)
'''

CONFIG_CELL = '''
    import copy
    BASE = json.loads((SOURCE_DIR / "experiment.json").read_text())
    CONFIG = copy.deepcopy(BASE)
    CONFIG.update({
        "seed": 2024,
        "train_pairs": 530, "validation_users": 265,
        "eval_users": 1,                      # test rows are built by prep but never scored (validation-only decisions)
        "history_items": 3, "smoke": False,
        "sft_epochs": 4, "dpo_epochs": 1,  # matched historical schedule; corrections are data/statistics only
        "sft_batch_size": 2, "sft_gradient_accumulation_steps": 8,   # train/mllm_sft.sh
        "batch_size": 4, "gradient_accumulation_steps": 8,           # configs/hanorec/*_hit1.yaml
        "eval_batch_size": 1, "verify_scoring_batching": False,
        "budget_seconds": 10800,
        "learning_probe_pairs": 8,
        "weights": [1.0, 0.5, 0.0],
        "sft_probe_pairs": 32,
    })
    print(json.dumps({k: CONFIG[k] for k in ("seed", "train_pairs", "validation_users", "sft_epochs", "dpo_epochs",
                                             "sft_batch_size", "batch_size", "gradient_accumulation_steps", "learning_rate", "noise_sigma", "beta0")}, indent=1))
'''

SFT_JOB_CELLS = [
    ("md", """
        ## 5. Prepare data (train-only pairs, seeded SFT negatives, validation candidates)
        Pinned artifacts are verified by SHA-256 inside `prep.prepare`. Test rows are read by `prep` to build its
        inference-asset object but are **removed before any model call**. Users whose target is not in SASRec's
        top-20 cannot be changed by a reranker, so Qwen scores only the informative users; metrics are expanded to
        the full validation cohort (absent users contribute 0 to every arm *and* to SASRec).
    """),
    ("code", '''
        from prep import prepare
        import train as T
        import analysis as A

        OUT_ROOT = Path("/kaggle/working/v11"); OUT_ROOT.mkdir(parents=True, exist_ok=True)
        # Reuse ONLY image bytes from the archived SFT output, never its trained adapter or pairs.
        import shutil
        archived_manifests = sorted(Path("/kaggle/input").rglob("job_manifest.json"))
        assert len(archived_manifests) == 1, "expected one archived image-cache parent"
        archived_path = archived_manifests[0]
        assert hashlib.sha256(archived_path.read_bytes()).hexdigest() == "7ec0525def7ff2fab2ba1fed7bfa5f1b43bb13ceabada136c9a567f9b7a06fba"
        archived_bundle = json.loads((archived_path.parent / "sft_real/sft_bundle.json").read_text())
        assert hashlib.sha256((archived_path.parent / "sft_real/sft_bundle.json").read_bytes()).hexdigest() == json.loads(archived_path.read_text())["sft_bundle_sha256"]
        reused_files = set()
        for item, record in archived_bundle["catalog"].items():
            source = archived_path.parent / "sft_real" / record["image_path"]
            if not source.is_file():
                continue  # prepare downloads missing bytes; hashes are checked again below
            assert hashlib.sha256(source.read_bytes()).hexdigest() == archived_bundle["provenance"]["image_provenance"][item]["sha256"]
            target = OUT_ROOT / "images" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)
            reused_files.add(source.name)
        t0 = time.monotonic()
        prepared = prepare(CONFIG, Path("/kaggle/input"), OUT_ROOT, SOURCE_DIR)
        archived_hashes = archived_bundle["provenance"]["image_provenance"]
        for item, image in prepared["provenance"]["image_provenance"].items():
            if item in archived_hashes:
                assert image["sha256"] == archived_hashes[item]["sha256"], f"image bytes drifted for item {item}"
        ALL_VALIDATION = list(prepared["validation"])
        SASREC_VALIDATION = list(prepared["baselines"]["sasrec_validation"])
        INFORMATIVE = [r for r in ALL_VALIDATION if int(r["target"]) in set(map(int, r["candidates"]))]
        prepared["validation"] = INFORMATIVE
        prepared["evaluation"] = []
        prepared["baselines"]["sasrec_test"] = []
        sft_pool_clean = all(p["sft_negative"] not in set(p["excluded_items"]) for p in prepared["train"])
        PREP_REPORT = {
            "prepare_seconds": round(time.monotonic() - t0, 1),
            "reused_unique_images": len(reused_files),
            "train_pairs": len(prepared["train"]), "catalog_items": len(prepared["catalog"]),
            "validation_events": len(ALL_VALIDATION), "true_validation_users": len({r["cluster_id"] for r in ALL_VALIDATION}),
            "informative_events": len(INFORMATIVE),
            "candidate_recall@20": len(INFORMATIVE) / len(ALL_VALIDATION),
            "sft_negatives_outside_excluded_items": bool(sft_pool_clean),
            "sft_negative_equals_hard_negative": sum(p["sft_negative"] == p["negative"] for p in prepared["train"]),
            "test_model_scoring": "disabled",
        }
        print(json.dumps(PREP_REPORT, indent=1))
        assert sft_pool_clean and len(INFORMATIVE) >= 20, "too few informative users for a usable gate"
    '''),
    ("md", """
        ## 6. Fixed image budget
        Keep `max_pixels=16384`, the previously measured T4 setting. No resolution search in this diagnostic.
        Actual peak memory is recorded again; historical DPO used up to 96.8% reserved VRAM.
    """),
    ("code", '''
        import gc
        # Preserve the already measured resolution instead of searching for a favorable new recipe.
        CONFIG["max_pixels"] = 16384
        PIXEL_ADAPTATION = {"frozen": 262144, "used": 16384, "reason": "fixed T4 recipe from audited prior measurement"}
        gc.collect(); torch.cuda.empty_cache()
        print({"max_pixels": CONFIG["max_pixels"], "adaptation": PIXEL_ADAPTATION})
    '''),
    ("md", """
        ## 6b. Preflight every catalog image before the fixed four-epoch diagnostic
        Every image (train and validation-only) is processed on CPU before spending training time.
        The recipe is fixed at four SFT epochs to isolate corrected labels/user identity from schedule changes.
    """),
    ("code", '''
        from transformers import AutoProcessor
        preflight_processor = AutoProcessor.from_pretrained(CONFIG["model_id"], revision=CONFIG["model_revision"])
        bad = []
        for item_id, record in prepared["catalog"].items():
            try:
                preflight_processor.image_processor(images=[T._resize_image(record["image_path"], CONFIG["max_pixels"])], return_tensors="pt")
            except Exception as exc:  # noqa: BLE001 - report every unusable image at once
                bad.append((item_id, repr(exc)[:160]))
        assert not bad, f"{len(bad)} catalog images cannot be processed: {bad[:5]}"
        del preflight_processor
        T_EX = 1.7547347672499995  # historical measured fwd+bwd s/example; actual timing recorded again
        ADAPTATIONS = {
            "gpu_bf16": None if GPU["bf16_native"] else "bf16 emulated on " + GPU["name"] + " (Kaggle API offers only T4/P100)",
            "max_pixels": PIXEL_ADAPTATION,
            "sft_epochs": {"frozen": 5, "used": CONFIG["sft_epochs"], "reason": "fixed matched four-epoch schedule; three-hour cap"},
        }
        print({"catalog_images_ok": len(prepared["catalog"]), "seconds_per_example": round(T_EX, 3),
               "sft_epochs": CONFIG["sft_epochs"], "adaptations": ADAPTATIONS})
    '''),
    ("md", """
        ## 7. SFT parent (real images)
        Task = HaNoRec hit=1: true next item -> `Yes`, seeded random unseen item -> `No` (50/50), four fixed epochs,
        micro-batch 2 x accumulation 8, cosine LR, LoRA r=8/alpha=32/dropout 0.05 on the **language-model** layers only.
    """),
    ("code", '''
        SFT_DIR = OUT_ROOT / "sft_real"
        CONFIG["budget_seconds"] = max(1, 10800 - (time.monotonic() - NOTEBOOK_T0) - 1200)
        sft = T.run_sft({**CONFIG, "sft_image_condition": "real"}, prepared, SFT_DIR, SOURCE_DIR)
        assert sft["status"] == "COMPLETE" and len(sft["sft_validation_predictions"]) == len(INFORMATIVE)
        print({"updates": sft["sft_optimizer_updates"], "lora_modules": sft["lora_module_count"],
               "reference_probe": sft["reference_reproducibility_check"], "elapsed_min": round((time.monotonic() - NOTEBOOK_T0) / 60, 1)})
        print(json.dumps(sft["sft_probe"], indent=1))
    '''),
    ("md", """
        ## 8. Gate: is the SFT reference still saturated, and where does SFT sit against SASRec?
        `positive_yes_fraction` / `sft_negative_no_fraction` are measured on 32 *training* pairs: a saturation detector
        (the old reference answered `Yes` to everything), not a generalisation test. Generalisation is the SFT-only
        vs SASRec paired bootstrap on the full validation cohort. Nothing is tuned from either.
    """),
    ("code", '''
        probe_stats = sft["sft_probe"]
        GATE = {
            "positive_yes_fraction": probe_stats["positive_yes_fraction"],
            "sft_negative_no_fraction": probe_stats["sft_negative_no_fraction"],
            "hard_negative_no_fraction": probe_stats["hard_negative_no_fraction"],
        }
        GATE["reference_learned_binary_task"] = bool(
            GATE["positive_yes_fraction"] >= 0.6 and GATE["sft_negative_no_fraction"] >= 0.6)
        sasrec_vals = A.full_cohort_values(SASREC_VALIDATION, SASREC_VALIDATION)
        sft_vals = A.full_cohort_values(SASREC_VALIDATION, sft["sft_validation_predictions"])
        sft_vs_sasrec = {m: A.paired_bootstrap(sft_vals, sasrec_vals, m) for m in A.METRICS}
        SUMMARY = {
            "gate": GATE,
            "sasrec_order": A.summarize(sasrec_vals), "sft_only": A.summarize(sft_vals),
            "sft_minus_sasrec": sft_vs_sasrec,
            "sft_vs_sasrec_verdict": A.retriever_verdict(sft_vs_sasrec["ndcg@10"]),
            "informative_events": len(INFORMATIVE), "validation_events": len(ALL_VALIDATION),
            "true_validation_users": len({row["cluster_id"] for row in ALL_VALIDATION}),
        }
        print(json.dumps(SUMMARY, indent=1))
    '''),
    ("md", """
        ## 9. Package artifacts
        `job_manifest.json` pins source hashes, GPU, chosen pixels and timings. The arm job mounts this kernel's
        output (SFT bundle, adapter, images) and refuses to start if hashes disagree.
    """),
    ("code", '''
        MANIFEST = {
            "job": "sft", "status": "COMPLETE", "sft_gate_passed": GATE["reference_learned_binary_task"],
            "must_not_be_used_as_confirmation": True,
            "stage_gpu_cap_seconds": 10800, "total_additional_gpu_cap_seconds": 28800,
            "source_sha256": SOURCE_SHA256, "gpu": GPU,
            "config": {k: v for k, v in CONFIG.items() if k != "artifact_pins"},
            "adaptations": ADAPTATIONS, "seconds_per_example": T_EX,
            "prep_report": PREP_REPORT, "summary": SUMMARY,
            "sft_bundle_sha256": hashlib.sha256((SFT_DIR / "sft_bundle.json").read_bytes()).hexdigest(),
            "sft_state_sha256": sft["sft_state_sha256"],
            "elapsed_minutes": round((time.monotonic() - NOTEBOOK_T0) / 60, 1),
        }
        (OUT_ROOT / "job_manifest.json").write_text(json.dumps(MANIFEST, indent=2, default=str))
        print("PACKAGED", MANIFEST["elapsed_minutes"], "min; SFT gate:", GATE["reference_learned_binary_task"],
              "| SFT vs SASRec:", SUMMARY["sft_vs_sasrec_verdict"])
    '''),
]


PARENT_LOAD = textwrap.dedent('''
    import train as T
    import analysis as A
    INPUT = Path("/kaggle/input")
    manifests = sorted(INPUT.rglob("job_manifest.json"))
    assert len(manifests) == 1, f"expected exactly one SFT job_manifest.json, got {manifests}"
    OUT_SFT = manifests[0].parent
    PARENT = json.loads(manifests[0].read_text())
    assert PARENT["job"] == "sft" and PARENT["status"] == "COMPLETE"
    assert PARENT["sft_gate_passed"], f"SFT reference saturation gate failed: {PARENT['summary']['gate']}"
    for name in SOURCE_SHA256:
        assert PARENT["source_sha256"][name] == SOURCE_SHA256[name], f"source drift vs SFT job: {name}"
    SFT_DIR = OUT_SFT / "sft_real"
    assert hashlib.sha256((SFT_DIR / "sft_bundle.json").read_bytes()).hexdigest() == PARENT["sft_bundle_sha256"]
    assert hashlib.sha256((SFT_DIR / "sft_lora_state.pt").read_bytes()).hexdigest() == PARENT["sft_state_sha256"]
    CONFIG = {**PARENT["config"], "artifact_pins": json.loads((SOURCE_DIR / "experiment.json").read_text())["artifact_pins"]}
    BUNDLE = json.loads((SFT_DIR / "sft_bundle.json").read_text())
    COHORT = BUNDLE["baselines"]["sasrec_validation"]
    print({"max_pixels": CONFIG["max_pixels"], "informative_events": len(BUNDLE["validation"]),
           "validation_events": len(COHORT), "true_users": len({r["cluster_id"] for r in COHORT})})
''')


def probe_cells():
    return [
        ("md", "## 5. Verify the new SFT parent; no training or optimizer steps"),
        ("code", PARENT_LOAD),
        ("md", "## 6. Fixed clean/noisy train and validation probes (0.5-hour session cap)"),
        ("code", '''
            import math
            T._seed_everything(int(CONFIG["seed"]), torch)
            T._reset_cuda_peak_memory(torch)
            processor, model = T._load_base_model(CONFIG)
            model = T._add_dpo_lora(T._wrap_sft_lora(model))
            T._load_adapter_state(model, torch.load(SFT_DIR / "sft_lora_state.pt", map_location=model.device, weights_only=True), "sft")
            disabled = T._disable_dropout(model, torch)
            T._assert_language_only_lora(model)
            model.eval()
            assert time.monotonic() - NOTEBOOK_T0 < 1500, "model setup used the probe budget"
            catalog = {item: {**record, "image_path": str(SFT_DIR / record["image_path"])}
                       for item, record in BUNDLE["catalog"].items()}
            yes = T._single_token_id(processor.tokenizer, "Yes")
            no = T._single_token_id(processor.tokenizer, "No")
            sft_before = T._adapter_state(model, "sft")
            dpo_before = T._adapter_state(model, "dpo")
            cpu_rng = torch.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state_all()
            result = T._learning_probe(CONFIG, processor, model, catalog, BUNDLE["train"], BUNDLE["validation"],
                                       yes, no, torch, torch.nn.functional, None)
            assert torch.equal(cpu_rng, torch.get_rng_state())
            assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
            sft_after = T._adapter_state(model, "sft")
            dpo_after = T._adapter_state(model, "dpo")
            assert all(torch.equal(value, sft_after[name]) for name, value in sft_before.items())
            assert all(torch.equal(value, dpo_after[name]) for name, value in dpo_before.items())
            for split in ("train", "validation"):
                assert result[split]["pairs"] > 0
                for field in ("clean_policy_minus_reference", "noisy_policy_minus_reference", "clean_chosen_minus_rejected"):
                    assert all(math.isfinite(v) for v in result[split][field]), f"non-finite {split}/{field}"
                assert max(abs(v) for v in result[split]["clean_policy_minus_reference"]) <= 0.25
            OUT_ROOT = Path("/kaggle/working/v11_probe"); OUT_ROOT.mkdir(parents=True, exist_ok=True)
            payload = {"status": "COMPLETE", "purpose": "clean_noisy_probe_only_no_training",
                       "must_not_be_used_as_confirmation": True, "source_sha256": SOURCE_SHA256,
                       "parent_manifest_sha256": hashlib.sha256(manifests[0].read_bytes()).hexdigest(),
                       "parent_sft_sha256": PARENT["sft_state_sha256"], "gpu": GPU,
                       "dropout_modules_disabled": disabled, "probe": result, "gpu_memory": T._cuda_peak_memory(torch),
                       "stage_gpu_cap_seconds": 1800, "elapsed_minutes": (time.monotonic() - NOTEBOOK_T0) / 60}
            (OUT_ROOT / "probe_result.json").write_text(json.dumps(payload, indent=2))
            print(json.dumps(payload, indent=1))
        '''),
    ]


def arms_cells(arms: list[tuple[float, str | None]]) -> list[tuple[str, str]]:
    cells: list[tuple[str, str]] = [
        ("md", """
            ## 5. Locate and verify the SFT parent
            Mounted from the SFT kernel's output. Source hashes, `sft_bundle.json` and adapter hash must match.
        """),
        ("code", PARENT_LOAD + textwrap.dedent('''
            T_EX = PARENT["seconds_per_example"]
            SESSION_LIMIT_S = 4.4 * 3600  # matched follow-up cap 4.5 h, including packaging
            CONFIG["dpo_epochs"] = 1
            ADAPTATIONS = {**PARENT["adaptations"],
                           "dpo_epochs": {"frozen": 5, "used": 1, "reason": "matched one-epoch corrected-data diagnostic"}}
            print({"dpo_epochs": CONFIG["dpo_epochs"], "adaptations": ADAPTATIONS})
            OUT_ROOT = Path("/kaggle/working/v11_arms"); OUT_ROOT.mkdir(parents=True, exist_ok=True)
            ARM_RESULTS = {}
            ARM_SECONDS = []
            EXECUTION_MATRIX = __ARM_MATRIX__
            declared_protocol = json.loads((SOURCE_DIR / "run-protocol.json").read_text())
            arm_key = lambda arm: (float(arm["weight"]), arm["image_condition"], arm.get("hardness_control"))
            assert len(EXECUTION_MATRIX) == len(declared_protocol["execution_matrix"])
            assert {arm_key(arm) for arm in EXECUTION_MATRIX} == {arm_key(arm) for arm in declared_protocol["execution_matrix"]}, "generated matrix narrows or changes the declared protocol"
        ''')),
        ("md", """
            ## 6. Arms (one block per arm)
            Each arm: fresh DPO adapter over the frozen SFT adapter, SFT-only reference, dropout off, NoDO noise,
            one fixed epoch, micro-batch 4 x accumulation 8. Fail-closed checks inside `run_arm`: policy == reference at
            step 0 (`init_parity_max_abs_logit`), exact optimizer-update count, SFT adapter unchanged, checkpoint reload.
            A block is skipped (and recorded) if it cannot finish inside the session limit.
        """),
        ("code", '''
            def run_one(weight, control):
                tag = f"w{weight}" + (f"_{control}" if control else "")
                if tag in ARM_RESULTS:
                    return ARM_RESULTS[tag]
                eta = max(ARM_SECONDS) * 1.15 if ARM_SECONDS else 0.0
                if time.monotonic() - NOTEBOOK_T0 + eta > SESSION_LIMIT_S:
                    ARM_RESULTS[tag] = {"status": "SKIPPED_SESSION_BUDGET"}
                    print(tag, ARM_RESULTS[tag]); return ARM_RESULTS[tag]
                t0 = time.monotonic()
                arm_config = {**CONFIG, "hardness_control": control,
                              "budget_seconds": max(1, SESSION_LIMIT_S - (time.monotonic() - NOTEBOOK_T0) - 1200)}
                arm = T.run_arm(arm_config, SFT_DIR, float(weight), "real",
                                OUT_ROOT / f"arm_{tag}", SOURCE_DIR)
                ARM_SECONDS.append(time.monotonic() - t0)
                assert arm["status"] == "COMPLETE" and arm["predictions"] == [], "test scoring must stay disabled"
                ARM_RESULTS[tag] = arm
                losses = arm["losses"]
                print({"arm": tag, "val_ndcg@10_informative": arm["validation_mean_ndcg@10"],
                       "init_parity": arm["init_parity_max_abs_logit"], "dropout_off": arm["dropout_modules_disabled"],
                       "first_loss": round(losses[0], 4), "mean_first_20": round(sum(losses[:20]) / len(losses[:20]), 4),
                       "mean_last_20": round(sum(losses[-20:]) / len(losses[-20:]), 4), "minutes": round(ARM_SECONDS[-1] / 60, 1)})
                return arm
        '''),
    ]
    for weight, control in arms:
        label = f"w={weight}" + (f", constant-hardness control ({control})" if control else "")
        cells.append(("md", f"### Arm {label}"))
        cells.append(("code", f"run_one({weight!r}, {control!r})"))
    cells += [
        ("md", """
            ## 7. Paired statistics (validation only)
            Planned primary contrasts: w=0.0 vs w=1.0 and w=0.5 vs w=1.0 (NDCG@10, Holm-adjusted). Every arm is also
            compared with the SASRec order and the SFT-only reranker so degradation is never hidden.
        """),
        ("code", '''
            vals = {"sasrec": A.full_cohort_values(COHORT, COHORT),
                    "sft_only": A.full_cohort_values(COHORT, BUNDLE["sft_validation_predictions"])}
            for tag, arm in ARM_RESULTS.items():
                if arm.get("status") == "COMPLETE":
                    vals[tag] = A.full_cohort_values(COHORT, arm["validation_predictions"])
            RESULT = {"metrics": {k: A.summarize(v) for k, v in vals.items()}, "vs_sasrec": {}, "vs_sft_only": {}, "mechanism": {}}
            for tag in vals:
                if tag in ("sasrec", "sft_only"):
                    continue
                RESULT["vs_sasrec"][tag] = {m: A.paired_bootstrap(vals[tag], vals["sasrec"], m) for m in A.METRICS}
                RESULT["vs_sasrec"][tag]["verdict"] = A.retriever_verdict(RESULT["vs_sasrec"][tag]["ndcg@10"])
                RESULT["vs_sft_only"][tag] = A.paired_bootstrap(vals[tag], vals["sft_only"], "ndcg@10")
            planned = {}
            for tag in ("w0.0", "w0.5"):
                if tag in vals and "w1.0" in vals:
                    planned[f"{tag}_minus_w1.0"] = A.paired_bootstrap(vals[tag], vals["w1.0"], "ndcg@10")
            primary_names = ("w0.0_minus_w1.0", "w0.5_minus_w1.0")
            adjusted = A.holm_adjust({name: planned[name]["p_two_sided"] if name in planned else 1.0
                                     for name in primary_names})
            RESULT["mechanism"] = {"contrasts": planned, "holm_adjusted_p": adjusted,
                                   "min_meaningful_ndcg_delta": A.MIN_MEANINGFUL_NDCG_DELTA}
            RESULT["mechanism"]["missing_primary_contrasts"] = [name for name in primary_names if name not in planned]
            if "w0.0_mean" in vals and "w0.0" in vals:
                RESULT["mechanism"]["cf_signal_beyond_constant_hardness"] = {
                    **A.paired_bootstrap(vals["w0.0"], vals["w0.0_mean"], "ndcg@10"),
                    "family": "secondary diagnostic, not Holm-adjusted",
                }
            print(json.dumps({"metrics": RESULT["metrics"], "verdicts": {k: v["verdict"] for k, v in RESULT["vs_sasrec"].items()}}, indent=1))
            print(json.dumps(RESULT["mechanism"], indent=1))
            from audit import audit_artifacts
            execution_protocol = json.loads((SOURCE_DIR / "run-protocol.json").read_text())
            artifact_audit = audit_artifacts(OUT_ROOT, execution_protocol, parent_root=SFT_DIR)
            if all(arm.get("status") == "COMPLETE" for arm in ARM_RESULTS.values()) and artifact_audit["status"] != "PASS":
                raise RuntimeError(f"completed arm artifacts fail audit: {artifact_audit['errors']}")
        '''),
        ("md", "## 8. Package artifacts"),
        ("code", '''
            summary = {
                "job": "arms",
                "status": ("FAILED" if any(a.get("status") not in ("COMPLETE", "SKIPPED_SESSION_BUDGET") for a in ARM_RESULTS.values())
                           else "PARTIAL_SESSION_BUDGET" if any(a.get("status") == "SKIPPED_SESSION_BUDGET" for a in ARM_RESULTS.values())
                           else "COMPLETE"),
                "adaptations": ADAPTATIONS,
                "must_not_be_used_as_confirmation": True,
                "stage_gpu_cap_seconds": 16200, "total_additional_gpu_cap_seconds": 28800,
                "single_seed": CONFIG["seed"],
                "source_sha256": SOURCE_SHA256, "gpu": GPU, "parent_manifest_sha256": hashlib.sha256(manifests[0].read_bytes()).hexdigest(),
                "arms": {tag: {k: a.get(k) for k in ("status", "optimizer_updates", "expected_optimizer_updates", "init_parity_max_abs_logit",
                                                     "dropout_modules_disabled", "hardness_control", "validation_mean_ndcg@10", "elapsed_seconds", "parent_sft_sha256")}
                         for tag, a in ARM_RESULTS.items()},
                "artifact_audit": artifact_audit,
                "result": RESULT, "elapsed_minutes": round((time.monotonic() - NOTEBOOK_T0) / 60, 1),
            }
            (OUT_ROOT / "arms_summary.json").write_text(json.dumps(summary, indent=2, default=str))
            # Per-user predictions stay in each arm_result.json for later paired analysis; drop nothing.
            print("PACKAGED", summary["status"], summary["elapsed_minutes"], "min")
        '''),
    ]
    matrix = [{"weight": weight, "image_condition": "real", "hardness_control": control} for weight, control in arms]
    return [(kind, text.replace("__N_ARMS__", str(len(arms))).replace("__ARM_MATRIX__", repr(matrix))) for kind, text in cells]


def build(job: str, arms: list[tuple[float, str | None]]) -> Path:
    blobs, hashes = embedded_sources()
    slug = {"sft": SFT_SLUG, "arms": ARMS_SLUG, "probe": PROBE_SLUG}[job]
    out_dir = ROOT / "kaggle" / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    title = {
        "sft": "# v11 repaired - SFT parent and gate",
        "arms": "# v11 repaired - DPO arms and paired statistics",
        "probe": "# v11 corrected - clean/noisy preference probe",
    }[job]
    intro = md(f"""
        {title}

        HaNoRec (pinned `587face7`) CF-hardness reranker over frozen LLM2Rec + SASRec top-20 on Amazon Games.
        This notebook is generated by `scripts/build_hanorec_notebook.py`; sources below are embedded and hash-checked.

        **Repairs vs the exploratory run:** SFT now learns the binary Yes/No task (50% Yes / 50% No, as upstream);
        DPO policy runs with dropout off (LLaMA-Factory default); LoRA is language-model only (vision tower frozen);
        equal scores keep the retriever order. **Adaptations are recorded, not hidden:** Kaggle serves a T4 (bf16 emulated),
        with fixed `max_pixels=16384`, four SFT epochs and one DPO epoch. New repairs: real-user exclusions,
        clustered uncertainty, null-based permutation p-values, and clean/noisy learning probes.
        **Not a confirmation:** one seed, validation only, test never scored.
    """)
    cells = [
        intro,
        md("## 1. Environment and GPU gate"), code(SETUP),
        md("## 2. Embedded, hash-verified sources"),
        code(f"EMBEDDED_SOURCES = {json.dumps(blobs)}\nSOURCE_SHA256 = {json.dumps(hashes, indent=1)}\n" + textwrap.dedent(WRITE_SOURCES).strip()),
        md("## 3. CPU regression tests"), code(RUN_TESTS),
        md("## 4. Protocol configuration"), *([code(CONFIG_CELL)] if job == "sft" else []),
    ]
    stage_cells = SFT_JOB_CELLS if job == "sft" else probe_cells() if job == "probe" else arms_cells(arms)
    for kind, text in stage_cells:
        cells.append(md(text) if kind == "md" else code(text))
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 4,
    }
    (out_dir / "notebook.ipynb").write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    datasets = ["trixuanle/llm2rec-games-5core", "trixuanle/llm2rec-amazonmix6-5core"]
    kernels = ["trixuanle/llm2rec-budgeted-games-sasrec-evaluation", "trixuanle/llm2rec-g1-preflight-and-visual-screen",
               "trlxun/hanorec-v11-repaired-sft"]
    metadata = {
        "id": f"{KAGGLE_USER}/{slug}", "title": slug, "code_file": "notebook.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "keywords": [],
        "dataset_sources": datasets if job == "sft" else [],
        "kernel_sources": kernels if job == "sft" else [f"{KAGGLE_USER}/{SFT_SLUG}"],
        "competition_sources": [], "model_sources": [],
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return out_dir


def parse_arm(spec: str) -> tuple[float, str | None]:
    weight, _, control = spec.removeprefix("w").partition(":")
    parsed = float(weight), (control or None)
    if parsed[0] not in (0.0, 0.5, 1.0) or parsed[1] not in (None, "mean"):
        raise ValueError("arm must be w0.0, w0.5, w1.0, or one of those weights with :mean")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", choices=["sft", "arms", "probe"], required=True)
    parser.add_argument("--arms", nargs="*", default=["w1.0", "w0.0", "w0.5", "w0.0:mean"])
    args = parser.parse_args()
    out_dir = build(args.job, [parse_arm(a) for a in args.arms])
    print(json.dumps({"output": str(out_dir)}))


if __name__ == "__main__":
    main()
