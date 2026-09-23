# Phase 2 Calibration Run — Real Evidence Report (v1 cancelled, v2 COMPLETE)

## Executive summary

- **Issue:** the calibration push (`hanorec-cf-hardness-calibration`, 40 train
  pairs, 20 eval users, 6 arms) ran far longer than the 3000s internal safety
  cap and ended in `KernelWorkerStatus.CANCEL_ACKNOWLEDGED` — this repository
  never issued a cancel; someone/something on the Kaggle side stopped it.
- **Impact:** no clean sizing numbers yet. 3 of 6 arms produced real
  checkpoints before cancellation; the final `hanorec_cf_hardness_result.json`
  (which would carry the real `stage_timings`) was never written because
  cancellation happened before that write.
- **Root cause (code-side, confirmed):** the pre-fix budget check
  (`remaining() <= 0`) was only evaluated between arms/steps, not inside the
  per-example loops. At 5x scale (40 pairs/20 users vs. the smoke test's 8/4),
  a single arm's stats+train pass over 80 DPO examples ran for several
  hundred seconds without any chance to bail early, so the 3000s cap could
  only take effect at the *next* arm boundary — by which point real wall time
  had already run well past it.
- **Status:** fixed in code (`train.py` now checks `remaining()` inside the
  SFT loop, both DPO per-example loops, and the reranking per-row loop — see
  `plan.md` Phase 2 Section 3). Not yet re-verified with a fresh push.
- **Fix:** re-run calibration with the fixed code, and get user confirmation
  on whether they cancelled the prior run before spending more quota.

## Real evidence recovered

Downloaded via `kaggle_kernel_output` from the cancelled kernel
(the downloaded cancelled-run output (workspace-only, not included in this repository)):

| Artifact | Real, present | Note |
|---|---|---|
| `experiment.json` | Yes | Calibration config: `train_pairs=40`, `eval_users=20`, same 6 arms |
| `prepared_manifest.json` | Yes, 102.5KB | Real `prep.prepare()` output — 40 pairs, 20 eval rows built successfully |
| `images/` | Yes, 334+ real downloaded JPGs | Confirms catalog grew to the real logged `catalog_items: 340` |
| `sft_lora_state.pt` | Yes, 14.2MB | Real SFT completed |
| `arm_w1.0_real.pt` | Yes, 14.2MB | Real arm 1/6 completed |
| `arm_w1.0_shuffle.pt` | Yes, 14.2MB | Real arm 2/6 completed |
| `arm_w0.0_real.pt` | Yes, 14.2MB | Real arm 3/6 completed |
| `arm_w0.0_shuffle.pt`, `arm_w0.5_real.pt`, `arm_w0.5_shuffle.pt` | **Missing** | Arms 4-6 never ran; cancellation occurred during or before arm 4 |
| `hanorec_cf_hardness_result.json` | **Missing** | Only written at the very end of `train_and_evaluate`; never reached |
| `.log` | Present but truncated at t≈133s | Kaggle's output download for a `CANCEL_ACKNOWLEDGED` kernel does not appear to return the full accumulated stdout — the `log_stage()` prints added for exactly this purpose are not visible in the retrieved log despite real work clearly having happened downstream (3 real checkpoints exist) |

Log's last real line: `{'train_pairs': 40, 'eval_rows': 20, 'catalog_items': 340}` at t=103.8s (right after `prep.prepare()` returns), followed by the Qwen2.5-VL slow-processor warning at t=133.1s (model load). Nothing after that was retrievable.

## What this does tell us, honestly

- The pipeline is real and correct at 5x scale: `prep.py` built 40 real pairs
  and 20 real eval rows from a 340-item real catalog with real downloaded
  images, and 3 full real SFT+DPO+checkpoint cycles completed.
- **It does not tell us precise seconds/unit.** File download timestamps are
  local (download time), not Kaggle-side creation time, so per-arm duration
  cannot be reconstructed from file metadata. The one piece of real
  timing evidence (t=103.8s for `prepare()`, t=133.1s for model load) only
  covers setup, not the expensive stages.
- Total wall time from push to observed cancellation was somewhere between
  ~4,400s and ~6,200s (bounded by my own poll intervals, not by an exact
  Kaggle timestamp) for 3 of 6 arms plus setup — order-of-magnitude **1.2 to
  1.7 hours for half the arms**, i.e. plausibly 2.5-3.5 hours for the full 6
  at this scale, well past both my 3000s intended cap and a comfortable
  calibration budget.

## Correction after re-investigation

The "ran far longer than the 3000s internal safety cap" framing above was
wrong. Re-reading the v1 notebook's orchestration cell before re-pushing
revealed the config's declared `budget_seconds: 3000` was silently
overridden: `EXPERIMENT_CONFIG["budget_seconds"] = max(60.0, DEADLINE -
time.monotonic())` where `DEADLINE = time.monotonic() + 6600.0` (set in the
setup cell). The real intended cap was **~6600s (110 minutes)**, not 3000s.
The observed 4,400-6,200s cancellation happened *before* that real deadline,
meaning v1 was cancelled by an external actor mid-run, not because it
overran its own budget check — the coarse between-arm-only check just never
got a chance to matter either way. Fixed in the v2 re-sync (Section below):
`DEADLINE` lowered to 2400s and the config's stated `budget_seconds` made
consistent with it.

## v2 re-run (COMPLETE, real numbers)

Re-synced `train.py` cell (index 5, exact-index + prefix-assert match, not
substring search) from the fixed disk file — fine-grained `remaining() <= 0`
checks now sit inside the SFT loop, both DPO per-example loops (stats pass
and train pass), and the reranking per-row loop, plus `stage_timings`
instrumentation via `log_stage()`. Lowered `DEADLINE` (cell 1) from 6600s to
**2400s** to keep this calibration spend small, and added a print of the
dynamically-computed effective `budget_seconds` for transparency. Pushed as
version 2, `kaggle_kernel_push` returncode 0.

Polled every 60s; reached `KernelWorkerStatus.COMPLETE` after **2543.5s**
total kernel wall time (0.706 GPU-hours) — no cancellation, no timeout,
clean self-termination.

Downloaded via `kaggle_kernel_output` to
the downloaded v2 output (workspace-only, not included in this repository). Real artifacts confirmed present:

| Artifact | Present | Note |
|---|---|---|
| `hanorec_cf_hardness_result.json` | Yes, 58,020 bytes, valid JSON | Full 8 top-level keys including `stage_timings` (18 real entries) and `arms` (6 entries) |
| `arm_w1.0_real.pt`, `arm_w1.0_shuffle.pt`, `arm_w0.0_real.pt`, `arm_w0.0_shuffle.pt`, `arm_w0.5_real.pt` | Yes, 14.2-14.9MB each | 5/6 arms `COMPLETE` |
| `arm_w0.5_shuffle.pt` | Correctly absent | 6th arm cleanly `SKIPPED_BUDGET` per `result.json["arms"][5]`, not a download failure — confirmed by the arm's own status field, not inferred from the missing file |
| `images/` | 213 real downloaded JPGs | Client-side download of this directory did not fully mirror the kernel's 340-item catalog before the wrapping shell command's own 180s timeout was hit; irrelevant to throughput analysis (checkpoints + `result.json` were already fully materialized on disk before that timeout) |

`reference_reproducibility_check`: `first`/`second` both `-0.9375150203704834`
— exact match, confirms `disable_adapter()` frozen-reference fix holds at 5x
scale.

Real per-stage timings (seconds), used to derive Section 5's sizing
coefficients in `plan.md`'s Phase 2 file:

| Stage | Seconds | Units | s/unit |
|---|---|---|---|
| `model_load` | 41.90 | — | — |
| `semantic_embedding` | 29.02 | 340 items | 0.0853 |
| `sft_total` | 49.59 | 80 examples | 0.6198 |
| `dpo_stats_pass_*` (5 arms) | 132.8-134.9 | 160 examples each | 0.830-0.843 |
| `dpo_train_pass_*` (5 arms) | 171.4-174.2 | 160 examples each | 1.071-1.089 |
| `rerank_*` (4 full arms + 1 partial) | 172.8-175.0 (full), 34.4 (partial, 80/400 units) | 400 (full), 80 (partial) | 0.429-0.435 |

Per-unit costs are consistent within ~2% across all 5 completed arms — a
trustworthy basis for sizing. Full sizing model and 2-3 concrete `(P, U)`
options: `the workspace plan (phase 2, full-dataset scaling)`
Sections 5-6.
