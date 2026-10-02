---
title: "v11: evidence audit and bounded decision study"
description: "Audit current execution before any additional diagnostic; user approved an 8 GPU-hour ceiling."
status: in-progress
priority: P1
created: 2026-10-02
---

# v11 evidence-first decision study

## Approved delivery contract

Outcome: decide whether to continue investing in v11, not manufacture a positive result.
Constraints: at most 8 additional Kaggle GPU-hours including failed runs; preserve raw data and historical artifacts; frozen CF formula; validation-only development; no test tuning; user-level uncertainty must use actual identities. No native published-score reproduction claim.
Non-goals: unlimited training, automatic full three-seed real/shuffle matrix, changes to hardness/encoder to rescue results, claims of universal ineffectiveness.
Acceptance: every load-bearing setup point classified PASS/adaptation/BREAK/unproven with executable evidence; correct metric/statistical units; checkpoint/numerical parity audits; learning probes; a matched decision study only after setup passes; final investment decision or an explicit external-run wait with no fabricated result.

## Steps (reviewed against current code and approved advisory)

1. Freeze source/config/data/checkpoint evidence for SFT v2 and arms v1 (content-addressed manifests; download .pt files before interpretation).
2. Recover real users from source CSV (downstream item IDs equal CSV IDs + 1); audit exclusions and held-out invariance. `prep.py:225-232,429-488` uses prefixes/surrogate row IDs: test actual train-user seen-item sets before accepting negative labels.
3. Compare HaRS/DPO/responsiveness/NoDO numerically with upstream commit 587face74524e4553b5a7aa295fe962004682382; CPU fp32 first, GPU only for unavoidable runtime/precision checks.
4. Inspect saved adapters, optimizer/scheduler/RNG state and update norms; verify SFT reference cannot change.
5. Audit `analysis.py:20-95`: paired/cluster-aware units, null-based p-values, multiplicity and +0.002 meaningfulness.
6. If BREAK is confirmed, fix the cause across preparation/training/statistics callers, keep historical bundles immutable, and revalidate locally before a new run.
7. Measure clean/noisy preference gaps on train and validation probes; budget <=0.5 h for a GPU-only probe and <=3 h for a discriminating training diagnostic.
8. Choose a matched follow-up from measured diagnostics, <=4.5 h and <=8 h total; no blind matrix expansion.
9. Record continue/stop/inconclusive and limits. A stop-investment decision is not proof of global scientific ineffectiveness.

## Verification

Current CPU verification environment: `/tmp/v11-ablation-venv` (torch 2.10.0+cpu, transformers 4.51.3, peft 0.15.2; no GPU spend).
Run source-data reconstruction, held-out mutation tests, upstream numerical parity, actual .pt inspection, full-cohort metric recomputation, null-distribution checks, and tiny-Qwen end-to-end before any new push. GPU kernels contain assertions and a global cumulative deadline; COMPLETE cannot describe a missing stage.

## Observed initial BREAK

Exact source CSV/downstream alignment verified (122577 train rows, 15322 validation rows, item ID offset +1). Completed sample reconstructs exactly. 530 train rows correspond to 505 distinct source users; 265 validation rows correspond to 258 users. 27 hard negatives and 1 SFT negative occur in the same user's TRAIN-known interactions. 224 sampled pairs omit some train-known interactions from exclusions. No additional GPU hours consumed yet.

## Current progress

| Phase | Completed checklist items | State |
|---|---:|---|
| Freeze, diagnose and repair | 5/5 | Audit complete; 31 tests and fresh-model/probe smoke pass; historical mechanism acceptance withdrawn |
| Learning diagnostic | 2/2 | Corrected SFT and real clean/noisy probe COMPLETE/audited; transient logits attenuate strongly, causality still untested |
| Matched decision | 0/2 | Matched semantic-only NoDO-on/off v1 submitted; waiting for completion and independent result audit |

Total scientific checklist: 7/9 complete. Probe audit: parent workspace `data/kaggle/hanorec-v11-user-corrected-probe/probe_result_audit.json`; runtime/budget reconciliation `reports/gpu-budget.json`. Matched-study pre-push proof: `reports/nodo-ablation-verification.md`. Cumulative worst-case reservations are6h;2h remain conservatively unreserved within the8h ceiling. No final v11 investment verdict or training-harm claim yet; submission is not scientific completion.
