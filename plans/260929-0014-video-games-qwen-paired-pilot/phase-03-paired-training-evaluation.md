---
phase: 3
title: "Paired Training and Evaluation"
status: completed
priority: P1
effort: "2d"
dependencies: [1, 2]
---

# Phase 3: Paired Training and Evaluation

## Overview

Train the title-only baseline and Qwen3-VL real-cue treatment on the same Video_Games interactions, then compare their recommendation metrics under the same evaluation candidate set.

## Current Behavior

- `research/caption_augmentation/kaggle/pipeline.py:101-143` registers preflight and caption-generation stages but rejects `full_matrix`; no downstream full-matrix training stage exists.
- `research/caption_augmentation/kaggle/csft_caption.py:60-93,222-279` consumes a historical V10 Florence manifest and runs one real-caption CSFT path over AmazonMix-6.
- `research/caption_augmentation/kaggle/tiny_chain_csft.py:19-23,43-66,122-189` supports title-only and real inputs only as a one-optimizer-step technical smoke over the old 108,753-row V10 corpus.
- `research/caption_augmentation/kaggle/iem_caption.py:163-170,227-285,305-350` trains MNTP/SimCSE from AmazonMix-6 titles; `evaluate_caption.py:337-388` evaluates Games_5core with full-catalog Recall/NDCG for seeds 2024/2025/2026.
Unknowns: the source trainers' domain-only input behavior and the paired six-run L4 cost must be proven before training.

## Target Change

- Implement a real paired runner for `title-only` and Qwen3-VL `real` that uses the same Video_Games train/validation/test files, model initialization, optimization settings, and candidate set.
- Apply each arm consistently to CSFT history inputs and representation-stage item text; preserve original title targets and train-only interaction-derived quantities.
- User decision (2026-09-29): run one LLM chain per arm (CSFT → MNTP → SimCSE with chain seed 2024), then evaluate SASRec with seeds 2024, 2025, and 2026. The previous plan of six full chains was projected at 37.7 GPU-hours, over the 30-hour cap; see `results/video_games_pilot/budget_projection.json`. Seed variance therefore covers only SASRec, not LLM training.
- Implementation: `kaggle/video_games_paired_arm.py`. It uses one L4 kernel per arm and can resume from its own saved stages. Split-CSV item IDs map to downstream IDs by ASIN. Both arms keep the same history suffix, sized so the real-arm rendering fits `cutoff_len` 1024. Cues are capped at 32 tokens, and target titles are left unchanged. MNTP runs for 5 epochs so that the released 1000-step stop is reached on 9,517 domain texts.

## Requirements Covered

R-02, R-03, R-04, R-05

## Implementation Steps

1. [R-02] Add a domain-only CSFT/IEM preparation path using the pinned LLM2Rec source; assert all interactions and item IDs remain inside Video_Games.
2. [R-03] Run title-only and real arms with identical seeds, split hashes, model revision, optimizer settings, evaluation candidate set, and full-catalog metrics.
3. [R-04] Measure the paired pipeline on L4 and compare the total projected GPU-hours to the approved cap before scheduling all six runs.
4. [R-05] Store run status, split/source/model hashes, checkpoints, timing, image coverage, and Recall/NDCG outputs under `research/caption_augmentation/results/video_games_pilot/`.
5. [R-05] Verify Kaggle status is COMPLETE and reconcile every expected arm/seed result; fail on missing checkpoints, metrics, or domain leakage.

## Verification

- `python -m py_compile caption_augmentation/*.py caption_augmentation/kaggle/*.py` → paired runner and package compile.
- `python -m unittest discover -s caption_augmentation -p "test_*.py"` → domain bounds, identical paired splits/seeds, arm separation, original-title targets, finite metrics, and provenance validation pass.
- Kaggle completion → six arm/seed runs produce nonempty checkpoints and Recall@10/20 plus NDCG@10/20 on the same Video_Games candidate set.
- Budget gate → cumulative measured/projected GPU use, including generation, remains ≤ 30 hours.

## Load-Bearing Assumptions & Risks

- Upstream LLM2Rec trainers accept the domain-scoped split format — breaks if they require AmazonMix-6-only fields; adapt with a pinned, hash-checked compatibility layer, not by silently substituting another dataset.
- Six paired chains fit the cap — breaks if the measured total exceeds 30 GPU-hours; stop before batch training and ask for a new resource/protocol decision.


## Outcome (2026-09-30)

Protocol per user decision: one LLM chain per arm (chain seed 2024) × SASRec seeds 2024/2025/2026, test split. Both arms PASS on L4 (`title-only` v3; `real` v3 CSFT + v4 resume with matching CSFT SHA `0ab4cbed…`). Arm v1 failed because corpus arm records lacked `asin`; fixed by attaching ASINs from hash-verified caption shards. Arm v2 was cancelled externally (4.33 GPU-h lost; cancelled versions are not mounted as `kernel_sources`).

| Metric | title-only | real | paired Δ | real wins |
|---|---|---|---|---|
| Recall@10 | 0.0838 ± 0.0008 | 0.0783 ± 0.0047 | −0.0054 (−6.5%) | 0/3 |
| Recall@20 | 0.1102 ± 0.0008 | 0.1028 ± 0.0048 | −0.0074 (−6.7%) | 0/3 |
| NDCG@10 | 0.0499 ± 0.0021 | 0.0482 ± 0.0044 | −0.0018 (−3.6%) | 1/3 |
| NDCG@20 | 0.0566 ± 0.0020 | 0.0543 ± 0.0045 | −0.0023 (−4.0%) | 1/3 |

Verdict: `NO_GAIN_FROM_CAPTIONS`. Caveat: n=1 LLM chain per arm, so seed spread covers SASRec only. Budget: 30.77 GPU-h total vs 30 cap; the ~0.8 h overrun was approved by the user. Details: `results/video_games_pilot/paired_comparison.json`.
