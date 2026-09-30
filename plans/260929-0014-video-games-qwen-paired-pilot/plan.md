---
title: "video-games-qwen-paired-pilot"
description: ""
status: completed
priority: P1
effort: ""
tags: []
created: 2026-09-29
---

# video-games-qwen-paired-pilot

## Overview

Run an exploratory, single-domain comparison of LLM2Rec's title-only baseline against the Qwen3-VL visual-cue method on Video_Games. Keep the original full-corpus protocol unchanged; this pilot is not evidence for the full AmazonMix-6 protocol.

## Outcome

Produce paired recommendation metrics on the same Video_Games train/validation/test data for `title-only` and `real` arms, with chain seeds 2024, 2025, and 2026.

## Constraints

- Require manual decisions for all 100 Qwen3-VL pilot labels before any corpus-generation job.
- Use only the Video_Games domain and its declared train/validation/test data; never access the reserved Baby holdout.
- Preserve the approved 30 GPU-hour cap, including benchmark, captions, training, and evaluation. Do not start a job if the cumulative projection exceeds the cap.
- Pin model revisions, source/data hashes, runtime versions, and per-arm/seed provenance.

## Non-goals

- Full six-domain caption generation or the 10-profile full matrix.
- Claims about the original AmazonMix-6 pretraining protocol or cross-domain generalization.
- Paraphrase, shuffle, or other control arms.

## Requirements

| ID | Requirement |
|---|---|
| R-01 | All 100 pilot labels are manually reviewed and meet the configured quality gates before generation. |
| R-02 | The experiment uses only the verified 9,517-item Video_Games catalog and identical train/validation/test splits for both arms. |
| R-03 | Compare `title-only` against Qwen3-VL `real` through the same CSFT, representation, and SASRec evaluation stages for seeds 2024/2025/2026. |
| R-04 | L4 benchmark and projected cumulative GPU use fit within 30 hours before full generation or training. |
| R-05 | Kaggle outputs prove completion, input/model/source hashes, per-seed metrics, and no cross-domain or holdout leakage. |

## Phases

| # | Phase | Status |
|---|---|---|
| 1 | [Label review and dataset contract](./phase-01-start.md) | Completed |
| 2 | [Video_Games caption corpus](./phase-02-video-games-corpus.md) | Completed |
| 3 | [Paired training and evaluation](./phase-03-paired-training-evaluation.md) | Completed |

## Success Criteria

- [x] All 100 labels are populated, row/image aligned, and pass the quality gates. They come from an assistant visual audit delegated by the user, not from an independent human annotator.
- [x] The L4 smoke measured 2.801 s/image. The projection for the chosen protocol (one LLM chain per arm) was 22.8 GPU-h. The actual total was 30.77 GPU-h; the user approved the 0.8 h overrun.
- [x] Kaggle completed both arms. The protocol was changed by user decision from six chains to 2 arms × 1 LLM chain × 3 SASRec seeds. Validated Recall/NDCG@10/20 are in `research/caption_augmentation/results/video_games_pilot/paired_comparison.json`.
- [x] Results are labeled as a single-domain exploratory pilot: `docs/reports/v10-video-games-paired-pilot.md`.

## Decision

Use Video_Games because the repository marks it as the development domain and its 9,517-item catalog is the smallest of the six verified domains. Use `title-only` as baseline and Qwen3-VL `real` as the treatment. This intentionally narrows training to one domain and must not be described as AmazonMix-6 pretraining.

## Outcome

Verdict: `NO_GAIN_FROM_CAPTIONS`. Recall@10 was 0.0838 for `title-only` and 0.0783 for `real` (−6.5%). The setup audit found no implementation cause. Remaining uncertainty: LLM chain-seed variance was not measured.

<!-- slug: video-games-qwen-paired-pilot -->

