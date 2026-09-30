---
phase: 2
title: "Video_Games Caption Corpus"
status: completed
priority: P1
effort: "1d"
dependencies: [1]
---

# Phase 2: Video_Games Caption Corpus

## Overview

Generate Qwen3-VL visual cues for the Video_Games catalog only, build the two requested arms, and measure L4 throughput without consuming the full run budget.

## Current Behavior

- `research/caption_augmentation/crosswalk.py:39-62` defines the verified Video_Games catalog block and its global-ID range.
- `research/caption_augmentation/smoke.py:79-87` now selects a deterministic, evenly spaced Video_Games-only sample.
- `research/caption_augmentation/package.py` registers `video_games_l4_smoke`, which captions without title paraphrasing.
- L4 smoke passed on 64/64 Video_Games records: 64/64 image decodes, 64/64 caption status `ok`, zero mismatches and token-cap hits; Qwen3-VL 4B revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, prompt SHA256 `eff7c0caf4c4366844239f70e460cfca73a827af4ecebfa61f0dc0b7090cc091`. Result: `research/caption_augmentation/results/video_games_pilot/l4_smoke/video_games_l4_smoke_results.json`.
- Measured generation is 2.801 s/image, projecting 7.41 L4 GPU-hours for 9,517 items; measured metadata/image-fetch overhead scales to about 0.93 hours, plus one model load (~0.03 hours). Training and evaluation are not included, so the 30-hour end-to-end gate remains unproven. Full-domain generation remains blocked until the 100 pilot labels are manually reviewed.

## Target Change

- Add a deterministic Video_Games-only generation path that reuses the pinned Qwen3-VL model, prompt, immutable revisions, existing image/title hashes, and checkpoint validation.
- Generate only `title-only` and `real` arm text; preserve raw `MISMATCH` output but use `unavailable` in the real arm.
- Omit paraphrasing and all other control arms.
- Run an L4 smoke on a deterministic Video_Games sample, capped at one GPU-hour; proceed to full-domain generation only if the measured end-to-end projection, including training and evaluation, is at most 30 GPU-hours.

## Requirements Covered

R-01, R-02, R-03, R-04

## Implementation Steps

1. [R-02] Add deterministic Video_Games sample selection using the verified catalog block; reject mixed-domain samples.
2. [R-01, R-03] Require a validated human-label acceptance artifact before any Video_Games catalog-generation job.
3. [R-03] Build and checkpoint only the title-only and real arm inputs; retain source item IDs, image/title hashes, caption status, and model identity.
4. [R-04] Package and push the L4 smoke, cap it at one GPU-hour, and record measured throughput, image coverage, and projection.
5. [R-04] Do not schedule full generation if any end-to-end stage estimate makes cumulative GPU use exceed 30 hours.

## Verification

- `python -m py_compile caption_augmentation/*.py caption_augmentation/kaggle/*.py` → generated package sources compile.
- `python -m unittest discover -s caption_augmentation -p "test_*.py"` → sample selection, mismatch handling, hashes, checkpoint identity, and two-arm output checks pass.
- L4 smoke output → all sampled records are Video_Games, revisions and hashes are present, no paraphrase work occurs, and runtime is ≤ 1 GPU-hour.
- Full corpus output → exactly 9,517 domain IDs are accounted for; missing images and mismatch captions remain explicit rather than dropped.

## Load-Bearing Assumptions & Risks

- The human pilot gate passes before generation — breaks if any label remains blank or a quality threshold fails; stop and return to review.
- Caption throughput and training runtime fit the approved cap — breaks if the end-to-end projection exceeds 30 GPU-hours; stop before full-domain generation and seek a revised decision.


## Outcome (2026-09-30)

Corpus kernel `trixuanle/llm2rec-qwen3vl-video-games-corpus-v1` reached COMPLETE on v2 (v1 stopped at 6,299/9,517 and was resumed). 9,517/9,517 IDs, 19 hash-verified shards; captions ok 8,898, mismatch 616, None 3; mismatch/missing-image items kept in the `real` arm as `unavailable`. GPU: 8.23 h. Artifacts: `research/caption_augmentation/results/video_games_pilot/corpus_v1/`.
