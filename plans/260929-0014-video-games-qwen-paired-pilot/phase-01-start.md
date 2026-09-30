---
phase: 1
title: "Label Review and Dataset Contract"
status: completed
priority: P1
effort: "1d"
dependencies: []
---

# Phase 1: Label Review and Dataset Contract

## Overview

Collect the user's manual decisions for the 100-row Qwen3-VL pilot and freeze the single-domain Video_Games data contract. No corpus-generation job starts until the reviewed labels pass.

## Current Behavior

- `research/caption_augmentation/results/qwen3vl_pilot/label_sheet_ai_draft.csv:1` includes a `human_label` field, blank in all 100 rows; matching images already exist at `research/caption_augmentation/results/qwen3vl_pilot/images/001.jpg` through `100.jpg`.
- `research/caption_augmentation/experiment.json:47-58,107-127` approves a 30 GPU-hour cap, designates Video_Games as development, and requires the 100-row structured-label gates.
- `research/caption_augmentation/crosswalk.py:39-62` defines Video_Games as the 9,517-item development block, global IDs 66,082–75,598.
- `research/caption_augmentation/kaggle/evaluate_caption.py:189-198,344-377` records hashes for Video_Games train/validation/test and item-title files, then evaluates the Games_5core candidate catalog.
Resolved 2026-09-29: all 100 labels were set through a user-delegated assistant visual audit, not independent human annotation. Evidence is in `results/qwen3vl_pilot/assistant_visual_audit.csv` and `assistant_visual_audit_manifest.json`. Counts: ADDS 76, SUSPECT 10, REDUNDANT 9, OCR 2, GENERIC 2, MISMATCH_OK 1, so all gates pass. The sheet has 50 Video_Games rows and 50 rows from other domains. The dataset contract is `results/video_games_pilot/video_games_dataset_contract.json`: 9,517 contiguous titles and train/valid/test rows of 122,577/15,322/15,323. The split CSVs use 0-based item IDs (0..9516), a different permutation from the 1-based downstream IDs. CSV ID 0 is a real item. The two ID spaces are bridged by ASIN.

## Target Change

- Use the existing AI draft CSV as the review sheet; require all 100 `human_label` values to be explicit before applying human decisions to the canonical pilot labels.
- Add a fail-closed check for row/image/title alignment, the allowed label enum, and the existing pilot quality thresholds.
- Pin the Video_Games train/validation/test/item-title files and verify the dataset contains exactly the declared domain.

## Requirements Covered

R-01, R-02

## Implementation Steps

1. [R-01] Validate the 100 `n` values against `images/{n:03d}.jpg`, preserve AI suggestions, and stop while any human label is blank.
2. [R-01] After user review, merge only approved labels and enforce ADDS ≥ 40, OCR+REDUNDANT ≤ 20, and SUSPECT ≤ 10.
3. [R-02] Resolve the mounted Video_Games dataset, record train/validation/test/item-title hashes and row/item counts, and reject any file outside that domain.
4. [R-02] Record the single-domain exploratory scope without modifying the registered full-protocol `experiment.json`.

## Verification

- `python -m unittest discover -s caption_augmentation -p "test_*.py"` → review alignment and quality-gate checks pass; missing labels fail closed.
- Kaggle input preflight → artifact contains only Video_Games train/validation/test/item-title paths with hashes and 9,517 catalog items.
- Human review state → `label_sheet_ai_draft.csv` has 100 explicit human labels and none are silently copied from the AI column.

## Load-Bearing Assumptions & Risks

- The user will complete all 100 human decisions — without them, Qwen corpus generation is prohibited.
- `Video_Games` split files are compatible with the pinned LLM2Rec source — breaks if required train/validation fields or item IDs differ; adapt only after inspecting the mounted files.

