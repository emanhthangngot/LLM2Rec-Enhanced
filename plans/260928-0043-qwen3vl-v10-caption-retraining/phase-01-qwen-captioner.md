---
phase: 1
title: "Qwen Captioner Package"
status: cancelled
priority: P1
effort: "Local package, protocol, and regression coverage"
dependencies: []
---

# Phase 1: Qwen Captioner Package

## Overview

Replace the Florence-specific caption path with the user-selected Qwen3-VL-4B structured generator. Keep local construction and generated Kaggle source reproducible without loading real weights on the workstation.

## Current Behavior

- `external/LLM2Rec-Research/research/caption_augmentation/caption.py:17-47,50-75,178-284` pins Transformers 4.44.2, defines Florence `<CAPTION>` generation, and keeps Qwen only for title paraphrase.
- `external/LLM2Rec-Research/research/caption_augmentation/smoke.py:518-639,642-689` calls the image-only `Captioner` and creates a run identity containing only the Florence revision.
- `external/LLM2Rec-Research/research/caption_augmentation/pipeline.py:56-97` runs full-corpus generation but does not register a working `full_matrix` stage.
- `external/LLM2Rec-Research/research/caption_augmentation/package.py:373-544,553-559` inlines the package into the Kaggle notebook and currently emits the Florence generator.
- `external/LLM2Rec-Research/research/caption_augmentation/test_package.py:47-68` asserts the generated full-corpus notebook embeds `Florence2Captioner`.
Unknowns: the Qwen3-VL immutable model/processor revision and exact Kaggle runtime pins must be frozen after the pre-run smoke.

## Target Change

- Implement `Qwen3VLCaptioner` on the existing `Captioner` boundary; pass each item's title with its image using the pilot's exact structured prompt. Preserve raw output and distinguish `MISMATCH` from valid captions.
- Treat `MISMATCH` as an explicit image/caption status, not a literal cue. Keep the catalog row and use the protocol's `unavailable` cue consistently in caption-dependent arms; report its count and coverage.
- Include model ID/revisions, processor revision, prompt hash, decoding settings, image hash, and title hash in cache/run identity. Never accept old Florence shard identity.
- Isolate Qwen generation on a runtime compatible with Qwen3-VL (`transformers>=4.57.0`); keep the LLM2Rec training runtime pinned separately.
- Regenerate the full-corpus Kaggle notebook and metadata from `package.py`; do not hand-edit generated notebook code.

## Requirements Covered

R-01, R-03, R-04

## Implementation Steps

1. Replace the caption backend and update `Captioner` call sites to provide `(item_id, title, image_path)`.
2. Update the protocol and corpus cache identity; preserve deterministic greedy decoding and record immutable revisions.
3. Regenerate the Kaggle package with a new kernel slug, fresh Qwen output path, isolated dependency pins, and no Florence self-source mount.
4. Update regression coverage for Qwen notebook generation, prompt/input identity, `MISMATCH` status handling, and unchanged row/arm invariants.

## Verification

- `cd external/LLM2Rec-Research/research && python -m py_compile caption_augmentation/*.py caption_augmentation/kaggle/*.py` → all package and Kaggle scripts compile.
- `cd external/LLM2Rec-Research/research && python -m unittest discover -s caption_augmentation -p "test_*.py"` → relevant corpus, smoke, and package tests pass.
- Regenerate the full-corpus notebook and parse every code cell with `ast.parse`; confirm it uses the Qwen backend and does not embed Florence runtime installation.
- No full-corpus Kaggle push in this phase.

## Load-Bearing Assumptions & Risks

- The 100-row pilot labels must pass before a corpus-generation job is submitted; failure returns to prompt/model review, not item-level cherry-picking.
- The selected structured prompt conditions captions on titles. This is an intentional protocol revision and must be disclosed in all reports and model claims.
- New Qwen identity and kernel slug are mandatory because the existing kernel self-mounts its prior Florence checkpoints.

## Closure (2026-09-30)

Cancelled: the v10 caption direction was closed as negative before the full matrix ran. Evidence: `docs/reports/v10-video-games-paired-pilot.md` and `docs/reports/v10-caption-screening.md`.
