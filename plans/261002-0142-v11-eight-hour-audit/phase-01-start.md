---
phase: 1
title: "Freeze, diagnose and repair setup"
status: in-progress
priority: P1
---

# Phase 1: Freeze, diagnose and repair setup

## Current behavior and changes

- `prep.py` previously identified event rows as users and matched sequence prefixes. Actual source CSV alignment (+1 item offset) recovered 505 users for 530 train events and 258 users for 265 validation events. 27 hard negatives were known TRAIN positives for the same user (12 earlier/current, 15 later TRAIN interactions); 1 SFT random negative was also known.
- Source CSV pins, hashed `cluster_id`, all-TRAIN user exclusions and a train-only negative universe now replace prefix inference. Event IDs/candidates remain stable.
- `analysis.py` now carries clusters, rejects duplicate predictions, uses whole-user bootstrap and null sign-flip p-values, and requires lower CI bound >=0.002 for meaningful gain.
- `train.py` now records clean/noisy before/after preference probes and clones cached reference scalars to avoid retaining full-vocabulary storage. Objective, encoder, CF formula and NoDO are unchanged.

## Verification and evidence

- Frozen hashes/source snapshots: `data/kaggle/v11-audit-20261002/freeze_manifest.json` in the parent rs-wiki workspace.
- Pinned upstream archive SHA-256: `48c52e5625cd4e25e80600fed9bed8aad8906556d267b87d52890b49bf33d6a6`.
- `/tmp/v11_numeric_audit.py`: CPU fp64/fp32 HaRS/top-K, responsiveness, actual-loss AST/gradients, NoDO outputs/gradients/cleanup PASS. Actual DPO checkpoints: 504 language-only adapter tensors, all 252 B tensors nonzero, optimizer steps 34, scheduler last_epoch/T_max 34/34, final LR 0.0.
- `/tmp/v11_prepare_audit.py`: real SASRec checkpoint/data, image I/O stubbed only. 530 train pairs; zero known-positive negatives; 265 events/258 users; identical SASRec baseline; held-out target perturbation preserves training hash. Correction changes 29 hard-negative choices and 179 SFT-negative choices.
- Historical cluster-aware reanalysis: `data/kaggle/v11-audit-20261002/historical_cluster_reanalysis.json`. Still inconclusive; historical mechanism acceptance withdrawn, not a fair rejection of CF-hardness.

## Checklist

- [x] Freeze source/config/data/checkpoint evidence for both completed runs.
- [x] Audit user identity, mapping, exclusions and held-out invariance.
- [x] Verify numerical parity with pinned upstream.
- [x] Inspect real checkpoint tensors and optimizer/scheduler state.
- [x] Verify corrected inference and gain threshold; run final regressions and end-to-end smoke before GPU submission.

## Risks

CPU parity does not claim bit-identical native-bf16 GPU behavior or native HaNoRec score reproduction. Training/evaluation samples remain event-based; uncertainty clusters by true user. Historical artifacts remain immutable.
