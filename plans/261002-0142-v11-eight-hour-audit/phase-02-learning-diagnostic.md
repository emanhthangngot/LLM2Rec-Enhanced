---
phase: 2
title: "Correct-label SFT and clean/noisy diagnostic"
status: in-progress
priority: P1
---

# Phase 2: Correct-label SFT and clean/noisy diagnostic

## Overview

Historical parent labels fail the TRAIN-known-user exclusion contract. Train a fresh SFT parent with corrected labels before interpreting DPO learning. Preserve sampled events, candidate protocol, seed 2024, pixels 16384, four SFT epochs, batch2/accumulation8 and LR1e-4 so the correction is not confounded with a new training recipe.

## Current behavior and target

`prep.py` now recovers true user clusters from pinned CSVs; `train.py::_learning_probe` records clean/noisy chosen-minus-rejected and policy-minus-reference gaps with private noise RNG. `scripts/build_hanorec_notebook.py` produces new `hanorec-v11-user-corrected-sft`, reuses only hash-verified image bytes from the archived parent, and refuses source or image mismatches. No historical adapter is used.

## Verification

- CPU regression and real-data held-out invariance must pass before push.
- Tiny real Qwen2.5-VL SFT→DPO smoke must emit both train/validation before/after probes, exact counts, parity0 and frozen reference state.
- Kaggle MCP SFT push: account a (`trlxun`), T4, timeout_seconds 10800 (3 GPU-session hours maximum).
- On COMPLETE download `job_manifest.json`, bundle and checkpoint; verify hash, clusters, zero seen negatives, counts and saturation detector before accepting the new parent.
- Clean/noisy GPU probes require that new parent and are limited to the separate 0.5-hour reserve before deciding on a matched follow-up. Validation non-target candidates are unlabeled, not proven dislikes.

## Current external wait

SFT v1 COMPLETE and audited: https://www.kaggle.com/code/trlxun/hanorec-v11-user-corrected-sft .
Parent eligibility PASS: hashes/identity pins, zero known-positive negatives, 268 updates, finite adapter values and user-cluster metrics verified. Notebook elapsed150.4min; full log observed9052.65s. Reference gate passes; ranking superiority remains INCONCLUSIVE.
Clean/noisy probe v1 COMPLETE and audited: https://www.kaggle.com/code/trlxun/hanorec-v11-user-corrected-probe . Runtime3.084min (last log203.03s), T4, no training. Parent/all ten source hashes match, finite outputs and clean policy-reference delta0 on all 32 examples. In-run weight/CPU-CUDA RNG assertions passed. Absolute margins shrink from 5.125 to0.258 (train) and6.320 to0.303 (validation) on this fixed noise draw; pair-order changes3/8 and5/8. This is transient perturbation, not proof of training harm or ranking failure. Read a matched NoDO-on/off training comparison before assigning causality. Detailed audit: parent workspace `data/kaggle/hanorec-v11-user-corrected-probe/probe_result_audit.json`.

## Checklist

- [x] Complete and audit corrected-label SFT within three GPU-hours.
- [x] Measure clean/noisy fixed train/validation probes on a scientifically eligible parent.

## Stop gate

No DPO follow-up on a failed/incomplete parent or a failed saturation detector. Failed jobs count against the 8-hour total. Submission is not execution proof; no automatic polling or speculative conclusion.
