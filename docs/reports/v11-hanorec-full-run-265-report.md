# Phase 2 Full Run — Verified Six-Arm Report

## Execution status

The resumable Option B design completed all seven real Kaggle kernels:

- SFT kernel: `trlxun/hanorec-cf-hardness-sft`, version 3, `COMPLETE`
- Arm kernels: six `COMPLETE` runs, each loading the frozen SFT bundle and running one DPO/reranking arm
- Account: `trlxun` (Kaggle account A)
- Dataset regime: real Games data, 530 train pairs, 265 evaluation users, 2,082 catalog items
- Each arm: 2 DPO steps, real checkpoint `.pt`, result JSON, and kernel log downloaded locally

No result below is inferred from kernel status alone. Every row was checked from the downloaded `arm_result.json`.

## Arm results

| Weight | Image condition | Status | NDCG@10 | Recall@10 | Candidate recall@20 | Elapsed seconds | Result artifact |
|---:|---|---|---:|---:|---:|---:|---|
| 1.0 | real | COMPLETE | 0.0227874 | 0.0603774 | 0.1169811 | 6888.44 | `research/hanorec_cf_hardness/results/full_run_265/hanorec-cf-hardness-arm-w10-real/arm_result.json` |
| 1.0 | shuffle | COMPLETE | 0.0147305 | 0.0377358 | 0.1169811 | 6891.42 | `research/hanorec_cf_hardness/results/full_run_265/hanorec-cf-hardness-arm-w10-shuffle/arm_result.json` |
| 0.0 | real | COMPLETE | 0.0242022 | 0.0641509 | 0.1169811 | 6611.65 | `research/hanorec_cf_hardness/results/full_run_265/hanorec-cf-hardness-arm-w00-real/arm_result.json` |
| 0.0 | shuffle | COMPLETE | 0.0149298 | 0.0377358 | 0.1169811 | 6405.25 | `research/hanorec_cf_hardness/results/full_run_265/hanorec-cf-hardness-arm-w00-shuffle/arm_result.json` |
| 0.5 | real | COMPLETE | 0.0220688 | 0.0566038 | 0.1169811 | 6099.90 | `research/hanorec_cf_hardness/results/full_run_265/hanorec-cf-hardness-arm-w05-real/arm_result.json` |
| 0.5 | shuffle | COMPLETE | 0.0150677 | 0.0377358 | 0.1169811 | 6053.40 | `research/hanorec_cf_hardness/results/full_run_265/hanorec-cf-hardness-arm-w05-shuffle/arm_result.json` |

Checkpoint artifacts are present beside each result JSON as `arm_w*.pt`; the corresponding kernel logs are present in each arm directory.

## Direct observations

- Real images improve `Recall@10` over shuffled images for every weight:
  - `w=1.0`: 0.0603774 vs 0.0377358
  - `w=0.0`: 0.0641509 vs 0.0377358
  - `w=0.5`: 0.0566038 vs 0.0377358
- Real-image `NDCG@10` is also higher for every weight:
  - `w=1.0`: 0.0227874 vs 0.0147305
  - `w=0.0`: 0.0242022 vs 0.0149298
  - `w=0.5`: 0.0220688 vs 0.0150677
- Candidate recall is identical across all six arms (`0.1169811`). This is expected: the arm reranks the fixed candidate set; it does not change candidate generation.
- Among real-image arms, `w=0.0` has the highest observed NDCG@10 and Recall@10 in this run. This is a descriptive result, not a statistical claim: the evaluation has 265 users and no repeated-seed confidence interval.

## Scope and limitations

This is the largest completed real subset in the current execution: 530 training pairs and 265 evaluation users, not the literal 122,577-row corpus. The six arms share the same frozen candidate construction and evaluation users, so the image-condition comparison is paired at the pipeline level. The report does not claim generalization beyond this Games regime, the selected subset, or the single training seed.

## Next phase

Phase 2 execution is complete. Phase 3 should independently audit the downloaded JSON/checkpoint/log set against the pre-registered gates, then decide whether the observed result supports a follow-up experiment or a stop/null-result conclusion.
