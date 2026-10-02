# v11 setup audit — 2026-10-02

## Verdict

Historical `hanorec-v11-repaired-sft` v2 and `hanorec-v11-repaired-arms` v1 executed successfully but failed the true-user TRAIN-known-negative contract. Preserve their descriptive metrics and artifacts; withdraw their mechanism acceptance. The adopted HaRS/DPO/NoDO equations passed CPU numerical parity. Correct the data/statistics causes before spending further GPU time.

## Evidence ledger

| Contract | Verdict | Observed evidence |
|---|---|---|
| Immutable run/source/data provenance | PASS | 12 artifact hashes, 9 executed source snapshots, 9 input pins in parent workspace `data/kaggle/v11-audit-20261002/freeze_manifest.json` |
| Source CSV item mapping | PASS | 122577 train rows and 15322 validation rows align exactly with downstream sequences after adding 1 to source item IDs |
| Independent-user interpretation | BREAK, repaired | 530 train events belong to 505 users; 265 validation events belong to 258 users. Stable event IDs retained; pseudonymous `cluster_id` now identifies real users |
| Historical known-positive negative exclusion | BREAK, repaired | 27 hard negatives are same-user TRAIN-known positives (12 earlier/current, 15 later TRAIN); 1 SFT random negative also known. New preparation excludes all same-user TRAIN interactions and restricts negative universe to train-only items |
| Repaired preparation | PASS smoke | 530 pairs, zero known-positive negatives; 29 hard choices and 179 random choices change. Baseline/candidates unchanged. Mutating held-out targets does not alter train/hash/shuffle. Image I/O alone stubbed in CPU smoke |
| Temporal split on selected validation events | PASS scoped | Zero of the 265 selected validation events has a later TRAIN interaction for the same source user |
| HaRS/top-K, responsiveness | PASS numerical | Exact pinned upstream `587face7`, archive SHA `48c52e...33d6a6`; fp64/fp32 arrays and statistics agree |
| DPO objective and gradients | PASS numerical | Executed actual local loss AST, compared loss/gradients to upstream on fixed inputs including heavy hardness values |
| NoDO outputs/gradients/cleanup | PASS numerical | Same weights/noise generator; outputs and gradients equal, parameters unchanged, hooks removed |
| Actual checkpoints | PASS scoped | All four contain 504 language-only DPO tensors; all 252 B tensors nonzero; optimizer step=34; scheduler last_epoch/T_max=34/34; final LR=0.0. Parent hash matches. No embedded SFT tensor snapshot in arm file; frozen-reference evidence additionally relies on in-run invariants and fresh-model smoke |
| Historical bootstrap p-values | BREAK, repaired | Old code used bootstrap-tail proportions, not a null distribution. New code uses whole-user sign flips under label-exchangeability assumptions; exact for <=16 active clusters, otherwise Monte Carlo |
| Meaningful gain | BREAK, repaired | Old branch allowed an upper CI bound crossing +0.002 to count as sufficient. New meaningful-gain branch requires lower bound >=+0.002; positive but threshold-crossing CI is explicitly uncertain |
| Duplicate prediction events | BREAK, repaired | Old dictionary silently overwrote duplicates; new cohort assembly rejects duplicate or missing-informative events |
| Reference-cache storage | Memory risk, repaired | A detached 4-byte scalar retained 2430976 bytes of vocabulary storage; cloning keeps 4 bytes with bitwise-equal value (~614 MiB avoidable retention over 265 batches) |

## Corrected historical reanalysis (not new mechanism evidence)

True-user clustered CIs still cross zero: SFT-SASRec [-0.010257,0.035228]; CF-SFT [-0.001168,0.000830]; CF-semantic [-0.000478,0.007333]; mixed-semantic [-0.006004,0.001582]; CF-constant [-0.000347,0.008565]. Whole-cluster permutation p-values differ from the old bootstrap-tail numbers. Historical reanalysis is exploratory and cannot rehabilitate faulty training labels.

## Exercised verification

- Unit suite: 31 passed, zero skipped. Expected negative-fixture FAIL JSON is not a failing suite.
- Real preparation smoke with genuine SASRec checkpoint/data: PASS, image download I/O stubbed only.
- Pinned oracle numerical and real checkpoint audit: PASS.
- Fresh tiny Qwen2.5-VL models with actual pinned processor: SFT plus semantic and mean-control DPO COMPLETE, one update per smoke stage, parity0, true-cluster metadata, empty test predictions, controls saved. Frozen SFT state unchanged.
- Learning probes contain before/after train and validation. Direct assertions: no global torch RNG consumption, parameter changes or leaked hooks. Synthetic catalog/tiny weights and CPU telemetry are smoke boundaries, not scientific results.
- Three generated notebooks compile (10/5/11 code cells); all ten embedded source blobs match on-disk hashes. Generated no-training probe cells pass a tiny-model parent fixture; artifact CLI with a separate parent directory passes.
- Final review fixes preserve hash-locked image bytes even after download fallback, reserve 1200 seconds after training, retain the two-test Holm family, label the secondary control, and prevent positive CI from overriding an unrejected null. Nominal measured-time simulation runs all four arms in about 4.084 hours with ETA factor1.15. Slower sessions can still be partial or fail-closed; no universal no-crash promise.

## Scientific limits and next gate

Game-specific binary next-token objective (no assistant terminator/EOS in the scoring head), stacked adapters, sampled events, 16384-pixel images, emulated bf16 on T4 and short schedules are explicit adaptations. No native-score or bit-identical LLaMA-Factory-internals claim. Permutation assumes within-user label exchangeability; cluster bootstrap conditions on fitted models and does not estimate seed uncertainty. Validation negatives are unlabeled candidates, not proven dislikes.

Use a fresh corrected-label parent under a three-hour cap, then inspect its gate and clean/noisy probes before selecting a matched follow-up under the remaining eight-hour ceiling. No new GPU execution result is claimed in this report.

## Submission evidence

Repository Kaggle MCP (`rs-wiki-kaggle` v0.3.0) returned \"Kernel version 1 successfully pushed\" for `trlxun/hanorec-v11-user-corrected-sft`; no invalid-source or slug warnings. T4, timeout10800s. Runtime completion/results have not been observed. Budget ledger records a maximum three-hour reservation, not fabricated actual GPU spend. Probe and follow-up notebooks are ready but deliberately not submitted against an unfinished parent.
