# HaNoRec fidelity and Games protocol ledger

## Decision boundary

This package reproduces the observable HaNoRec `hit=1` trainer contract on a deliberate Games adaptation. It does **not** claim native HaNoRec published-score reproduction. A Kaggle `COMPLETE` smoke run is execution evidence only; it is not signal evidence.

## Authority

- HaNoRec: commit `587face74524e4553b5a7aa295fe962004682382`, archive SHA-256 `48c52e...3d6a6`.
- LLM2Rec: commit `73b481f710f67166ab958f4985d27b27fb410871`, archive SHA-256 `6bcee7...c1cc8`.
- LLaMA-Factory: `llamafactory==0.9.3.dev0` from the pinned upstream requirement. The requirement does not expose a source commit, so this work proves the HaNoRec observable trainer behavior from the pinned source rather than claiming byte-identical dependency internals.

## Frozen behavior matrix

| Contract | Upstream authority | Corrected Games implementation | Adaptation label |
|---|---|---|---|
| SFT precedes DPO | `README.md`, `microlens_hit1.yaml` | SFT adapter is trained first; DPO uses a fresh adapter over the frozen SFT adapter | Games model/data and subset only |
| Reference | `trainer/dpo.py` | Reference activates SFT adapter only; DPO adapter is excluded | PEFT adapter stacking is explicit |
| DPO objective | `trainer/dpo.py` | Same preference-logit subtraction, sigmoid loss, beta floor, and detached hardness/statistics | Candidate scoring is Games-specific |
| Responsiveness | `trainer/dpo.py`, `hars/math.py` | One global mini-batch statistic, requiring at least three examples | No distributed gather in single-GPU smoke |
| NoDO | `nodo/hooks.py` | Forward-hook perturbation with generator, no parameter mutation, exception cleanup | Single-GPU smoke |
| Hardness | `hars/hardness.py` | Fused train-only title/visual catalog and exact Top-K math | Games CF hardness is an added condition |
| Data preparation | upstream preference builder | Train pairs and hardness use only train rows/items; held-out rows are inference-only | True next-item Games targets |
| Evaluation | README (`AUC` for native hit-1) | Frozen SASRec candidate set; report NDCG@10/Recall@10 and candidate recall | Deliberate Games reranking adaptation |

## Data contract

The train item universe is derived before validation/test rows are consulted. Negatives exclude padding, the positive, every observed item in the user row, and train-known future items. Evaluation targets and candidates can add inference assets but cannot change the training pair, hardness, or shuffle hashes. Selection is seeded at user-row level; no fixed prefix is used.

The shuffle control is a deterministic fixed-point-free bijection within frequency strata. Singleton strata are merged with adjacent strata before derangement. The same mapping is applied to SFT prompts, DPO prompts, hardness visual inputs, and evaluation prompts for the shuffle condition.

## Training contract

The implementation uses real micro-batches, `drop_last`, gradient accumulation, partial-accumulation scaling, AdamW, cosine schedule, warmup zero, gradient clipping one, and deterministic seeds for Python/NumPy/Torch CPU/CUDA/data/noise generators. Checkpoints carry model, optimizer, scheduler, RNG, stage, and count state. `COMPLETE` requires every expected stage and exact observed counts.

A correctness smoke is intentionally smaller than the five-epoch research recipe and is stamped as non-signal. Full signal execution is quota-gated after smoke timings and must use seeds 2024/2025/2026, real/shuffle parents, weights 0/0.5/1, validation selection, and untouched confirmation scoring.

## Withdrawn historical evidence

The old 530-pair pilot (the 265-user six-arm run) remains stored under `research/hanorec_cf_hardness/results/full_run_265/`, and its reports (`v11-hanorec-full-run-265-report.md`, `v11-hanorec-paired-bootstrap-audit.md`) remain historical artifacts. Its mechanism acceptance is withdrawn because it leaked held-out catalog items, admitted training-history negatives, used an unmatched shuffle, used base-model reference semantics, and executed two full-dataset updates instead of the configured mini-batch recipe.
