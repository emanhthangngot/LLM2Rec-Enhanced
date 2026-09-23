# v11 — HaNoRec CF-hardness reranking over frozen LLM2Rec + SASRec

A Qwen2.5-VL-3B reranker is trained with SFT then hardness-scaled DPO
(HaNoRec, pinned commit `587face7`) to rerank SASRec's real top-20 Games
candidates. The per-pair hardness mixes a HaNoRec semantic term and a
collaborative-filtering margin term:

```text
lambda(pair) = lambda_sem ** w  *  lambda_cf ** (1 - w)        # train.py:637-641
lambda_cf    = sigmoid(m) / sigmoid(mean(m)),  m = score(i+) - score(i-)   # unclamped, frozen
```

So `w = 1.0` is pure semantic hardness and `w = 0.0` is pure CF hardness. The
`shuffle` condition applies one frequency-matched, fixed-point-free item→image
bijection to SFT, DPO, hardness and evaluation prompts.

## Layout

| Path | Content |
|---|---|
| `prep.py` | Train-only pair construction from frozen SASRec candidates, frequency-matched derangement |
| `train.py` | SFT adapter, stacked DPO adapter with SFT-only reference, NoDO hooks, hardness mixture, candidate scoring |
| `audit.py` | Fail-closed protocol and artifact audit |
| `experiment.json`, `run-protocol.json` | Frozen protocol and pinned inputs |
| `test_hanorec_fidelity.py`, `test_hanorec_protocol.py` | Regression tests (`python -m unittest discover -s . -p "test_hanorec_*.py"` from this directory) |
| `scripts/build_hanorec_{kaggle,exploratory,scaled75}.py` | Generators of the self-contained Kaggle runners. Running them reproduces the committed `kaggle/*/runner.py` and metadata byte-for-byte |
| `kaggle/<kernel>/` | Every Kaggle kernel pushed for v11, with its `kernel-metadata.json` |
| `results/` | Downloaded arm results (see below) |

## Kernels and what they are evidence of

| Kernels | Code generation | Status | Evidence value |
|---|---|---|---|
| `hanorec-cf-hardness-preflight`, `-calibration`, `-real-run`, `-full-run-b` | first implementation, notebooks | calibration / early attempts | none (engineering only) |
| `hanorec-cf-hardness-sft` + 6 × `hanorec-cf-hardness-arm-w{00,05,10}-{real,shuffle}` | first implementation | `COMPLETE`, 265 test users, 530 train pairs, 1 seed | **withdrawn**: see below |
| 7 × `*-seed2025` | first implementation, seed 2025 | prepared; no execution record in the research workspace | none |
| `hanorec-faithful-signal-smoke` (versions 5–17) | corrected implementation | smoke `COMPLETE` (v14, v17) | correctness and timing only, stamped non-signal |
| `hanorec-exploratory-1seed` | corrected implementation | `COMPLETE` (v2): seed 2024, 25/25 users, 1 epoch, max_pixels 4096 | exploratory: `NO_SIGNAL_ABOVE_RETRIEVER` |
| `hanorec-scaled-75users` | corrected implementation | generated; no execution record | none |

The frozen 3-seed matrix (6 SFT parents + 18 DPO arms, 5 epochs) was projected
at 87.6–114.3 T4 GPU-hours against an 8-hour ceiling and never ran
(`BLOCKED_NO_FIT`).

## Results in `results/`

- `full_run_265/` — six `arm_result.json` from the first-implementation run.
  Real beats shuffle on point estimates at every weight (NDCG@10
  0.02420 vs 0.01493 at `w=0.0`), but the paired bootstrap CI crosses zero in
  every cell (`docs/reports/v11-hanorec-paired-bootstrap-audit.md`). The run's
  mechanism acceptance was later **withdrawn**. It leaked held-out catalog
  items, admitted training-history negatives, used an unmatched shuffle, used
  base-model reference semantics, and ran two full-dataset updates instead of
  the mini-batch recipe (`docs/reports/v11-hanorec-fidelity-and-protocol.md`).
- `exploratory_1seed/` — six arm results and the manifest from the corrected
  implementation. Every reranker arm scores below the frozen SASRec order on
  the same candidates (validation NDCG@10 0.0510–0.0761 vs 0.0914)
  (`docs/reports/v11-hanorec-exploratory-1seed-analysis.md`).

The package code (`prep.py`, `train.py`, `audit.py`) is the corrected
implementation. It is not the code that produced `full_run_265/`; that code
survives only inside the self-contained first-implementation runners under
`kaggle/`.
