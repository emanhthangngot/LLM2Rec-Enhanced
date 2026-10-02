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
| `prep.py` | Pinned CSV/downstream identity alignment; train-only negatives exclude all true-user TRAIN interactions; frequency-matched derangement |
| `train.py` | SFT adapter, stacked DPO adapter with SFT-only reference, NoDO hooks, hardness mixture, candidate scoring |
| `audit.py` | Fail-closed protocol and artifact audit |
| `experiment.json`, `run-protocol.json` | Frozen protocol and pinned inputs |
| `test_hanorec_fidelity.py`, `test_hanorec_protocol.py` | Regression tests (`python -m unittest discover -s . -p "test_hanorec_*.py"` from this directory) |
| `analysis.py` | Event-weighted user-cluster bootstrap CI, paired null sign-flip p-values, Holm, meaningful-gain verdicts |
| `scripts/build_hanorec_notebook.py` | Generates NEW `kaggle/hanorec-v11-user-corrected-{sft,probe,arms}/notebook.ipynb`. `repaired-*` notebooks are frozen historical evidence, never regenerated |
| `scripts/build_nodo_ablation.py` | Generates the prospectively registered matched semantic-only NoDO-on/off notebook; ten core sources remain pinned to the eligible SFT parent |
| `scripts/build_hanorec_{kaggle,exploratory,scaled75}.py` | Archived generators of the first-implementation / exploratory runners. They inline the **live** `prep.py`/`train.py`, so re-running them no longer reproduces the committed `kaggle/*/runner.py`; those runners are frozen historical evidence |
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
| `hanorec-v11-repaired-sft` v2 + `hanorec-v11-repaired-arms` v1 | later repaired implementation | all COMPLETE (4 SFT epochs, 1 DPO epoch, 4 arms) | mechanism-ineligible: 27 hard negatives/1 SFT negative were same-user TRAIN positives; 265 events are 258 users |
| `hanorec-v11-user-corrected-{sft,probe}` | current identity/statistics repair | both v1 COMPLETE and independently audited PASS | eligible parent; SFT performance inconclusive; probe perturbation is not evidence of training harm |
| `hanorec-v11-user-corrected-arms` | current identity/statistics repair | prepared, not submitted | generic CF matrix deferred until matched diagnostic evidence |
| `hanorec-v11-nodo-ablation` | frozen core plus matched-study orchestration | v1 submitted; completion not observed | registered semantic-only sigma0.05 vs0.0; no new training-effect result yet |

The frozen 3-seed matrix (6 SFT parents + 18 DPO arms, 5 epochs) was projected
at 87.6–114.3 T4 GPU-hours against an 8-hour ceiling and never ran
(`BLOCKED_NO_FIT`).

## Results in `results/`

- `full_run_265/` — six `arm_result.json` from the first-implementation run.
  Real beats shuffle on point estimates at every weight (NDCG@10
  0.02420 vs 0.01493 at `w=0.0`), but the paired bootstrap CI crosses zero in
  every cell (`docs/reports/v11-hanorec-cf-hardness.md` §2). The run's
  mechanism acceptance was later **withdrawn**. It leaked held-out catalog
  items, admitted training-history negatives, used an unmatched shuffle, used
  base-model reference semantics, and ran two full-dataset updates instead of
  the mini-batch recipe (`docs/reports/v11-hanorec-cf-hardness-design.md` §4).
- `exploratory_1seed/` — six arm results and the manifest from the corrected
  implementation. Every reranker arm scores below the frozen SASRec order on
  the same candidates (validation NDCG@10 0.0510–0.0761 vs 0.0914)
  (`docs/reports/v11-hanorec-cf-hardness.md` §3).

The package code (`prep.py`, `train.py`, `audit.py`) is the corrected
implementation. It is not the code that produced `full_run_265/`; that code
survives only inside the self-contained first-implementation runners under
`kaggle/`.

## Current audit commands and scope

`python audit.py protocol --protocol run-protocol.json` validates the protocol.
`python audit.py artifacts --protocol run-protocol.json --root <arms-output> --parent-root <sft-parent>` audits the explicit real-only four-arm matrix and validation predictions. The legacy `audit.py analyze` branch and its bootstrap-tail p-values were removed; current inference lives in `analysis.py` and the generated summaries.

Use `python scripts/build_hanorec_notebook.py --job sft`, then `--job probe` only after the new parent is COMPLETE and passes its gate. Read the no-training probe before choosing `--job arms`. Do not push dependent jobs against unfinished parents. The fixed one-seed event cohort can support a bounded investment decision, not native-score reproduction or a universal null-effect claim.

The selected follow-up is built by `python scripts/build_nodo_ablation.py`, not the generic CF matrix. Its scientific contract lives in `plans/261002-0142-v11-eight-hour-audit/reports/nodo-ablation-protocol.json`; submission and cumulative reservations live in `reports/gpu-budget.json` under that plan. Keep sigma variants in distinct output directories with their execution configs and hashed receipts; shared weight/condition names do not identify sigma. See `reports/nodo-ablation-verification.md` for pre-push proof and CPU/synthetic limits.

Audit report: `plans/261002-0142-v11-eight-hour-audit/reports/setup-audit-20261002.md`.
