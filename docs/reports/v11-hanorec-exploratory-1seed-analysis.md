# Exploratory 1-seed (v11 preprocessing) — zero-GPU re-analysis (option A)

Date: 2026-09-23. GPU cost: 0 (CPU-only reconstruction).

- Run: `research/hanorec_cf_hardness/results/exploratory_1seed/` (`runtime_preprocessing_patch: v11-qwen25vl-min28-letterbox56`, seed 2024, 530 pairs, 25/25 users, 1 epoch, batch 3, max_pixels 4096).
- Script: the workspace analysis script `analyze_hanorec_exploratory.py`. It needs local SASRec checkpoints and data and is not included here. It is deterministic: a rerun produced a byte-identical JSON.
- Raw output: `v11-hanorec-exploratory-1seed-analysis.json` (same directory). Its `run_root` field records the original workspace path.
- Reconstruction: validation and test rows were rebuilt locally with the unchanged `prep.prepare`. All 9 pinned inputs are SHA-verified. Only image downloads are stubbed. For every arm, the rebuilt users, targets and candidate sets exactly match the downloaded arm predictions. This makes the SASRec-order baseline directly comparable.

## Verdict

**Exploratory signal: `NO_SIGNAL_ABOVE_RETRIEVER`.** The positive-looking validation contrasts come from 3–4 users. They are indistinguishable from chance. Every reranker arm sits below the frozen SASRec order on the same candidates. No claim transfers to the confirmatory question.

## Evidence

| Validation (25 users, 6 informative) | NDCG@10 |
|---|---:|
| SASRec order (no rerank) | **0.0914** |
| w0.5 real | 0.0761 |
| w0.5 shuffle | 0.0698 |
| w0.0 real | 0.0663 |
| w0.0 shuffle | 0.0636 |
| w1.0 real | 0.0622 |
| Random order (expected) | 0.0545 |
| w1.0 shuffle | 0.0510 |

- Only 6/25 validation users and 1/25 test users have the target inside the top-20 candidates. All other users contribute exactly 0 to every arm.
- Paired bootstrap, 20,000 resamples. Each CI crosses zero:
  - w0.5−w1.0 real: +0.0139, CI [−0.0029, +0.0448], 2 users up / 1 down.
  - w0.0−w1.0 real: +0.0041, CI [−0.0338, +0.0452].
  - real−shuffle: +0.0111, +0.0027 and +0.0063 at w = 1.0, 0.0 and 0.5, each resting on 3–4 changed users.
- Every arm minus the SASRec order is negative (−0.015 to −0.040). In each case 5 of 6 informative users get worse.
- The rerankers are weakly anti-correlated with the SASRec order: mean Spearman is −0.02 to −0.12. Their NDCG is at chance level. Analysis: the Qwen2.5-VL Yes/No score does not yet rank candidates better than random, so there is nothing for a w or real/shuffle contrast to modulate.
- Test (already exposed in the manifest, not used for the decision): 1 informative user. SASRec 0.0142; w1.0 shuffle 0.0126; every other arm 0.

## Training diagnostics (must be resolved before more GPU)

1. **`lambda_cf` is heavy-tailed.**
   - The CF margin is negative for 94.3% of pairs (mean −3.18), because the negative is SASRec's top non-excluded item.
   - The frozen formula `sigmoid(m)/sigmoid(mean m)` therefore gives weights from 0.00095 to 24.8 (median 1.04, p90 9.58). That is a ~26,000× spread.
   - The w0.0 DPO loss peaks at 50.0 (w0.5: 11.4; w1.0: 2.9). About a third of w0.0 loss values exceed 5.
   - CF-only training is therefore dominated by a few pairs. The formula was pre-registered as unclamped, so changing it is a protocol decision, not a bug fix.
2. **The visual input is nearly empty.**
   - max_pixels 4096 resizes the catalog to about 64×64 (the most common final size is 64×64 for 1,988 items).
   - Analysis: with Qwen2.5-VL's 28-px merged patch, that is about 4 visual tokens per image.
   - Real-vs-shuffle is still very different per user: Spearman between the two orders is only 0.47–0.49. But the metric difference stays at noise level, so the contrast cannot test the visual hypothesis at this budget.
3. **The reference may be saturated.** The SFT reference log P(Yes) on the train-positive probe is −4.8e−7 (real) and −1.2e−7 (shuffle). Only one probe was stored. Whether saturation holds on validation cannot be checked, because the SFT bundle (with `sft_validation_predictions`) was not downloaded.
4. **Padding position checked, not a bug.** The pinned tokenizer config (`Qwen/Qwen2.5-VL-3B-Instruct@66285546`) sets no `padding_side`, so HF defaults to right padding. `_batch_answer_logprobs` then reads the true last token.

## Consequence for option B

Option B was a score-only rerun on the existing checkpoints over ~200 informative users. It would most likely only confirm "below SASRec". Scaling the evaluation cannot create a signal the rerankers do not have. The next GPU spend should first answer a narrower question: **does any Qwen2.5-VL reranker (SFT-only or w1.0 real) beat the SASRec order on informative users at all?** Only if yes do w or real/shuffle contrasts become meaningful.

Decisions for the human (not taken here):

- Keep or bound the frozen `lambda_cf` formula, for example with a clamp or rank-based normalization. Either change needs a protocol revision.
- Accept max_pixels 4096 as "near text-only", or budget more memory for images.
- Download or regenerate the SFT bundle to test reference saturation on validation.

## Gaps

- The SFT-only baseline is unavailable locally (`sft_bundle.json` was not in the downloaded output).
- The DPO loss here is a per-micro-batch mean (`train.py:722-730`), and the scale of beta differs by arm. Loss magnitude is compared only within the same arm family. The w1.0 mean loss (~1.85) sits well above ln 2 from the very first micro-batch, even though a fresh DPO adapter should start near ln 2. This is unexplained; one candidate is NoDO noise acting on a saturated SFT reference. It needs a GPU probe.
