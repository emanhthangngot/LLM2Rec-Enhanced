# Zero-GPU caption screening — pre-registered protocol

Registered 2026-09-30, before any AmazonMix-6 domain result was computed. The
only prior outcome is the Video_Games paired pilot (`NO_GAIN_FROM_CAPTIONS`,
`docs/reports/v10-video-games-paired-pilot.md`).

## Question

In each AmazonMix-6 domain, do image captions carry information about the next
purchased item **beyond what the titles already say**?

## Inputs (pinned)

- Transition pairs: output of Kaggle CPU kernel
  `trixuanle/llm2rec-caption-screening-pairs-v1`, which reads
  `trixuanle/llm2rec-amazonmix6-5core`. The kernel manifest records source and
  output SHA-256 values. Only the `train` split is used.
- Florence-2 captions for all six domains: v10 corpus
  `v10-full-corpus.tar.gz` (SHA-256 `22bc6809…5657`); the shard manifest
  (`521849b2…84d7`) and every shard hash are verified.
- Qwen3-VL captions for Video_Games (robustness only): `results/video_games_pilot/corpus_v1`.
- Join check: each CSV `item_asin` must equal the caption record's `asin` at the
  same `global_id`.

## Measurement

- Tokens are lowercase alphanumeric strings of length ≥ 2. A fixed stoplist is
  removed. Caption tokens whose within-domain document frequency is above 0.3
  are dropped as template words. Similarity is IDF-weighted Jaccard.
- `s_title(a,x)` is the similarity of the two titles.
- `s_cap(a,x)` is the similarity of the **residual** captions. The residual of
  an item's caption drops every token that appears in either title of the pair.
- Positives are the unique `train` pairs (last history item `a` → target `b`)
  with `a ≠ b` and a usable caption on both sides.
- Each positive gets K = 5 negatives `b'` from the same domain. Each negative is
  in the same target-popularity quintile as `b` and the same caption-length
  tertile as `b`. It is not `a`, not `b`, and `(a, b')` is not a positive pair.
  The RNG seed is 20260930.
- A triple `(a, b, b')` scores 1 if `s(a,b) > s(a,b')`, 0.5 on a tie, and 0
  otherwise. AUC is the mean score over triples.
- The **main metric M** is the AUC of `s_cap` restricted to title-tied triples,
  where `s_title(a,b) == s_title(a,b')`. In these triples the title cannot
  separate the positive from the negative.
- Also reported: the AUC of `s_title`, the unconditioned AUC of the residual
  and raw captions, the share of title-tied triples, and the share of ties.
- The **shuffle control** recomputes M after replacing every caption with the
  caption of a fixed within-domain derangement donor.
- The 95% CI comes from 1,000 bootstrap resamples of positives.

## Decision rule

The validity gates must all pass; otherwise the proxy is invalid and no GPU run
is scheduled.

- **G1:** in every domain, the shuffle-control M lies in [0.49, 0.51] and its
  CI contains 0.5.
- **G2:** in every domain, AUC(`s_title`) ≥ 0.55. This is the positive control
  that the pairs carry signal.
- **G3 (calibration):** Video_Games (Florence) must not be among the top 2 of
  the 6 domains by M. The paired pilot found no gain on Video_Games, so a proxy
  that ranks it high does not predict training outcomes.

Selection: take the domain with the largest M. It qualifies for a paired GPU
run only if M − 0.5 ≥ 0.02 **and** the bootstrap 95% CI of
(M_domain − M_Video_Games) excludes 0. If no domain qualifies, close the caption
direction (option C).

## Amendment 1 — registered 2026-09-30, before any non-Video_Games result

**Motivation.** A calibration run on Video_Games came first; Video_Games is
the only domain with a known training outcome. With **Qwen3-VL** captions, the
type of caption actually trained on, M was 0.581 [0.578, 0.583], a clear signal,
yet the paired training showed no gain. With Florence captions M was 0.519. The
shuffle and title controls were clean in both runs. The excess residual tokens
are genre, style and product-type words (`cartoonish`, `action`, `mature`,
`anime`, `plastic`, `included`). The interaction data itself may already carry
this information. M measures information beyond the **title**, but not beyond
**collaborative signal**.

**Added metric M_cf (exploratory).** It uses the same tokens, weights, negative
matching and seed as M, with these changes:

- Positives are the unique **test**-split pairs (last history item → target).
  These transitions are unseen in training.
- Popularity and CF come from the **train** split only. `s_cf(a,x)` is the
  number of train transitions between `a` and `x` in either direction.
- M_cf is the caption-residual AUC over triples that are both title-tied and
  CF-tied (`s_cf(a,b) == s_cf(a,b')`). The shuffle control is computed the same
  way.

**Calibration gate for the amendment (C_cf).** M_cf for Video_Games with Qwen3-VL
captions must be ≤ 0.52. Otherwise the proxy still reports caption information
on the one domain where training showed no gain. Neither M nor M_cf would then
be a valid predictor of training benefit, and no GPU run is selected on either
basis.

The pre-registered decision on M is still computed and reported unchanged; M_cf
is reported next to it.

## Amendment 2 — late-fusion headroom check (zero GPU), registered 2026-09-30 before running

**Question.** On Video_Games, does adding the caption channel to the trained
title-only embedding improve next-item discrimination?

- **Scores.** `s = cos(e_title(a), e_title(x)) + λ · s_cap(a, x)`.
  `e_title` is the pilot title-only embedding. `s_cap` is the title-residual
  Qwen3-VL caption similarity defined above.
- **Triples.** Positives are the unique valid-split and test-split pairs, with
  negatives matched as above. Popularity is computed from train.
- **Tuning.** λ is chosen on **valid** from {0, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2}.
  It is reported, never tuned, on test.
- **Control.** The same procedure is repeated with the shuffled (derangement)
  caption, and λ is chosen on valid again.
- **Pass (GPU late-fusion run allowed).** Both conditions must hold on test:
  - the gain `AUC(fused) − AUC(title)` is ≥ 0.005, and its 95% bootstrap CI
    (1,000 resamples of positives) excludes 0;
  - the gain for real captions minus the gain for shuffled captions has a
    95% CI that excludes 0.

  Otherwise no GPU run is made.

## Amendment 3 — GPU late-fusion run, registered 2026-09-30 before pushing

Amendment 2 passed: on test, the fused score gained +0.032 AUC, CI [0.027, 0.036];
the shuffled caption gained +0.0005.

**Setup.** All arms use the pinned LLM2Rec SASRec with `SASREC_ARGS` and seeds
2024/2025/2026, on the Video_Games test split. The frozen item-feature matrix
is the only difference between arms:

- `title`: the pilot title-only embedding, unchanged. It must reproduce the
  pilot result: Recall@10 within 0.002 of the pilot value for every seed.
  Otherwise the run is invalid.
- `fused_real`: `[e_title ; s · e_cap]`.
  - `e_cap` is the L2-normalized IDF-weighted bag of an item's Qwen3-VL caption
    tokens, after removing template tokens and the item's own title tokens.
    It is reduced to 256 dimensions by an unsupervised SVD fitted on item
    features only; the SVD uses no interactions.
  - Items without a usable caption get a zero vector.
  - `s` is the mean row norm of `e_title`.
- `fused_shuffle`: identical, except that `e_cap` rows are permuted by the
  seed-20260930 derangement over the 9,517 items.

**Pass** requires both of the following:

- `fused_real` beats `fused_shuffle` on Recall@10 in 3/3 seeds, and
- the mean Recall@10 of `fused_real` is ≥ 1.02 × the mean Recall@10 of `title`.

NDCG@10 is reported alongside. A pass supports "late fusion recovers caption
signal". It does not claim that the result generalizes to other domains.
