# LLM2Rec-Research: Version-Tracked Multimodal Extensions to LLM2Rec

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c.svg)](https://pytorch.org/)
[![Upstream](https://img.shields.io/badge/Upstream-LLM2Rec%20KDD%2725-informational.svg)](https://github.com/HappyPointer/LLM2Rec)
[![Status](https://img.shields.io/badge/Status-Research%20in%20Progress-yellow.svg)](#version-generations)
[![License](https://img.shields.io/badge/Upstream%20License-absent-lightgrey.svg)](#citation--license)

> **Independent research fork**
> **Focus:** does aligning LLM2Rec's text-only sequential-recommendation pipeline with visual and collaborative-hardness signals improve ranking, and under what conditions?
> **Upstream:** He et al., *LLM2Rec: Large Language Models Are Powerful Embedding Models for Sequential Recommendation*, KDD 2025 ([HappyPointer/LLM2Rec](https://github.com/HappyPointer/LLM2Rec), pinned commit `73b481f`).

---

## Overview

**LLM2Rec-Research** takes the vanilla LLM2Rec pipeline — CSFT → MNTP → SimCSE → item-embedding table → sequential recommender — and asks one question repeatedly, at a different intervention point each generation: *does adding non-text evidence (image content, or collaborative "hardness") improve recommendation, once every non-causal explanation is controlled for?*

The upstream paper text-conditions a small LLM (`Qwen2-0.5B`) into an item-embedding generator using only item titles. Nine generations of extension work, `v0` through `v11`, are tracked below as one lineage, exactly as they were designed, tested, and — for six of them — **rejected with a documented reason**. Two generations, `v10` and `v11`, are currently running and have **no concluded result yet**; this README states that explicitly rather than presenting partial numbers as findings, per the evidence discipline this fork was built under (`AGENTS.md` in the parent workspace: "No fabricated citations. If a claim lacks a source, write `(unsourced)`.").

Two structural lessons carry across every generation:

1. **Visual fusion that reshapes the candidate item-embedding table or the user-sequence state is dangerous.** It changes candidate geometry and history representation simultaneously, so a positive result cannot be attributed to "the image" — see `v9.0` and `v9.1` below.
2. **A positive result on one dataset is not a method.** Every generation that looked promising on `Games` was required to clear a **second, harder gate** — a frequency-matched non-semantic control (Gaussian noise, PCA of the text embedding itself) and, where applicable, a genuinely different dataset (`Sports`) — before being called visual-specific. Most failed that second gate; the ones that didn't are marked `PASS` explicitly.

---

## Version Generations

| Version | Codename | Core Idea | Status | Dataset(s) | Backbone(s) | Result vs. Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **v0** | `LLM2Rec-Baseline` | Faithful text-only compatibility reproduction (CSFT → MNTP → SimCSE → SASRec/GRU4Rec/BERT4Rec) under a single-T4 compute budget | Closed — reference | Games (dev), Arts (replication), AmazonMix-6 (pretrain) | SASRec, GRU4Rec, BERT4Rec | NDCG@10 `0.0504` (IEM ckpt-1000) vs. paper `0.0521` (`-3.3%` gap) |
| **v9.0** | `Additive-Item-Fusion` | CLIP image feature → MLP → summed directly into the item embedding | **Rejected** | Games | SASRec | Real-vs-shuffle NDCG@10 swings `-16.3%` to `+1.8%`; random visual input alone damages ranking |
| **v9.1** | `Sequence-Side-Fusion` (S1/S2) | Visual signal folded into the user-sequence hidden state | **Rejected** | Games | SASRec | `92%` of the apparent gain traced to `740` immediate-repeat rows; a recency-only reranker with no image input beats it on all 3 seeds |
| **v9.2** | `Frozen-Score-Residual` (I1‑A → C0.5, 10 sub-experiments) | `score_final = score_text + α·z_visual` at the ranking boundary; text recommender and candidate table stay frozen | **Closed — conditional positive, program exhausted (`STOP2`)** | Games (dev), Sports (transfer) | SASRec, BERT4Rec, GRU4Rec (stress test) | `PASS` on Games/BERT4Rec (`+7.2%` NDCG@10 vs. text) and the C0.5 matched-shuffle contrast (`+0.0015` NDCG@10, 95% CI above zero); `FAIL` on Sports transfer and on GRU4Rec, where a text-PCA control beats real visual |
| **v10** *(active)* | `Caption-Augmentation` | Florence‑2 offline image→text captions injected into CSFT/MNTP/SimCSE input history via 5 controlled arms (`title-only`/`null`/`real`/`shuffle`/`paraphrase`) | **Running — corpus generation resumed, no downstream training yet** | Games (dev), Arts (replication), AmazonMix-6 (pretrain); Baby reserved sealed | SASRec (matched) | — |
| **v11** *(active)* | `HaNoRec-CF-Hardness` | Freeze LLM2Rec + SASRec; blend a CF score margin (`w ∈ {1.0, 0.5, 0.0}`) with HaNoRec's semantic hardness to weight a DPO-tuned Qwen2.5‑VL reranker over SASRec's real top‑20 | **Running — SFT + 6 branch kernels mid-verification** | Games only | Qwen2.5-VL reranker over SASRec | — |

Blank cells in the last column are intentional: `v10`/`v11` have not cleared the matched-control gate that every closed generation above was required to pass before a result was recorded (see `AGENTS.md`: "No fabricated citations. If a claim lacks a source, write `(unsourced)`."). `v10` and `v11` inherit the same discipline as `v0`–`v9.2`: frozen upstream checkpoints where possible, matched non-semantic controls, seed-paired bootstrap, and a hard rule that corpus completion, a running kernel, or a positive training loss is not itself evidence of effectiveness.

---

## Architecture

### Shared upstream backbone (all versions)

```mermaid
flowchart LR
    subgraph Pretrain["Upstream LLM2Rec (v0)"]
        A[AmazonMix-6 item titles] --> B["Stage 1: CSFT on Qwen2-0.5B"]
        B --> C["Stage 2a: MNTP (bidirectional)"]
        C --> D["Stage 2b: SimCSE / item-level contrastive"]
        D --> E[Item embedding table]
    end
    E --> F[Adapter, 128-d projection]
    F --> G[SASRec / GRU4Rec / BERT4Rec]
    G --> H[Full-catalog ranking]
```

Every version below inserts its intervention at a different point in this backbone, which is exactly what makes the comparisons across generations legible: `v9.0` at `E` (item embedding), `v9.1` inside `G` (sequence state), `v9.2` after `H` (score boundary), `v10` before `B` (CSFT input text), `v11` after `H` (reranking a frozen `H`'s candidates).

### v9.0 — Additive item-embedding fusion (rejected)

```mermaid
flowchart LR
    A[CLIP image embedding, 512-d] --> B[Trainable MLP, ~575K params]
    C[Text item embedding, from E] --> D["item_embedding_final = text + MLP(image)"]
    B --> D
    D --> E[SASRec candidate table]
    E --> F[Full-catalog ranking]
    style D fill:#ffe0e0,stroke:#c00
```

Rejected because the visual branch (`~575K` params) is `5x` larger than the text adapter (`~115K` params) it is added to, so it dominates and reshapes the candidate geometry instead of augmenting it — confirmed by a **shuffled** (mismatched item-image) visual branch damaging ranking almost as much as the real one.

### v9.1 — Sequence-side fusion, S1/S2 (rejected)

```mermaid
flowchart LR
    A[User history images] --> B[Visual history encoder]
    C[User history text] --> D[Text sequence encoder]
    B --> E[Fused user-sequence state]
    D --> E
    E --> F[SASRec scoring head]
    F --> G[Full-catalog ranking]
    G --> H{Split by target type}
    H --> I[Immediate-repeat rows, 740/15323]
    H --> J[Novel-item rows, 14583/15323]
    style E fill:#ffe0e0,stroke:#c00
```

Rejected because `92%` of the apparent gain lived in the immediate-repeat slice `I`, where a plain recency reranker (no image input at all) already wins — the fused state `E` was learning "copy the last item," not visual semantics.

### v9.2 — Frozen score-level residual (closed, conditional positive)

```mermaid
flowchart TB
    subgraph TextPath["Text path — frozen after training"]
        A[Text user state] --> B[score_text]
    end
    subgraph VisualPath["Visual path — no gradient into text model"]
        C[User visual history] --> D[Recency-weighted visual profile]
        E[CLIP item embedding] --> F[Cosine similarity]
        D --> F
        F --> G["Standardize -> z_visual"]
    end
    B --> H["score_final(u,i) = score_text(u,i) + alpha * z_visual(u,i)"]
    G --> H
    H --> I[Full-catalog ranking]
    I --> J{"alpha = 0 ?"}
    J -->|yes| K[Must equal text-only ranking exactly - G0 parity check]
    J -->|no| L[Validation-selected alpha, test labels never used]
```

The `alpha = 0` parity check (`J` → `K`) is the invariant that makes this design auditable: every sub-experiment in the `v9.2` family (`I1-A`, patched rerun, joint-trained, `R1` controls, exposure-gated, `C0.5`) reuses this exact diagram and only changes how `z_visual` or `alpha` is fit.

### v10 — Caption augmentation (active)

```mermaid
flowchart TB
    subgraph Corpus["Offline corpus generation (per catalog item i)"]
        A[AmazonMix-6 titles + item IDs] --> D[Immutable catalog order]
        B[Catalog image] --> E["Florence-2-large, prompt: &lt;CAPTION&gt;"]
        A --> F["Qwen2.5-3B title-only paraphrase, no image"]
    end
    subgraph Arms["5 arm-specific item texts, same history-retention policy"]
        D --> G1[title-only: T_i]
        D --> G2["null: T_i + 'unavailable'"]
        E --> G3["real: T_i + caption(image_i)"]
        E --> G4["shuffle: T_i + caption(image_perm(i))"]
        F --> G5["paraphrase: T_i + paraphrase(T_i)"]
    end
    G1 & G2 & G3 & G4 & G5 --> H[Common retained history suffix]
    H --> I["CSFT — target stays original title"]
    I --> J[MNTP] --> K[SimCSE] --> L[Item embeddings per arm]
    L --> M[Matched SASRec, identical config per arm]
    M --> N[Full-catalog ranking]
    N --> O["Paired-seed uncertainty audit: real vs each control"]
```

The five arms in `Arms` are five causal contrasts sharing one pipeline, not five different pipelines: `real` vs `shuffle` isolates whether the image is linked to the *correct* item; `real` vs `paraphrase` isolates whether the gain is "more text" rather than "visual evidence."

### v11 — HaNoRec CF/history-aware hardness (active)

```mermaid
flowchart TB
    subgraph Frozen["Frozen retrieval stage — no gradient here"]
        A[Frozen LLM2Rec title embeddings] --> B[Frozen SASRec]
        C[Train histories + targets] --> B
        B --> D[Hardest non-target negative per user]
        B --> E[Real top-20 test candidates]
    end
    subgraph Hardness["Hardness signal construction"]
        F[Item titles + images] --> G[Qwen2.5-VL semantic embeddings]
        G --> H["HaRS semantic hardness, lambda_sem"]
        D --> I["CF score margin: m_CF = score(i+) - score(i-)"]
        H --> J["Geometric mixture by w in {1.0, 0.5, 0.0}"]
        I --> J
    end
    subgraph Reranker["Qwen2.5-VL reranker training"]
        C --> K[Qwen2.5-VL LoRA SFT]
        J --> L[Hardness-scaled DPO]
        K --> L
    end
    L --> M["Yes-minus-No candidate score"]
    E --> M
    M --> N[Reranked top-20]
    N --> O[NDCG@10 / Recall@10 / candidate Recall@20]
```

`w = 1.0` uses pure CF hardness, `w = 0.0` uses pure HaNoRec semantic hardness, `w = 0.5` mixes both — the six branch kernels are `{w} x {real, shuffle}`, where `shuffle` breaks the item-image link the same way it does in `v10`, to isolate whether the hardness signal is doing anything beyond a generic difficulty prior.

---

## Dataset Selection Criteria

Every generation in this lineage is evaluated on a subset of five Amazon-review domains, chosen for a specific methodological role rather than for convenience. The core discipline is separating **datasets you are allowed to look at repeatedly while designing a method** from **datasets you look at exactly once to confirm it** — this is what stopped several generations (`v9.0`, `v9.1`, `v9.2`'s Sports transfer) from being reported as more general than the evidence supports.

| Dataset | Role | Why this role | Look policy |
| :--- | :--- | :--- | :--- |
| **AmazonMix-6** (Arts/Crafts/Sewing, Electronics, Home/Kitchen, Video Games, Movies/TV, Tools) | Pretraining corpus | This is the exact CSFT corpus the upstream paper uses to adapt `Qwen2-0.5B`. Substituting or narrowing it would silently change baseline fidelity for every downstream comparison, not just the intervention being tested. | Used identically across every version; never subsetted per experiment. |
| **Games** (`Video_Games_5core`) | Primary development dataset | Largest existing asset base (frozen SASRec/BERT4Rec checkpoints, ~9,517-item CLIP table, prior CSFT/IEM runs already audited), and the only candidate where a competitive-baseline gate was actually cleared: mean text-only NDCG@10 `0.0195` (BERT4Rec) exceeded the pre-registered floor `0.0182`. A dataset with a near-degenerate baseline (see GRU4Rec below) cannot support a causal claim, because any residual looks like an arbitrarily large percentage improvement over noise. | Outcomes examined repeatedly during design. Every report explicitly labels Games results as **development evidence, not confirmation**. |
| **Arts** (`Arts_Crafts_and_Sewing`) | In-domain replication | Second evaluation domain used to check a result isn't a Games-only artifact. It is honestly labeled **in-domain, not out-of-domain**: Arts is one of the six AmazonMix-6 pretraining domains, so the CSFT stage has already seen its item text. This transductive caveat is stated in every plan that uses Arts, not hidden. | Reviewed for prior exposure records without reading test metrics ahead of a run. |
| **Sports** (`Sports_5core`) | Out-of-domain transfer stress test | The one dataset actually held to a "does this generalize beyond the tuning domain" bar for `v9.2`. This is precisely where the frozen-residual family **failed**: real-vs-text `+0.000016` (fail), Recall@10 down past threshold on 2/3 seeds — which is why `v9.2` is reported as Games-conditional, not as a general method. | One transfer run per candidate method; failures are kept in the report, not re-run until they pass. |
| **Baby** (`Baby_5core`) | Sealed pre-registered confirmation | Reserved for a single one-shot confirmatory run against gates fixed *before* the run (text-baseline strength, replication direction, Recall-vs-NDCG dissociation, noise floor). Validation Half-B was spent once on an early heuristic screen; the test split is documented as **100% untouched** until that one run. This is the actual bias-control mechanism — not dataset difficulty. | Explicitly marked `never read here` in `v10`'s own frozen experiment config; owned by a separate reality-check plan, not by the two active generations. |

Backbone choice follows the same anti-cherry-picking logic on a second axis: `SASRec` is the primary backbone, `BERT4Rec` is a second, competitively-gated backbone used to confirm a result isn't SASRec-specific, and `GRU4Rec` is kept deliberately as a **stress test with a near-degenerate text baseline** (`0.00053`–`0.00070` NDCG@10) — its huge relative "gains" are reported and then explicitly discounted, rather than dropped from the record.

**Where `v10` and `v11` currently sit on this map:**
- `v10` (caption augmentation) follows the full discipline: Games for development, Arts for in-domain replication, AmazonMix-6 for pretraining everywhere, and Baby reserved untouched for a later, separate confirmation run.
- `v11` (HaNoRec CF-hardness) currently runs on **Games only**, because it depends on the frozen, already-audited LLM2Rec+SASRec checkpoint pair that only exists for that dataset. Extending it to Arts/Sports/Baby is explicit future work, not yet started — stated here rather than implied.

---

## SOTA Landscape (published results, same benchmark family)

To put `v0`'s reproduction gap and the closed/active generations above in context, this table reproduces the upstream paper's own baseline comparison — Table 3 of He et al. (2025), SASRec downstream, full-catalog `Recall@K`/`NDCG@K` — verbatim from `raw/rs-llm/…LLM2Rec…pdf` in the parent workspace, plus this fork's `v0` reproduction row for direct comparison. General-purpose text encoders (`BERT`, `GTE`, `BGE`, `LLM2Vec`) and recommendation-specific embedding models (`BLAIR`, `EasyRec`, `LLMEmb`) are included so the landscape isn't just "our baseline vs. our extensions."

**Games (in-domain)**

| Model | R@10 | N@10 | R@20 | N@20 | Source |
| :--- | ---: | ---: | ---: | ---: | :--- |
| BERT | 0.0585 | 0.0311 | 0.0863 | 0.0381 | Paper Table 3 |
| GTE | 0.0641 | 0.0349 | 0.0911 | 0.0418 | Paper Table 3 |
| BLAIR (recommendation-specific) | 0.0654 | 0.0361 | 0.0954 | 0.0437 | Paper Table 3 |
| EasyRec (recommendation-specific) | 0.0647 | 0.0357 | 0.0926 | 0.0428 | Paper Table 3 |
| BGE | 0.0733 | 0.0410 | 0.1022 | 0.0483 | Paper Table 3 |
| LLM2Vec | 0.0740 | 0.0407 | 0.1029 | 0.0480 | Paper Table 3 |
| LLMEmb (recommendation-specific) | 0.0813 | 0.0487 | 0.1085 | 0.0555 | Paper Table 3 |
| **LLM2Rec (paper, official, Qwen2-0.5B, full compute)** | **0.0865** | **0.0521** | **0.1157** | **0.0595** | Paper Table 3 |
| **LLM2Rec (this fork's `v0`, T4 compatibility reproduction, IEM ckpt-1000)** | 0.0821 | 0.0504 | 0.1091 | 0.0572 | `docs/reports/llm2rec-final-teacher-report.md` §3.1, cross-checked against the independent raw-rank audit |

**Sports (out-of-domain, excluded from CSFT pretraining)**

| Model | R@10 | N@10 | R@20 | N@20 | Source |
| :--- | ---: | ---: | ---: | ---: | :--- |
| GTE | 0.0823 | 0.0584 | 0.1001 | 0.0629 | Paper Table 3 |
| BLAIR (recommendation-specific) | 0.0893 | 0.0614 | 0.1091 | 0.0664 | Paper Table 3 |
| EasyRec (recommendation-specific) | 0.0887 | 0.0627 | 0.1061 | 0.0671 | Paper Table 3 |
| BERT | 0.0860 | 0.0649 | 0.1017 | 0.0689 | Paper Table 3 |
| BGE | 0.0974 | 0.0736 | 0.1141 | 0.0778 | Paper Table 3 |
| LLM2Vec | 0.1079 | 0.0854 | 0.1234 | 0.0893 | Paper Table 3 |
| LLMEmb (recommendation-specific) | 0.1131 | 0.0936 | 0.1257 | 0.0969 | Paper Table 3 |
| **LLM2Rec (paper, official)** | **0.1170** | **0.0976** | **0.1289** | **0.1006** | Paper Table 3 |

One caveat, stated rather than smoothed over: this landscape uses the paper's own **aggregate** full-catalog protocol (all targets, immediate-repeat included). The `v9.2` family's Games/Sports numbers in the Comprehensive Experimental Benchmark below use a **novel-target-only, repeat-debiased** protocol instead, discovered necessary precisely because aggregate metrics were found to hide an immediate-repeat confound (see `v9.1`). The two protocols are not directly comparable row-for-row — only `v0`'s reproduction row above uses the paper's own aggregate protocol, which is why it is the only fork row placed in this table.

---

## Comprehensive Experimental Benchmark

**Same dataset?** Mostly yes — every row below except one is **Games** (`Video_Games_5core`); one row is the **Sports** transfer test. What is *not* the same across rows is the **checkpoint** (pre-patch vs. patched-IEM — a mid-program bidirectional-attention bug fix), the **evaluation slice** (all test rows vs. novel-target-only, after `v9.1` showed aggregate metrics hide an immediate-repeat confound), and the **backbone** (SASRec/BERT4Rec/GRU4Rec). Those three columns are kept explicit in the table below instead of hidden, because a row that silently mixed checkpoints or slices would misattribute the patch's or the slicing's own effect to the visual intervention. One master table, not one baseline number.

| Generation | Dataset | Backbone | Checkpoint | Slice | Baseline NDCG@10 | Method NDCG@10 | Δ NDCG@10 | Baseline Recall@10 | Method Recall@10 | Δ Recall@10 | Verdict |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `v0` (reference) | Games | SASRec | pre-patch, ckpt-1000 | all rows | 0.0521 (paper) | 0.0504 | `-3.31%` | 0.0865 (paper) | 0.0821 | `-5.13%` | Reproduction, not an intervention |
| `v9.0` additive fusion | Games | SASRec | pre-patch (own table) | all rows | 0.04889 | 0.04016 | `-17.86%` | — | — | — | **Rejected** |
| `v9.2` I1‑A / patched rerun | Games | SASRec | patched-IEM | all rows | 0.04637 | 0.05741 | `+23.82%` | 0.07818 | 0.09060 | `+15.89%` | Fails matched-shuffle gate (shuffle alone: `+22.39%`) |
| `v9.2` joint-trained | Games | SASRec | patched-IEM | all rows | 0.04637 | 0.05107 | `+10.14%` | — | — | — | Confounded (shuffle alone: `+4.01%`) |
| `v9.2` G2 exposure-residual | Games | SASRec | pre-patch (own table) | novel-only | 0.02276 | 0.02413 | `+6.03%` | 0.04665 | 0.05074 | `+8.77%` | **Passes** vs. Gaussian/text-PCA/shuffle controls |
| `v9.2` BERT4Rec replication | Games | BERT4Rec | pre-patch (own table) | novel-only | 0.01953 | 0.02094 | `+7.21%` | +0.00357 (delta only) | — | — | **Passes**, replicates SASRec result |
| `v9.2` GRU4Rec stress test | Games | GRU4Rec | pre-patch (own table) | novel-only | 0.00065 | 0.01401 | `+2044%` | — | — | — | Discounted — text-PCA control scores higher (0.01665) |
| `v9.2` Sports transfer | Sports | BERT4Rec | pre-patch (own table) | novel-only, 3-seed mean | 0.00702 | 0.00711 | `+0.000016` abs (`fail`) | — | — | — | **Fails** — no transfer off Games |
| `v9.2` C0.5 matched-shuffle, SASRec | Games | SASRec | patched-IEM | novel-only, ATE | −0.00031 (shuffle ATE) | 0.00137 (real ATE) | `+0.00168`, CI `[0.00126, 0.00220]` | — | — | — | **`PASS_VISUAL_SPECIFICITY`** |
| `v9.2` C0.5 matched-shuffle, BERT4Rec | Games | BERT4Rec | patched-IEM | novel-only, ATE | −0.00010 (shuffle ATE) | 0.00141 (real ATE) | `+0.00151`, CI `[0.00086, 0.00214]` | — | — | — | **`PASS_VISUAL_SPECIFICITY`** |
| `v10` caption augmentation | Games, Arts | SASRec (matched) | — | — | — | — | — | — | — | — | Running, no result |
| `v11` HaNoRec CF-hardness | Games | Qwen2.5-VL reranker | — | — | — | — | — | — | — | — | Running, no result |

**Reading the table honestly:** the two `C0.5` rows are the only ones that isolate the visual signal from every other explanation (frozen checkpoint, matched-shuffle control, ATE with a 95% CI excluding zero) and are the closed program's actual headline result: a small, positive, Games-specific visual-specificity effect (`~+0.0015`–`+0.0017` NDCG@10), not the much larger `+23.82%`/`+2044%` deltas above them, which look impressive only because their comparator (plain text, or a near-degenerate GRU4Rec baseline) is a weaker check than a matched shuffle. `v9.0`'s rejection and `v9.2` Sports' transfer failure are the two rows that keep this program from being called a general method.

---

## Repository Structure

```plaintext
LLM2Rec-Research/
├── UPSTREAM_README.md                  # Original LLM2Rec README (He et al., KDD 2025)
├── run_LLM2Rec_CSFT.sh                 # Upstream Stage 1 (CSFT) launcher
├── run_LLM2Rec_IEM.sh                  # Upstream Stage 2 (MNTP + SimCSE) launcher
├── script_extract_and_evaluate.sh      # Upstream embedding extraction + downstream eval
├── llm2rec/                            # Upstream package: CSFT/MNTP/SimCSE configs and entry points
├── seqrec/                             # Upstream downstream sequential-recommender evaluators
├── baselines/                          # Upstream baseline recommenders
├── research/                           # This fork's active generations (v10, v11)
│   ├── caption_augmentation/           # v10 — arm construction, Florence-2/Qwen backends, corpus tests
│   │   ├── corpus.py                   # Arm construction, derangement, common-history budgeting
│   │   ├── caption.py                  # Captioner/Paraphraser/Tokenizer backends
│   │   ├── crosswalk.py                # Catalog ID <-> ASIN crosswalk
│   │   ├── package.py / smoke.py       # Kaggle packaging and end-to-end smoke checks
│   │   ├── experiment.json             # Frozen protocol: arms, seeds, caps, model pins, quality gates
│   │   └── test_*.py                   # stdlib-only regression suite
│   └── hanorec_cf_hardness/            # v11 — CF-margin/HaRS mixture, SFT+DPO reranking
│       ├── prep.py                     # Real pair construction from frozen SASRec candidates
│       ├── train.py                    # Qwen2.5-VL LoRA SFT + hardness-scaled DPO
│       ├── audit.py                    # Independent artifact/hash/metric audit
│       └── experiment.json             # Frozen protocol: pair counts, seeds, weights, artifact pins
└── docs/
    └── reports/                        # Source evidence for every version-generation claim above
        ├── llm2rec-final-teacher-report.md          # v0 - v9.2 pipeline, results, and closure rationale
        ├── 260920-llm2rec-two-methods-research.md   # v10 + v11 mechanism deep-dive
        └── two-llm2rec-methods-20260919-1517.md     # v10 + v11 current execution state and next steps
```

---

## Installation & Setup

### 1. Prerequisites (upstream)

- Python >= 3.9
- `torch >= 2.6.0`, `transformers >= 4.44.2`, `llm2vec == 0.2.3`, `flash-attn >= 2.7.4`
- A CUDA GPU. The compatibility profile used to produce the `v0` baseline numbers above targets a single Kaggle T4 (`sm_75`) and substitutes `SDPA` for FlashAttention‑2 and `FP16` for `BF16` — see `docs/reports/llm2rec-final-teacher-report.md` §2 for the full deviation list.

### 2. Environment

```bash
git clone <this-repo-url> LLM2Rec-Research
cd LLM2Rec-Research

conda create -n llm2rec_env python=3.9 -y
conda activate llm2rec_env

pip install torch>=2.6.0 transformers>=4.44.2 llm2vec==0.2.3
# flash-attn requires sm_80+; on T4/P100 use the SDPA compatibility path instead (see docs/reports).
```

### 3. Datasets

Upstream preprocessed datasets: [Google Drive link](https://drive.google.com/file/d/1GIXWaaaNuUkUtuFy5JTN0OwAQiLGb2z4/view?usp=sharing), unzipped under `./data`. `v10`/`v11` additionally require the Amazon Reviews'23 image manifests for their respective domains (Games, Arts) — see `research/caption_augmentation/experiment.json` and `research/hanorec_cf_hardness/experiment.json` for exact artifact hashes and pinned dataset slugs.

### 4. Reproduce the upstream baseline (v0)

```bash
bash run_LLM2Rec_CSFT.sh
bash run_LLM2Rec_IEM.sh
bash script_extract_and_evaluate.sh
```

### 5. Run the active generations' local test suites

```bash
python -m py_compile research/caption_augmentation/*.py
python -m unittest discover -s research/caption_augmentation -p "test_*.py"

python -m py_compile research/hanorec_cf_hardness/*.py
```

Neither package touches a network, GPU, or real model weight outside its designated Kaggle kernel packaging step — local runs only validate arm construction, pairing, and serialization logic, never caption/reranking quality.

---

## Citation & License

This is an unofficial research fork studying LLM2Rec, not the paper's implementation. The upstream repository has **no LICENSE file**, so all upstream code in this fork is used and copied under that constraint (source-available, redistribution rights not established); this fork's own `research/` additions are the authors' own code, not upstream-derived.

Cite the upstream paper if you build on the baseline pipeline:

```bibtex
@inproceedings{he2025llm2rec,
  title={LLM2Rec: Large Language Models Are Powerful Embedding Models for Sequential Recommendation},
  author={He, Yingzhi and Liu, Xiaohao and Zhang, An and Ma, Yunshan and Chua, Tat-Seng},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2},
  pages={896--907},
  year={2025}
}
```

`v11` additionally builds on HaNoRec ([`wangyu0627/HaNoRec`](https://github.com/wangyu0627/HaNoRec), pinned commit `587face7`).

---

## Unresolved Questions

- `v10`: does a correctly-linked image caption improve recommendation beyond `title-only`, `null`, `shuffle`, and `paraphrase` controls on **both** Games and Arts, at the pre-registered family-wise significance level? Not yet answered — corpus generation is incomplete.
- `v11`: does CF-margin-weighted DPO (`w ∈ {0.5, 1.0}`) beat pure semantic hardness (`w = 0`) on reranking SASRec's real top-20? Not yet answered — full six-arm run is unverified.
- Neither `v10` nor `v11` has been tested on Sports or the sealed Baby split; whether either replicates the `v9.2` failure-to-transfer pattern is open.
- Whether `v10`'s text-mediated visual signal and `v11`'s CF-hardness reranking are complementary (stackable) or redundant has not been investigated; they currently run as fully independent generations.

*Version lineage compiled 2026-09-21 from `docs/reports/` in this repository, itself sourced from the parent research workspace's frozen experiment configs and independently audited run reports.*
