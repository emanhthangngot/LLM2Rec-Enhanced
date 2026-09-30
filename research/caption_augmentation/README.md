# LLM2Rec caption-augmentation experiment package

Implementation package for
`plans/260928-0043-qwen3vl-v10-caption-retraining/plan.md`. It defines the
Qwen3-VL caption replacement and registers the full LLM2Rec matrix;
the historical Florence v10 outputs remain separate evidence.

## Contract

- `corpus.py` — arm construction, frequency-bin donor derangement, token-cap
  budgeting, common-history-suffix selection, and atomic corpus serialization.
  No numpy/torch/transformers import anywhere in this file.
- `caption.py` — `Captioner`/`Paraphraser`/`Tokenizer` protocols with a real
  Qwen3-VL-4B structured image-plus-title backend, Qwen2.5-3B title-only
  paraphraser, and Qwen tokenizer (GPU-only, lazy imports), plus CPU-only
  stubs for deterministic local checks.
- `experiment.json` — the Qwen prompt/model protocol, arms, seeds, caps,
  quality gates, and archive pin. The runtime-resolved Qwen revision remains
  `null` here until the Kaggle smoke records the immutable model/processor SHA.
- `test_corpus.py` — stdlib-only `unittest` regression suite. Run:

  ```bash
  cd research
  python -m py_compile caption_augmentation/*.py caption_augmentation/kaggle/*.py
  python -m unittest discover -s caption_augmentation -p "test_*.py"
  ```

## Executed Kaggle kernels (`kaggle/`)

These are retained historical outputs from the previous Florence v10 corpus
and earlier `real`-arm experiments. The Qwen3-VL package is generated in
`kaggle/qwen3vl_full_corpus/`; it must not resume from these Florence outputs.

| Stage | Script | Kaggle kernel | Output |
|---|---|---|---|
| Historical caption corpus (Florence-2 + Qwen2.5-3B paraphrase) | `full_corpus_generation.ipynb` | `trixuanle/llm2rec-caption-full-corpus-v2-inline` | 108,753 records, 213 shards |
| Tiny-chain input preflight (no training) | `tiny_chain_preflight.py` | see metadata | `tiny-chain-input-preflight.json` |
| Tiny-chain CSFT smoke (execution proof only) | `tiny_chain_csft.py` + `run-protocol-tiny-chain-csft.json` | see metadata | no recommendation metrics |
| CSFT, `real` arm, 1,000 steps | `csft_caption.py` | `trixuanle/llm2rec-caption-full-csft` | `caption_csft_artifact.json` |
| IEM (MNTP + SimCSE, patched bidirectional Qwen2) | `iem_caption.py` | `trixuanle/llm2rec-caption-full-iem` | checkpoints 500/1000 |
| Embedding extraction + SASRec (3 SASRec seeds) | `evaluate_caption.py` | `trixuanle/llm2rec-caption-full-evaluation` | `games_evaluation_artifact.json`, `results.txt` |

`results/v8/` and `results/v9/` hold the downloaded artifacts of the two
`real`-arm runs. v9 includes the history-token-budget fix, which changed
1 history item. Both runs use a single seed-42 chain, so they are not a
multi-chain estimate. Each `iem/` subfolder holds the MNTP/SimCSE configs and
`trainer_state.json` loss/grad-norm histories for that run, retrieved per
version with `ListKernelSessionOutput.version_label = "v1"`/`"v2"`. The
CLI form `owner/slug/N` silently returns the latest version.

`results/caption_manual_check/sample100_labels.json` holds the fixed 100-item
sample (seed 20260924) with Florence-2 captions and manual labels.
`colab/qwen3vl_caption_pilot.ipynb` re-captions the same 100 items with
Qwen3-VL. The items are embedded in the notebook and the images are downloaded
at run time. It runs three conditions: `structured`, `generic` and
`title_only`.

See `docs/reports/v10-caption-augmentation.md` for the result tables and
`docs/reports/v10-caption-augmentation-design.md` for the design, corpus
audit, manual check and captioner plan. The Video_Games paired pilot is
reported in `docs/reports/v10-video-games-paired-pilot.md`.

## Video_Games paired pilot (Qwen3-VL `real` vs `title-only`)

Plan: `plans/260929-0014-video-games-qwen-paired-pilot/`. This is a
single-domain exploratory pilot. It is not evidence for the AmazonMix-6
protocol.

**Design.** Both arms run the same shortened LLM2Rec chain on Video_Games
only: CSFT (1,000 steps, about one epoch over 122,577 training rows) →
MNTP → SimCSE (both stop at step 1,000) → mean-pooled extraction → SASRec
with seeds 2024/2025/2026. They share the chain seed (2024), the splits and
the hyperparameters. The only difference is the item text:

| Arm | Item text | Where the text is used |
|---|---|---|
| `title-only` | original title | CSFT history, IEM corpus, extraction |
| `real` | `Title: <title>; Visual cues: <Qwen3-VL cue, ≤32 tokens>` | CSFT history, IEM corpus, extraction |

The CSFT target is always the original title, and the loss covers only the
target tokens. Items whose caption is `mismatch`, missing, or whose image
failed (619 of 9,517) keep the cue `unavailable`; they are not dropped. The
user chose one LLM chain per arm, because six full chains (37.7 GPU-h)
exceeded the 30 GPU-h cap.

| Stage | Code | Kaggle kernel | Output |
|---|---|---|---|
| Review gate + dataset contract + paired-arm records | `video_games_pilot.py` | — (local) | `results/video_games_pilot/video_games_dataset_contract.json`, `results/qwen3vl_pilot/assistant_visual_audit*.{csv,json}` |
| Caption smoke (64 images) | `kaggle/video_games_l4_smoke/` | `trixuanle/llm2rec-qwen3vl-video-games-l4-smoke-v1` | `results/video_games_pilot/l4_smoke/` |
| Caption corpus (9,517 items, 19 shards) | `kaggle/video_games_corpus/` | `trixuanle/llm2rec-qwen3vl-video-games-corpus-v1` (v2 COMPLETE) | `results/video_games_pilot/corpus_v1/` |
| Paired arm chain + SASRec | template `kaggle/video_games_paired_arm.py` → `kaggle/video_games_paired_{title-only,real}/runner.py` | `trixuanle/llm2rec-vg-paired-title-only-v1` (v3 PASS), `trixuanle/llm2rec-vg-paired-real-v1` (v4 PASS) | `results/video_games_pilot/{title-only_arm_v3,real_arm_v3,real_arm_v4}/` |
| Comparison + setup audit | — (local) | — | `results/video_games_pilot/paired_comparison.json`, `setup_audit.json` |

To generate a runner, replace `ARM = "__ARM__"` in the template. For a resume
push, also set `RESUME_REQUIRED = True` and list the kernel itself in
`kernel_sources`. The runner then fails fast instead of silently retraining
CSFT. Kaggle does not mount the output of a cancelled version.

**Result (test split, mean ± sd over 3 SASRec seeds):**

| Metric | title-only | real | Δ |
|---|---|---|---|
| Recall@10 | 0.0838 ± 0.0008 | 0.0783 ± 0.0047 | −6.5% |
| Recall@20 | 0.1102 ± 0.0008 | 0.1028 ± 0.0048 | −6.7% |
| NDCG@10 | 0.0499 ± 0.0021 | 0.0482 ± 0.0044 | −3.6% |
| NDCG@20 | 0.0566 ± 0.0020 | 0.0543 ± 0.0045 | −4.0% |

Verdict: `NO_GAIN_FROM_CAPTIONS`. The setup audit (`setup_audit.json`) found
no implementation cause. Captions reach every stage and pull the embedding
space toward visual attributes: next-item AUC falls from 0.620 to 0.588,
including on pairs where both items have real cues. Caveats: there is one
LLM chain per arm, so the spread reflects SASRec seeds only. The runners
requested `NvidiaL4`, but the artifacts record `Tesla T4`. Total spend was
30.77 GPU-h against the 30 h cap; the user approved the overrun.

Large regenerable outputs are not committed: `arm-inputs/*.csv` (up to
147 MB) and the item-embedding `.npy` files. The artifact JSONs record the
embedding SHA-256.

The Python package directory was renamed `caption_augmentation` in this
repository. The Kaggle packaging code (`package.py`) and the committed
notebook still use the import name `visual_delta_fusion`, which is the name
they were executed under.

## Naming

In the original workspace the directory was `visual_delta_fusion`
(underscore, importable), matching its sibling `baseline/` and `multimodal/` packages.
That name predates the plan's "no lexical delta filter" decision. The
current arms do not compute a lexical "delta".

## Kaggle-only boundary

Importing modules and running local tests does not load real weights, access
the network, or require a GPU. The generated Qwen3-VL caption kernel is
GPU-only; its Transformers runtime is isolated from downstream LLM2Rec
training.

## Current execution boundary

The local preflight validates all 10 registered profiles across chain seeds
2024/2025/2026. The full AmazonMix-6 matrix is still not executed. The only
Qwen3-VL training run completed so far is the Video_Games paired pilot
above.
