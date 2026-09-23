# LLM2Rec caption-augmentation experiment package

Phase 1 of `plans/260915-0955-visual-delta-fusion-pilot/plan.md`. This package
builds and tests, locally and standard-library-only, the deterministic parts
of a controlled experiment testing whether image-caption text improves
LLM2Rec's CSFT-then-embedding pipeline beyond title text, formatting, and
title-only rewriting.

## Contract

- `corpus.py` — arm construction, frequency-bin donor derangement, token-cap
  budgeting, common-history-suffix selection, and atomic corpus serialization.
  No numpy/torch/transformers import anywhere in this file.
- `caption.py` — `Captioner`/`Paraphraser`/`Tokenizer` protocols with a real
  Florence-2-large / Qwen2.5-0.5B-Instruct / Qwen2 backend (GPU-only, lazy
  imports) and a stub backend (used only by `test_corpus.py` and by future
  Phase 2 wiring tests, never shipped as the production captioner).
- `experiment.json` — the versioned, frozen protocol: arms, seeds, caps,
  model IDs, quality gates, and the archive pin. Resolved model revisions stay
  `null` here until Phase 2's real-runtime probe fills them from a live
  `huggingface_hub.model_info` call — never guessed.
- `test_corpus.py` — stdlib-only `unittest` regression suite. Run:

  ```bash
  cd research
  python -m py_compile caption_augmentation/*.py caption_augmentation/kaggle/*.py
  python -m unittest discover -s caption_augmentation -p "test_*.py"
  ```

## Executed Kaggle kernels (`kaggle/`)

These are the exact kernels that produced the v10 corpus and the v8/v9 `real`
arm results. Each `kernel-metadata-*.json` points `code_file` at its script.

| Stage | Script | Kaggle kernel | Output |
|---|---|---|---|
| Caption corpus (Florence-2 + Qwen2.5-3B paraphrase), resumable over 10 pushes | `full_corpus_generation.ipynb` | `trixuanle/llm2rec-caption-full-corpus-v2-inline` | 108,753 records, 213 shards |
| Tiny-chain input preflight (no training) | `tiny_chain_preflight.py` | see metadata | `tiny-chain-input-preflight.json` |
| Tiny-chain CSFT smoke (execution proof only) | `tiny_chain_csft.py` + `run-protocol-tiny-chain-csft.json` | see metadata | no recommendation metrics |
| CSFT, `real` arm, 1,000 steps | `csft_caption.py` | `trixuanle/llm2rec-caption-full-csft` | `caption_csft_artifact.json` |
| IEM (MNTP + SimCSE, patched bidirectional Qwen2) | `iem_caption.py` | `trixuanle/llm2rec-caption-full-iem` | checkpoints 500/1000 |
| Embedding extraction + SASRec (3 SASRec seeds) | `evaluate_caption.py` | `trixuanle/llm2rec-caption-full-evaluation` | `games_evaluation_artifact.json`, `results.txt` |

`results/v8/` and `results/v9/` hold the downloaded artifacts of the two
`real`-arm runs. v9 includes the history-token-budget fix, which changed
1 history item. Both runs use a single seed-42 chain, so they are not a
multi-chain estimate. See `docs/reports/v10-caption-augmentation.md`
for the result tables and `docs/reports/v10-caption-augmentation-design.md` for
the design and corpus audit.

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

Nothing in this package touches the network, a GPU, or a real model weight.
`Florence2Captioner`, `QwenParaphraser`, and `QwenTokenizer` in `caption.py`
import their backends lazily and are only ever constructed on Kaggle, inside
the Phase 2 kernel package. Phase 1's own acceptance is local-only: import
cleanliness, arm/derangement/budgeting correctness, and serialization
integrity — not caption quality or model behavior.

## What Phase 1 is not

This package does not download the LLM2Rec archive, fetch item images, run
Florence-2, run the CSFT/IEM/extraction pipeline, or touch Kaggle. Those are
Phase 2 (`plans/260915-0955-visual-delta-fusion-pilot/phase-02-kaggle-package.md`)
and Phase 3. Phase 1's `build_catalog_rows` has been exercised against the
real, committed
`experiments/multimodal_llm_rs/kaggle/llm2rec-visual/item_asin_map.json`
(9,517 Games items) as a smoke check, but the archive download, image fetch,
and captioning itself are still unexecuted.
