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
  python -m py_compile code/llm2rec/visual_delta_fusion/*.py
  python -m unittest discover -s code/llm2rec/visual_delta_fusion -p "test_*.py"
  ```

## Naming

The directory is `visual_delta_fusion` (underscore, importable), matching the
sibling `code/llm2rec/baseline/` and `code/llm2rec/multimodal/` convention.
This name predates the current plan's "no lexical delta filter" decision and
is kept for import/path stability, not because the current arms compute a
lexical "delta" — they do not.

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
