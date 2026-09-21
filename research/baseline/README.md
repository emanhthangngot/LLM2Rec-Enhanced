# LLM2Rec compatibility baseline

This directory is the canonical research baseline for the verified text-only
LLM2Rec reconstruction. Future variants should be added as sibling directories
under `code/llm2rec/`; this baseline remains frozen unless a new reproduction
run explicitly supersedes it.

## What is reproduced

The pipeline preserves the released stage order:

1. CSFT on the full AmazonMix-6 5-core corpus.
2. IEM MNTP followed by SimCSE.
3. Item-title embedding extraction for Games_5core.
4. SASRec downstream evaluation at IEM steps 500 and 1000.
5. Functional bidirectional and raw-rank audit.

The exact values are machine-readable in
`config/compatibility-profile.json`. The executable Kaggle stage scripts are
under `kaggle/`, with one metadata file per kernel because Kaggle quota and
runtime boundaries require separate jobs.

## Reproduction profile

This is a successful `compatibility-profile`, not a strict official run:

- one Kaggle NVIDIA T4, one process;
- float32 parameters with float16 autocast;
- SDPA instead of FlashAttention-2;
- CSFT capped at 1,000 steps to fit the approved GPU budget;
- SimCSE batch 32 instead of the official 256;
- IEM checkpoints 500 and 1,000;
- SASRec seeds 2024, 2025, and 2026 with full-catalog scoring.

The T4 profile also applies a guarded Qwen2 padding-only SDPA mask. The audit
requires `Qwen2BiModel`, `is_causal=false`, and measurable future-token
sensitivity before accepting the run.

## Execution order

Push the four `kernel/` scripts as separate Kaggle kernels in this order:

```text
kaggle/csft_chain.py
  -> kaggle/iem_pipeline.py
  -> kaggle/evaluate_games.py
  -> kaggle/audit_games.py
```

Use the corresponding `kernel-metadata-*.json` file as the metadata contract.
The IEM kernel consumes CSFT `checkpoint-1000`; evaluation consumes IEM
`checkpoint-500` and `checkpoint-1000`; audit consumes evaluation outputs,
IEM checkpoints, and Games_5core.

## Local validation

From the repository root:

```bash
python code/llm2rec/baseline/validate_baseline.py
python -m py_compile code/llm2rec/baseline/kaggle/*.py
```

The verified evidence is retained by path in the configuration and provenance
files. No HF token is required for the public Qwen2-0.5B model; Kaggle
credentials stay outside the repository.
