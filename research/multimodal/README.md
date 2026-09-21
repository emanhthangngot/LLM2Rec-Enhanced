# LLM2Rec visual-signal branch

This folder is the isolated visual extension of the frozen text-only package in
`code/llm2rec/baseline/`. The first experiment adds a **late visual branch at
the downstream item-embedding adapter**, not inside CSFT or IEM.

## Why this insertion point

The released downstream path freezes the LLM2Rec title embedding table, maps it
through a trainable MLP adapter, and scores the resulting catalog embeddings in
SASRec. Adding a frozen CLIP item vector at this boundary:

- preserves the released LLM/CSFT/IEM stages;
- keeps full-catalog scoring and the metric implementation unchanged;
- isolates whether image evidence adds item-discriminative signal; and
- permits matched `T-null`, `T-mask`, `L-shuffle`, and `L-real` controls.

## Tensor contract

| Tensor | Shape | State |
|---|---:|---|
| title embedding | `[N, 896]` | frozen LLM2Rec output |
| CLIP image embedding | `[N, 512]` | frozen, L2-normalized |
| availability mask | `[N, 1]` | `1` only for a decoded image |
| visual branch input | `[N, 513]` | image vector concatenated with mask |
| visual branch hidden | `[N, 896]` | trainable projection |
| visual branch output | `[N, hidden]` | added to text adapter output |

Missing images remain catalog rows with a zero vector and mask `0`. They are
never deleted silently.

## Arms

- `T-null`: zero visual vector and zero mask.
- `T-mask`: zero visual vector and the real availability mask.
- `L-shuffle`: deterministic within-frequency-bin derangement over available
  rows; missing-image rows stay fixed with zero vectors and mask `0`.
- `L-real`: real frozen CLIP vectors and real mask.

All arms must share model initialization, optimizer, seeds, budget, candidate
set, and full-catalog evaluator. `L-shuffle` is invalid for singleton available
bins; the helper fails closed instead of creating fixed points.

## Run boundary

`config/visual-compatibility-profile.json` records the contract. Before any
GPU run, create an immutable preflight manifest containing the resolved 40
character CLIP revision, source commit, catalog/test counts, image coverage,
decode failures, ID consistency, `image_manifest_sha256`, and
`visual_features_sha256`. `preflight.py` rejects a missing revision, invalid
hash, ID conflict, or coverage below 80%.

The current implementation is a real module and contract/preflight gate. It does
not claim a positive visual result and does not start training before image
coverage and provenance are available.

## Kaggle execution bundle

The executable four-arm screen is packaged at
`experiments/multimodal_llm_rs/kaggle/llm2rec-visual/`. Its kernel:

1. verifies the pinned LLM2Rec source and checkpoint-500 IEM input;
2. joins the 9,517 downstream IDs to parent ASINs;
3. downloads Amazon Reviews'23 Video Games metadata and decodes images;
4. extracts immutable-revision CLIP ViT-B/32 features;
5. trains `T-null`, `T-mask`, `L-shuffle`, and `L-real` for seeds 2024–2026;
6. writes the preflight manifest, feature hashes, aggregate metrics, and
   per-user top-20 predictions under
   `llm2rec_visual_signal_results/`.

The Kaggle kernel is a labelled compatibility diagnostic, not an official
multimodal LLM2Rec reproduction. A coverage or provenance failure aborts before
training.

## Approved score-level iteration

`experiments/multimodal_llm_rs/kaggle/llm2rec-visual-score/` preserves the
text item table and SASRec state, then adds a centered PCA-64 visual score at
the full-catalog logit boundary. A single scalar `alpha` is initialized to
exactly zero, so the first forward pass must be bitwise equal to the text-only
logits. The matched arms are `S-null`, `S-shuffle`, and `S-real`; PCA uses
frozen v9 CLIP features without interaction labels.
## Current project status

- The v9 late visual screen completed on Kaggle. Coverage, provenance, feature
  integrity, and shuffle controls passed.
- The history-only fusion iteration completed 12/12 arm-seed jobs on Kaggle in
  6,192.32 seconds. Exact null parity and projection liveness passed.
- Its scientific result failed both predeclared gates: `h-real` versus
  `h-shuffle-M1` had hierarchical NDCG@10 delta `-0.0002134` with CI
  `[-0.0044593, +0.0030441]`; versus `text-exact`, delta was `-0.0029337`
  with CI `[-0.0075312, +0.0003128]`.
- Inference overhead passed at `1.0320x`, but Recall@10 harm versus
  `text-exact` exceeded the 1% safety budget. No further history-fusion
  iteration or visual baseline promotion is authorized.

## Linux and Windows handoff

The repository is portable, but Kaggle kernels are the execution authority for
GPU experiments. Keep local paths relative to the repository and use POSIX
paths (`/`) in Python strings; `pathlib.Path` resolves them on both Linux and
Windows. Never commit `/home/...`, `C:\...`, `/kaggle/...`, credentials, or
machine-specific caches.

Before running locally on Windows:

1. create and activate a Python virtual environment;
2. install the pinned dependencies used by the target module;
3. run `python -m py_compile <script>.py`;
4. run the focused `python -m unittest ...` command from the repository root;
5. remember that a local environment without NumPy/PyTorch can only verify
   pure-stdlib contracts; skipped tensor tests are not GPU verification.

Before submitting to Kaggle from either OS, inspect the generated
`kernel-metadata.json`, verify its `id` matches the actual Kaggle slug, and
read the complete `kernel_push` stdout for dropped tags or kernel sources.
Use `kaggle_kernel_status` until `COMPLETE` or `ERROR`; on MCP output timeout,
retry the same `kaggle_kernel_output` request. Locate mounted files by
filename under `/kaggle/input`, not by slug directory.

Kaggle-specific failure modes and recovery rules are recorded in the omp
memory rule `rule://kaggle-mcp-experiments`.
