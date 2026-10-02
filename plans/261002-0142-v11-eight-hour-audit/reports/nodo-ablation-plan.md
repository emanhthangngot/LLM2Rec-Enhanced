# Matched NoDO training ablation

User approved continuation after the completed clean/noisy probe. The probe shows strong transient log-odds attenuation but does not establish training harm. Execute the discriminating contrast before another CF matrix.

Outcome: compare clean ranking after matched semantic-only DPO with sigma0.05 vs sigma0.0, with frozen SFT-only as the supporting baseline.
Constraints: current parent/ten core sources immutable; same true-user-corrected pairs, event cohort, candidates, seed2024, w1.0, one epoch, batch4/accum8, AdamW/cosine/clipping/beta; test scoring disabled. Prospective protocol is `nodo-ablation-protocol.json`. New 9000-second job fits the remaining16200-second follow-up reserve, leaving7200 seconds conservatively unreserved. All failed GPU runs count.
Non-goals: changing CF formula or encoder, calling sigma0 a replacement HaNoRec baseline, claiming ranking harm from a no-training probe, choosing by test, or a universal v11 verdict from one seed.

## Changes and reviewed touchpoints

- New `research/hanorec_cf_hardness/scripts/build_nodo_ablation.py`: compose the existing notebook cell factories and `PARENT_LOAD` from `scripts/build_hanorec_notebook.py:283-303`; generate standalone `kaggle/hanorec-v11-nodo-ablation/`.
- Reuse `train.py:1285-1390` unchanged. `noise_sigma` is already read at `train.py:755`; sigma0 makes the same NoDO hook algebraically zero, preserving the training RNG schedule. Weight/condition/control remain1.0/real/None.
- Core artifact audit keys do not distinguish sigma. Audit each arm independently with its single declared matrix, then additionally verify two distinct arm IDs/sigma receipts, source/parent/config hashes and identical events/candidates at experiment level. Do not mislabel two sigma variants as distinct CF controls.
- Supporting clean evaluation uses `analysis.py` unchanged: full-event cohort and true-user clusters. One primary off-minus-on NDCG contrast; SFT comparisons and Recall exploratory. Record meaningful threshold0.002 and null/CI uncertainty.
- Save exact per-arm execution configs before training and content-address result/checkpoint receipts afterwards; no modifications to historical/core outputs.

## Acceptance and verification

1. Validate the registered protocol, hashes, distinct arms and scientific-config equality except sigma; refuse duplicate or missing arms.
2. AST-compile every notebook code cell and verify all ten embedded sources match the eligible parent.
3. New path smoke with tiny real Qwen and pinned processor: two fresh models, complete updates, sigma0 vs0.05, correct receipts, same candidates, clean before/after probes and frozen reference. Explicitly label CPU/synthetic boundaries.
4. Independent code review and testing before the one MCP push (accounta/trlxun, T4, timeout9000s).
5. Kernel asserts exact34 updates and265 micro-batches per real arm; empty test predictions; parent/reload/source integrity. Only both finished+audited arms may label the experiment COMPLETE.
6. No polling. After completion notice download receipts/per-event predictions/checkpoints, verify configs and recompute the paired contrast before any causal claim or follow-up.

Runtime limits remain a risk, not a no-crash guarantee. Preserve scientific completeness and report failure/partial execution honestly rather than accepting shortened training.
