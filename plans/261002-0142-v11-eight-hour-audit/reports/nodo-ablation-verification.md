# Matched NoDO ablation: pre-push evidence

State: `trlxun/hanorec-v11-nodo-ablation` v1 accepted through Kaggle MCP; GPU completion and scientific results have not been observed. The scientific owner is `nodo-ablation-protocol.json`; submission hashes and conservative reservations are in `gpu-budget.json`.

## Exercised verification

- Existing core regression suite: 31 passed, zero skipped. No frozen core file was changed for this ablation.
- Generated notebook: all nine code cells compiled; all ten embedded source hashes matched the registered eligible parent and local sources. All 1924 distinct image dependencies for the actual training/scoring paths were present in the completed-parent output inventory before submission.
- Numerical NoDO check: sigma0 forward outputs and parameter gradients matched the no-hook path in fp32 and bf16; sigma0.05 differed. The matched private-generator check preserved the global RNG and equalized draw advancement across sigmas.
- Actual-cell CPU smoke: pinned processor plus a tiny real Qwen model, synthetic PIL catalog, two distinct training pairs and two validation events. Created a portable SFT parent, then loaded two fresh DPO models and executed the generated parent-load, registration, per-arm run and summary cells. Both arms completed the fixture's exact one update, passed separate artifact audits, retained identical candidates, emitted empty test predictions and passed reference/reload checks. Execution configs, sigma attribution, distinct checkpoint directories, content hashes and receipts were checked. The summary completed with one primary NDCG contrast and exploratory Recall separate.
- Deadline-controller smoke: a started arm reaching its allocated deadline recorded `INCOMPLETE_BUDGET` and was excluded from usable results; unrelated runtime failures propagated; inadequate remaining allowance skipped before creating the arm directory.
- Independent code review: final result PASS with no blocking findings after deadline allocation, primary-contrast separation and artifact attribution corrections.

Executed scratch harnesses were preserved under the vault-root path `data/kaggle/v11-audit-20261002/verification_sources/` as `nodo_actual_cells_smoke.py` and `nodo_budget_smoke.py`; their `/tmp/` copies were removed. CPU environment: `/tmp/v11-ablation-venv` with torch2.10.0+cpu, transformers4.51.3 and peft0.15.2.

## Limits and result gate

The actual-cell smoke used synthetic parent/probe manifests and an explicitly synthetic passing parent gate. It proves orchestration, not eligibility of a scientific parent or recommendation quality. Real parent eligibility was established separately by the completed SFT audit. CPU/tiny-model timing does not establish full-size T4 duration, precision fidelity or GPU memory safety. Deadline fixtures exercise controller behavior, not a promise that both real arms will finish.

The submitted study must satisfy the real protocol's exact update/micro-batch counts, parent/source integrity, per-arm audits, sigma receipts and paired cohort checks. Partial or failed training cannot enter the primary comparison. After the user's completion notice, download and verify artifacts and independently recompute the paired contrast before interpreting noise's effect on learned clean ranking. Single-seed, conditional fitted-model inference and the sparse informative cohort remain scientific limits. No automatic polling or CF-matrix expansion is authorized by this submission.
