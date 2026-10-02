---
title: "v11 repaired gate: SFT parent, DPO arms, paired statistics on Kaggle T4"
status: in-progress
created: 2026-10-01
---

# v11 repaired gate

**Outcome:** a valid answer to "does any Qwen2.5-VL reranker beat the SASRec order on informative validation users, and do `w` / CF hardness matter beyond constant hardness?" using a setup that matches HaNoRec/LLaMA-Factory.
**Constraints:** pinned inputs and frozen `lambda_cf` formula; one seed (2024); validation only, test never scored; Kaggle account `trlxun`; Kaggle's API serves only T4/P100 so bf16 is emulated (recorded as an adaptation); MCP push timeout cap 6 h so work is split in two jobs; `max_pixels` and epochs are derived from measured T4 speed/VRAM.
**Non-goals:** 3-seed confirmation, real/shuffle matrix, changing the CF formula, Sports/Baby transfer.
**Acceptance:** fail-closed checks pass in-kernel (`init_parity_max_abs_logit` <= 0.25, exact optimizer-update counts, SFT adapter unchanged, language-only LoRA, every catalog image processable); `job_manifest.json` / `arms_summary.json` carry source hashes and adaptations; verdicts come from paired bootstrap vs SASRec, SFT-only and w=1.0.

## Phases

| Phase | Status | Evidence |
|---|---|---|
| 1. Audit vs pinned upstream | done | `docs/reports/v11-hanorec-cf-hardness.md` 2026-10-01 section |
| 2. Repair `prep.py` / `train.py`, add `analysis.py` + tests | done | 25 CPU tests; tiny-Qwen2.5-VL CPU smoke (loss 0.693 at start, parity 0.0); independent code review findings H1-H3, M1-M8 addressed or documented |
| 3. Notebook generator and dry run of every cell | done | `scripts/build_hanorec_notebook.py`; both notebooks executed end to end on CPU |
| 4. `hanorec-v11-repaired-sft` on Kaggle T4 | v1 stopped at the GPU assert (served T4); v2 pushed, RUNNING | `trlxun/hanorec-v11-repaired-sft` |
| 5. Read SFT gate and per-example timing; choose arm list; push `hanorec-v11-repaired-arms` | pending | gate: `reference_learned_binary_task`, SFT vs SASRec verdict |
| 6. Audit arm outputs, update reports | pending | |

## Decision rule for phase 5

Arm job only starts if the SFT job is `COMPLETE`. If `reference_learned_binary_task` is false, fix SFT before spending DPO hours. Arm list defaults to `w1.0 w0.0 w0.5 w0.0:mean`; trim by measured seconds per example so it fits the 6 h cap (`SESSION_LIMIT_S` skips what cannot finish).
