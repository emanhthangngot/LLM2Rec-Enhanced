# v11 — HaNoRec CF/history-aware hardness: kết quả và trạng thái

## Tóm tắt trạng thái

- **Lần chạy 265 user (implementation đầu tiên): rút lại (withdrawn).** Real thắng shuffle về point estimate ở mọi `w`, nhưng paired bootstrap có CI 95% chứa 0 ở mọi ô. Sau đó run bị rút lại vì lộ item held-out, lấy negative từ lịch sử train, shuffle không matched, reference sai và chỉ chạy 2 update trên toàn dataset (xem phần 4 của `v11-hanorec-cf-hardness-design.md`).
- **Exploratory 1-seed:** historical `NO_SIGNAL_ABOVE_RETRIEVER`; setup-invalid for mechanism inference (see the 2026-10-01 audit).
- **Ma trận 3 seed đã đăng ký:** dự phóng 87.6–114.3 giờ GPU T4 so với trần 8 giờ, nên chưa chạy (`BLOCKED_NO_FIT`).
- Artifact nằm trong `research/hanorec_cf_hardness/results/`. Các phần gốc tiếng Anh bên dưới được giữ nguyên văn.

## 2026-10-02 — true-user audit and corrected-label diagnostic

The later repaired SFT v2 and four DPO arms v1 completed. SFT-only NDCG@10 was 0.064941 versus SASRec 0.052777; CF-only DPO was 0.064836. All paired intervals crossed zero. Their mechanism acceptance is now withdrawn: the 530 training events included 27 hard negatives and 1 SFT negative that were known TRAIN interactions of the same true user. The 265 validation events belong to 258 users, not 265 independent users.

Current preparation pins source CSVs, verifies the +1 downstream item-ID mapping, excludes all same-user TRAIN interactions, and carries pseudonymous user clusters. Current inference uses whole-user bootstrap and null sign flips; meaningful gain requires a lower CI bound >=+0.002 and a rejected null. Historical artifacts/source hashes remain immutable.

Verification: 31 CPU regressions pass with zero skips; real SASRec/data preparation smoke has zero seen-negative violations and unchanged baseline/held-out-invariant training hash. Pinned HaRS/DPO/NoDO math and gradients pass CPU parity; real optimizer/scheduler checkpoints have 34/34 updates. Fresh tiny-Qwen SFT/semantic/mean-control and no-training probe smoke pass; these are engineering checks, not new research results.

A fresh SFT diagnostic was submitted through Kaggle MCP as `trlxun/hanorec-v11-user-corrected-sft`, version 1, T4, hard timeout 10800 seconds. No result yet. It reuses only hash-verified image bytes, never the historical adapter. The no-training probe and matched follow-up remain dependency-gated. Total additional budget is 8 GPU-session hours including failures (caps 3h/0.5h/4.5h).

Details: `plans/261002-0142-v11-eight-hour-audit/reports/setup-audit-20261002.md`; budget ledger `plans/261002-0142-v11-eight-hour-audit/reports/gpu-budget.json`. One seed and 41 informative events remain a serious precision limitation: this is an investment decision study, not proof of universal ineffectiveness or a native HaNoRec reproduction.


## Cập nhật 2026-10-01 — audit setup và bản sửa (kết quả exploratory không còn là bằng chứng âm tính)

Đối chiếu với HaNoRec pin `587face7` (`utils/data_loader.py`, `hanorec/data/preference.py`, `train/mllm_sft.sh`) và LLaMA-Factory 0.9.3 (`dpo/trainer.py`, `finetuning_args.py`) cho thấy lần chạy exploratory 1-seed lệch protocol ở các điểm sau. Mỗi điểm đã đọc trực tiếp từ code/artifact:

| # | Lệch | Bằng chứng | Sửa |
|---|---|---|---|
| 1 | SFT chỉ học `Yes` (loss `-log P(Yes)` trên positive), không có mẫu `No`; upstream hit=1 trộn ~50% Yes / 50% No | `train.py` SFT loop cũ; reference log P(Yes) ≈ −1e-7 (bão hòa) | `prep.py` thêm `sft_negative` (seeded, train-only, loại item đã thấy); `_build_sft_examples` 50/50 |
| 2 | Policy DPO chạy `train()` với `lora_dropout=0.05` còn reference chạy `eval()`; LLaMA-Factory mặc định `disable_dropout=True` | `dpo/trainer.py:50-59` | `_disable_dropout`; thêm kiểm tra fail-closed `init_parity_max_abs_logit` (policy = reference trước bước DPO đầu) |
| 3 | `target_modules="all-linear"` gắn LoRA vào 162/414 module `visual.*`; upstream đóng băng vision tower và projector | đếm key trong `sft_lora_state.pt`; `finetuning_args.py:427` | `LORA_TARGET_REGEX` chỉ layer ngôn ngữ; `_assert_language_only_lora` |
| 4 | Tie điểm xếp theo item ID (63–66% cặp liền kề tăng dần) | `validation_predictions` của 6 arm | tie giữ thứ tự SASRec |
| 5 | Notebook Colab pilot sẽ crash ở bước kiểm tra scoring batching (`evaluation_rows` rỗng) và chia cho 0 ở mean test | `train.py` guard cũ | guard dùng validation khi test rỗng; `_mean_or_none` |
| 6 | Chạy trên Tesla T4 với bf16 giả lập: `torch.cuda.is_bf16_supported()` trả True trên T4 nên cổng bf16 cũ không chặn gì. Tham số `accelerator=NvidiaL4` cũng bị Kaggle bỏ qua (SDK chỉ liệt kê `NvidiaTeslaT4`, `NvidiaTeslaP100`, `Tpu1VmV38`); push v1 của notebook sửa đã chạy trên T4 và dừng ở assert (log: `got Tesla T4`) | log kernel `trlxun/hanorec-v11-repaired-sft` v1; `kagglesdk/kernels/types/kernels_api_service.py:191-195` | không còn assert T4; ghi `bf16_native=False` vào manifest như một adaptation; `max_pixels` và số epoch suy ra từ tốc độ/VRAM đo được trên T4 |

Chưa chứng minh: lỗi nào trong số này gây ra `NO_SIGNAL_ABOVE_RETRIEVER` (loss khởi đầu ≈ 1.3–2.5 thay vì ≈ ln 2 phù hợp với lỗi #2, nhưng chỉ run GPU mới xác nhận). Vì vậy kết luận đúng là *setup không hợp lệ để kết luận*, không phải *giả thuyết sai*. `lambda_cf` (0.00095–24.8) giữ nguyên công thức đã đóng băng; control hằng số `hardness_control="mean"` được thêm để tách "hardness có thông tin CF" khỏi "beta trung bình khác nhau".

Bản sửa chạy qua hai notebook Kaggle (`research/hanorec_cf_hardness/kaggle/hanorec-v11-repaired-{sft,arms}/`, sinh bởi `scripts/build_hanorec_notebook.py`), tài khoản `trlxun`, GPU T4 (bf16 giả lập), 1 seed, chỉ validation, test không được chấm. Kiểm thử trước khi push: 25 test CPU, và chạy toàn bộ cell của cả hai notebook trên mô hình Qwen2.5-VL thu nhỏ (CPU, dữ liệu tổng hợp) — xác nhận loss khởi đầu = 0.693, parity = 0.0, 14/14 module LoRA là layer ngôn ngữ. Đó là kiểm tra đường chạy, không phải bằng chứng khoa học. Code review độc lập cũng bổ sung: SFT dùng cả mẫu Yes và No cho mỗi pair (gấp đôi số mẫu/epoch và số update so với upstream), đã ghi vào protocol như một adaptation.

## 1. Lần chạy 265 user — kết quả 6 arm (withdrawn)


### Execution status

The resumable Option B design completed all seven real Kaggle kernels:

- SFT kernel: `trlxun/hanorec-cf-hardness-sft`, version 3, `COMPLETE`
- Arm kernels: six `COMPLETE` runs, each loading the frozen SFT bundle and running one DPO/reranking arm
- Account: `trlxun` (Kaggle account A)
- Dataset regime: real Games data, 530 train pairs, 265 evaluation users, 2,082 catalog items
- Each arm: 2 DPO steps, real checkpoint `.pt`, result JSON, and kernel log downloaded locally

No result below is inferred from kernel status alone. Every row was checked from the downloaded `arm_result.json`.

### Arm results

| Weight | Image condition | Status | NDCG@10 | Recall@10 | Candidate recall@20 | Elapsed seconds | Result artifact |
|---:|---|---|---:|---:|---:|---:|---|
| 1.0 | real | COMPLETE | 0.0227874 | 0.0603774 | 0.1169811 | 6888.44 | `research/hanorec_cf_hardness/results/full_run_265/hanorec-cf-hardness-arm-w10-real/arm_result.json` |
| 1.0 | shuffle | COMPLETE | 0.0147305 | 0.0377358 | 0.1169811 | 6891.42 | `research/hanorec_cf_hardness/results/full_run_265/hanorec-cf-hardness-arm-w10-shuffle/arm_result.json` |
| 0.0 | real | COMPLETE | 0.0242022 | 0.0641509 | 0.1169811 | 6611.65 | `research/hanorec_cf_hardness/results/full_run_265/hanorec-cf-hardness-arm-w00-real/arm_result.json` |
| 0.0 | shuffle | COMPLETE | 0.0149298 | 0.0377358 | 0.1169811 | 6405.25 | `research/hanorec_cf_hardness/results/full_run_265/hanorec-cf-hardness-arm-w00-shuffle/arm_result.json` |
| 0.5 | real | COMPLETE | 0.0220688 | 0.0566038 | 0.1169811 | 6099.90 | `research/hanorec_cf_hardness/results/full_run_265/hanorec-cf-hardness-arm-w05-real/arm_result.json` |
| 0.5 | shuffle | COMPLETE | 0.0150677 | 0.0377358 | 0.1169811 | 6053.40 | `research/hanorec_cf_hardness/results/full_run_265/hanorec-cf-hardness-arm-w05-shuffle/arm_result.json` |

Checkpoint artifacts are present beside each result JSON as `arm_w*.pt`; the corresponding kernel logs are present in each arm directory.

### Direct observations

- Real images improve `Recall@10` over shuffled images for every weight:
  - `w=1.0`: 0.0603774 vs 0.0377358
  - `w=0.0`: 0.0641509 vs 0.0377358
  - `w=0.5`: 0.0566038 vs 0.0377358
- Real-image `NDCG@10` is also higher for every weight:
  - `w=1.0`: 0.0227874 vs 0.0147305
  - `w=0.0`: 0.0242022 vs 0.0149298
  - `w=0.5`: 0.0220688 vs 0.0150677
- Candidate recall is identical across all six arms (`0.1169811`). This is expected: the arm reranks the fixed candidate set; it does not change candidate generation.
- Among real-image arms, `w=0.0` has the highest observed NDCG@10 and Recall@10 in this run. This is a descriptive result, not a statistical claim: the evaluation has 265 users and no repeated-seed confidence interval.

### Scope and limitations

This is the largest completed real subset in the current execution: 530 training pairs and 265 evaluation users, not the literal 122,577-row corpus. The six arms share the same frozen candidate construction and evaluation users, so the image-condition comparison is paired at the pipeline level. The report does not claim generalization beyond this Games regime, the selected subset, or the single training seed.

### Next phase

Phase 2 execution is complete. Phase 3 should independently audit the downloaded JSON/checkpoint/log set against the pre-registered gates, then decide whether the observed result supports a follow-up experiment or a stop/null-result conclusion.

## 2. Paired bootstrap audit của lần chạy 265 user


### Contract

- **Outcome:** test the pre-registered real-image versus shuffled-image gate using the six completed arm artifacts.
- **Constraints:** local CPU-only analysis; paired users and targets; no new GPU run; preserve the fixed candidate protocol.
- **Non-goals:** no new seed, no model retraining, no metric or candidate-set changes, no statistical claim beyond this selected 265-user slice.
- **Acceptance:** verify paired keys, compute per-user deltas for NDCG@10 and Recall@10, and report deterministic 95% paired-bootstrap intervals.

### Method

Source files: the six downloaded `arm_result.json` files under
`research/hanorec_cf_hardness/results/full_run_265/hanorec-cf-hardness-arm-w{10,00,05}-{real,shuffle}/`.

Each real arm was paired with its shuffle control by `(user_id, target)`. All
six arms had exactly 265 rows; all paired key sets and targets matched. For
each metric and weight, the audit computed the per-user difference
`real - shuffle`, then generated 20,000 bootstrap resamples of the 265 paired
users using deterministic seed `20260920`. The interval is the percentile 2.5%
to 97.5% interval of the bootstrap mean difference.

### Results

| Weight | Metric | Mean delta | Median delta | 95% bootstrap CI | Positive-user fraction | Gate |
|---:|---|---:|---:|---|---:|---|
| 1.0 | NDCG@10 | 0.0080569 | 0.0000000 | [-0.0018450, 0.0180222] | 5.28% | Not met |
| 1.0 | Recall@10 | 0.0226415 | 0.0000000 | [-0.0037736, 0.0490566] | 3.40% | Not met |
| 0.0 | NDCG@10 | 0.0092724 | 0.0000000 | [-0.0012437, 0.0199733] | 5.28% | Not met |
| 0.0 | Recall@10 | 0.0264151 | 0.0000000 | [0.0000000, 0.0528302] | 3.77% | Not met under strict `CI > 0` rule |
| 0.5 | NDCG@10 | 0.0070011 | 0.0000000 | [-0.0029529, 0.0171577] | 4.91% | Not met |
| 0.5 | Recall@10 | 0.0188679 | 0.0000000 | [-0.0037736, 0.0452830] | 3.02% | Not met |

The positive point estimates are driven by a small number of users whose
ranked target changes. Most paired users have zero delta; this is why the
median delta is zero for every cell and why the intervals are wide relative to
the means.

### Decision

The pre-registered gate is not passed by any cell. The result remains a
**descriptive real-image advantage**, not a confirmed effect. In particular,
`w=0.0` has the best point estimate among real-image arms, but its advantage is
not sufficient to promote CF-only hardness as a validated winner.

No additional GPU training is justified by this gate alone. A future repeated-
seed study is a separate decision and should be opened only with an explicit
new contract and budget.

## 3. Exploratory 1-seed, implementation đã sửa — phân tích zero-GPU


Date: 2026-09-23. GPU cost: 0 (CPU-only reconstruction).

- Run: `research/hanorec_cf_hardness/results/exploratory_1seed/` (`runtime_preprocessing_patch: v11-qwen25vl-min28-letterbox56`, seed 2024, 530 pairs, 25/25 users, 1 epoch, batch 3, max_pixels 4096).
- Script: the workspace analysis script `analyze_hanorec_exploratory.py`. It needs local SASRec checkpoints and data and is not included here. It is deterministic: a rerun produced a byte-identical JSON.
- Raw output: `research/hanorec_cf_hardness/results/exploratory_1seed/zero_gpu_analysis.json`. Its `run_root` field records the original workspace path.
- Reconstruction: validation and test rows were rebuilt locally with the unchanged `prep.prepare`. All 9 pinned inputs are SHA-verified. Only image downloads are stubbed. For every arm, the rebuilt users, targets and candidate sets exactly match the downloaded arm predictions. This makes the SASRec-order baseline directly comparable.

### Verdict

**Exploratory signal: `NO_SIGNAL_ABOVE_RETRIEVER`.** The positive-looking validation contrasts come from 3–4 users. They are indistinguishable from chance. Every reranker arm sits below the frozen SASRec order on the same candidates. No claim transfers to the confirmatory question.

### Evidence

| Validation (25 users, 6 informative) | NDCG@10 |
|---|---:|
| SASRec order (no rerank) | **0.0914** |
| w0.5 real | 0.0761 |
| w0.5 shuffle | 0.0698 |
| w0.0 real | 0.0663 |
| w0.0 shuffle | 0.0636 |
| w1.0 real | 0.0622 |
| Random order (expected) | 0.0545 |
| w1.0 shuffle | 0.0510 |

- Only 6/25 validation users and 1/25 test users have the target inside the top-20 candidates. All other users contribute exactly 0 to every arm.
- Paired bootstrap, 20,000 resamples. Each CI crosses zero:
  - w0.5−w1.0 real: +0.0139, CI [−0.0029, +0.0448], 2 users up / 1 down.
  - w0.0−w1.0 real: +0.0041, CI [−0.0338, +0.0452].
  - real−shuffle: +0.0111, +0.0027 and +0.0063 at w = 1.0, 0.0 and 0.5, each resting on 3–4 changed users.
- Every arm minus the SASRec order is negative (−0.015 to −0.040). In each case 5 of 6 informative users get worse.
- The rerankers are weakly anti-correlated with the SASRec order: mean Spearman is −0.02 to −0.12. Their NDCG is at chance level. Analysis: the Qwen2.5-VL Yes/No score does not yet rank candidates better than random, so there is nothing for a w or real/shuffle contrast to modulate.
- Test (already exposed in the manifest, not used for the decision): 1 informative user. SASRec 0.0142; w1.0 shuffle 0.0126; every other arm 0.

### Training diagnostics (must be resolved before more GPU)

1. **`lambda_cf` is heavy-tailed.**
   - The CF margin is negative for 94.3% of pairs (mean −3.18), because the negative is SASRec's top non-excluded item.
   - The frozen formula `sigmoid(m)/sigmoid(mean m)` therefore gives weights from 0.00095 to 24.8 (median 1.04, p90 9.58). That is a ~26,000× spread.
   - The w0.0 DPO loss peaks at 50.0 (w0.5: 11.4; w1.0: 2.9). About a third of w0.0 loss values exceed 5.
   - CF-only training is therefore dominated by a few pairs. The formula was pre-registered as unclamped, so changing it is a protocol decision, not a bug fix.
2. **The visual input is nearly empty.**
   - max_pixels 4096 resizes the catalog to about 64×64 (the most common final size is 64×64 for 1,988 items).
   - Analysis: with Qwen2.5-VL's 28-px merged patch, that is about 4 visual tokens per image.
   - Real-vs-shuffle is still very different per user: Spearman between the two orders is only 0.47–0.49. But the metric difference stays at noise level, so the contrast cannot test the visual hypothesis at this budget.
3. **The reference may be saturated.** The SFT reference log P(Yes) on the train-positive probe is −4.8e−7 (real) and −1.2e−7 (shuffle). Only one probe was stored. Whether saturation holds on validation cannot be checked, because the SFT bundle (with `sft_validation_predictions`) was not downloaded.
4. **Padding position checked, not a bug.** The pinned tokenizer config (`Qwen/Qwen2.5-VL-3B-Instruct@66285546`) sets no `padding_side`, so HF defaults to right padding. `_batch_answer_logprobs` then reads the true last token.

### Consequence for option B

Option B was a score-only rerun on the existing checkpoints over ~200 informative users. It would most likely only confirm "below SASRec". Scaling the evaluation cannot create a signal the rerankers do not have. The next GPU spend should first answer a narrower question: **does any Qwen2.5-VL reranker (SFT-only or w1.0 real) beat the SASRec order on informative users at all?** Only if yes do w or real/shuffle contrasts become meaningful.

Decisions for the human (not taken here):

- Keep or bound the frozen `lambda_cf` formula, for example with a clamp or rank-based normalization. Either change needs a protocol revision.
- Accept max_pixels 4096 as "near text-only", or budget more memory for images.
- Download or regenerate the SFT bundle to test reference saturation on validation.

### Gaps

- The SFT-only baseline is unavailable locally (`sft_bundle.json` was not in the downloaded output).
- The DPO loss here is a per-micro-batch mean (`train.py:722-730`), and the scale of beta differs by arm. Loss magnitude is compared only within the same arm family. The w1.0 mean loss (~1.85) sits well above ln 2 from the very first micro-batch, even though a fresh DPO adapter should start near ln 2. This is unexplained; one candidate is NoDO noise acting on a saturated SFT reference. It needs a GPU probe.
