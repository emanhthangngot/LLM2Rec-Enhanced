# v11 — HaNoRec CF/history-aware hardness: thiết kế, protocol và calibration

> Tài liệu kỹ thuật đi kèm `v11-hanorec-cf-hardness.md` (báo cáo kết quả). Gộp từ: phần Phương pháp 2 của báo cáo nghiên cứu hai phương pháp (2026-09-20), handoff 2026-09-19 (chi phí đo được, khó khăn triển khai), protocol ledger của implementation đã sửa, và báo cáo calibration. Các phần gốc tiếng Anh được giữ nguyên văn.

## 1. Thiết kế phương pháp

### 1.1. Vấn đề và ý tưởng cốt lõi

HaNoRec gốc định nghĩa hardness chủ yếu từ quan hệ semantic giữa positive và negative. Cách này không biết bộ truy hồi đang gặp khó khăn đến mức nào đối với **một user và một history cụ thể**. Phương pháp hiện tại bổ sung margin từ SASRec đã đóng băng:

```text
m_CF(h_u, i+, i-) = s_SASRec(h_u, i+) - s_SASRec(h_u, i-)
```

Margin nhỏ nghĩa là SASRec khó phân biệt positive với hard negative trong context đó. Tín hiệu này không thay semantic hardness; nó tạo một trục hardness cộng tác, có điều kiện theo user/history.

### 1.2. Kiến trúc

```mermaid
flowchart LR
    A[Frozen LLM2Rec title embeddings] --> B[Frozen SASRec]
    H[Train histories + targets] --> B
    B --> C[Top non-target hard negatives]
    B --> D[Real top-20 test candidates]
    E[Item title + image] --> F[Qwen2.5-VL]
    F --> G[Semantic hardness lambda_sem]
    C --> I[CF margin lambda_cf]
    G --> J[lambda_sem^w * lambda_cf^(1-w)]
    I --> J
    H --> K[Qwen2.5-VL LoRA SFT]
    K --> L[Hardness-scaled DPO + NoDO]
    J --> L
    L --> M[log P(Yes) - log P(No)]
    D --> M
    M --> N[Reranked top-20]
```

Phương pháp giữ nguyên backbone retrieval để câu hỏi chỉ còn là: **hardness có điều kiện theo CF giúp preference optimization/reranking tốt hơn semantic hardness không?** Nếu retriever cũng được thay đổi đồng thời, không thể biết gain đến từ hardness hay từ retrieval.

### 1.3. Dữ liệu đầu vào và pair construction

- Frozen LLM2Rec title embeddings từ checkpoint Qwen2-0.5B.
- Frozen SASRec Games, hidden size 128, 2 layers, 2 heads, dropout 0.3, max history length 10.
- Mỗi training row cần ít nhất ba item history; dùng ba item cuối trước target làm `h_u`.
- Positive `i+` là target kế tiếp thực.
- Chấm điểm toàn catalog bằng SASRec.
- Negative `i-` là non-target có score cao nhất sau khi loại padding và các tương tác tương lai đã biết bằng prefix expansion chính xác.
- Lưu `cf_margin` cùng mọi định danh và thứ tự item.

Loại trừ future interactions giảm false negative, nhưng không giải quyết được positive chưa quan sát. Vì vậy negative được gọi là “hard candidate”, không được diễn giải chắc chắn là item user không thích.

Đánh giá dùng đúng top-20 candidate do SASRec sinh ra. Target không được chèn nhân tạo. Do đó `candidate_recall@20` là trần của reranker: target không nằm trong candidate thì reranker không thể khôi phục.

### 1.4. Hai nguồn hardness và cách kết hợp

**Semantic hardness.** Qwen2.5-VL tạo text/vision embeddings; similarity profile của item được chọn và item bị từ chối được xây từ top-K neighborhood, rồi đo khoảng cách giữa hai profile và chuẩn hóa theo quy ước HaNoRec.

**CF hardness.** Protocol cố định:

```text
lambda_cf = sigmoid(m_CF) / sigmoid(mean(m_CF over batch))
```

Không được đảo dấu sau khi thấy kết quả. Tên “hardness” phụ thuộc quy ước scaling của HaNoRec; điều cần bảo toàn là đúng công thức và đúng mapping vào DPO.

**Geometric mixture:**

```text
lambda_combined = lambda_sem^w * lambda_cf^(1-w)
```

Ba nhánh:

- `w=1.0`: semantic-only, đối chứng HaNoRec.
- `w=0.0`: CF-only.
- `w=0.5`: kết hợp hai nguồn.

Mỗi nhánh chạy với ảnh `real` và ảnh `shuffle`, tổng cộng sáu cell. Real–shuffle phải dùng cùng pair/candidate protocol; nếu không, chênh lệch có thể do sample composition thay vì visual evidence.

### 1.5. SFT, DPO và reranking

**Policy.** Qwen2.5-VL-3B-Instruct được nạp 4-bit, fine-tune bằng LoRA trên `q_proj`, `k_proj`, `v_proj`, `o_proj`, rank 8, alpha 32, dropout 0.05.

**SFT.** Prompt chứa ba item history và một candidate, mỗi item có title/image. Model trả lời binary:

```text
Based on the user's history, will they like this candidate item next? Answer Yes or No.
```

Positive được huấn luyện là `Yes`. Bundle SFT phải bảo toàn pair, eval rows, image paths/hashes, hardness arrays, item order và reference metadata.

**DPO.** Mỗi training pair tạo hai preference examples:

- positive: chọn `Yes`, từ chối `No`;
- hard negative: chọn `No`, từ chối `Yes`.

Responsiveness và hardness điều chỉnh beta theo từng example:

```text
beta_i = max(1e-6, beta0 * responsiveness_i * lambda_combined_i)
```

NoDO tạm thời perturb LoRA parameters trong policy forward. Reference forward dùng base model với adapter disabled và không bị perturb. Loss là negative log-sigmoid của policy-vs-reference preference logit đã scaling. Fail-closed khi loss non-finite hoặc LoRA không thay đổi.

**Reranking.** Mỗi candidate trong real top-20 được chấm đúng một lần bằng:

```text
score(candidate) = log P(Yes) - log P(No)
```

Sắp xếp giảm dần; lưu ranking, target rank, `NDCG@10`, `Recall@10` và `candidate_recall@20`.

### 1.6. Quy mô và chỉ số

Protocol full run cố định ở 530 training pairs, 265 evaluation users, history length 3, top-20 candidates, seed 2024, hai bước SFT và hai bước DPO. Đây là quy mô theo ngân sách, không phải full convergence hay variance estimate đủ cho publication.

Chỉ số chính:

- `NDCG@10` sau rerank trên SASRec real top-20.
- `Recall@10` trên cùng candidate set.
- `candidate_recall@20` như diagnostic ceiling, phải báo cáo riêng.

Phép tương phản bắt buộc là `real - shuffle` tại từng `w`. Quy tắc thắng chỉ hợp lệ khi confidence interval của contrast không chứa zero. Không được lấy một mean tốt tại `w=0.5` làm kết luận nếu không so với `w=1`, `w=0` và shuffle tương ứng. HR@3/NDCG@3 của protocol HaNoRec gốc là phụ, không được trộn với top-20 LLM2Rec protocol.

### 1.7. Rủi ro diễn giải

- Hard negative có thể là positive chưa quan sát; DPO có thể học từ nhãn sai.
- Reranker bị giới hạn bởi candidate recall; gain không đồng nghĩa retriever tốt hơn.
- Một seed, hai bước và 530 pairs chỉ đủ cho feasibility/controlled comparison, không đủ khẳng định ổn định.
- Real–shuffle có thể bị nhiễu bởi ảnh lỗi, khác biệt preprocessing hoặc pair mismatch; hash và row identity phải được audit.
- Chi phí Qwen2.5-VL lặp lại cho sáu nhánh khiến sample size nhỏ; confidence interval quan trọng hơn loss.


> Lưu ý đọc: công thức trong code là `lambda = lambda_sem ** w * lambda_cf ** (1 - w)` (`research/hanorec_cf_hardness/train.py:637-641`), nên `w = 1.0` là thuần semantic và `w = 0.0` là thuần CF.


## 2. Chi phí đo được (calibration V2)

Calibration V2 dùng 40 cặp huấn luyện, 20 người dùng đánh giá, và sáu nhánh đã đăng ký:

| Giai đoạn | Chi phí đo được |
|---|---:|
| Nạp mô hình | 41.90s |
| Semantic embedding | 0.0853 s/item |
| SFT | 0.6198 s/example |
| Lượt thống kê DPO | 0.8358 s/example |
| Lượt huấn luyện DPO | 1.0791 s/example |
| Reranking | 0.4328 s/candidate |

Lần hiệu chuẩn kết thúc sau 2.543,5 giây thời gian thực. Năm nhánh hoàn tất; nhánh thứ sáu dừng lại gọn gàng tại ranh giới ngân sách. Chính các phép đo này, chứ không phải các phỏng đoán dựa trên số lượng tham số, là cơ sở cho lần chạy 530 cặp/265 người dùng.

## 3. Các khó khăn triển khai đã gặp

- `device_map="auto"` đã phân mảnh (shard) mô hình lượng tử hóa không đồng đều trên 2×T4 và gây OOM; bản triển khai ghim toàn bộ mô hình vào GPU 0.
- `get_peft_model()` làm biến đổi/bọc mô hình nền; việc giữ một biến Python khác không tạo ra một mô hình tham chiếu độc lập. Logit của reference nay dùng `disable_adapter()` ở chế độ eval.
- Tích lũy các loss rồi gọi backward một lần đã giữ lại nhiều đồ thị kích hoạt và gây OOM. Huấn luyện nay gọi backward cho từng ví dụ với loss đã chuẩn hóa.
- Việc chạy forward riêng biệt cho Yes và No làm tăng gấp đôi chi phí. Cả hai logit nay được đọc từ một lượt forward duy nhất tại vị trí token kế tiếp.
- Việc kiểm tra ngân sách chỉ giữa các nhánh có thể gây vượt quá bên trong các vòng lặp dài. Các lần kiểm tra nay diễn ra bên trong các vòng lặp SFT, thống kê DPO, huấn luyện DPO, và reranking.
- Hai lần chạy notebook nguyên khối kết thúc với `CANCEL_ACKNOWLEDGED`: ô điều phối vượt quá timeout phản hồi 1.800 giây quan sát được của nbclient, và trần phiên nền tảng thực tế là khoảng sáu giờ. Bảy script kernel hiện tại loại bỏ cả hai điểm ghép nối này.
- Việc tải ảnh tuần tự ở quy mô lớn gặp phải các lỗi reset/timeout kết nối thực sự; việc lấy ảnh nay dùng bốn lần thử với độ trễ tăng dần.
- Huấn luyện theo nghĩa đen trên toàn bộ các hàng là bất khả thi: 122.577 cặp × hai ví dụ nhị phân × hai bước DPO × sáu nhánh là khoảng 2,94 triệu lượt lặp ví dụ, dự phóng gần 2.043 giờ GPU trước cả SFT/reranking. "Quy mô đầy đủ" ở đây nghĩa là tập con thực lớn nhất đã đăng ký mà ngân sách đo được hỗ trợ, không phải toàn bộ các hàng.


## 4. Fidelity và protocol của implementation đã sửa


### Decision boundary

This package reproduces the observable HaNoRec `hit=1` trainer contract on a deliberate Games adaptation. It does **not** claim native HaNoRec published-score reproduction. A Kaggle `COMPLETE` smoke run is execution evidence only; it is not signal evidence.

### Authority

- HaNoRec: commit `587face74524e4553b5a7aa295fe962004682382`, archive SHA-256 `48c52e...3d6a6`.
- LLM2Rec: commit `73b481f710f67166ab958f4985d27b27fb410871`, archive SHA-256 `6bcee7...c1cc8`.
- LLaMA-Factory: `llamafactory==0.9.3.dev0` from the pinned upstream requirement. The requirement does not expose a source commit, so this work proves the HaNoRec observable trainer behavior from the pinned source rather than claiming byte-identical dependency internals.

### Frozen behavior matrix

| Contract | Upstream authority | Corrected Games implementation | Adaptation label |
|---|---|---|---|
| SFT precedes DPO | `README.md`, `microlens_hit1.yaml` | SFT adapter is trained first; DPO uses a fresh adapter over the frozen SFT adapter | Games model/data and subset only |
| Reference | `trainer/dpo.py` | Reference activates SFT adapter only; DPO adapter is excluded | PEFT adapter stacking is explicit |
| DPO objective | `trainer/dpo.py` | Same preference-logit subtraction, sigmoid loss, beta floor, and detached hardness/statistics | Candidate scoring is Games-specific |
| Responsiveness | `trainer/dpo.py`, `hars/math.py` | One global mini-batch statistic, requiring at least three examples | No distributed gather in single-GPU smoke |
| NoDO | `nodo/hooks.py` | Forward-hook perturbation with generator, no parameter mutation, exception cleanup | Single-GPU smoke |
| Hardness | `hars/hardness.py` | Fused train-only title/visual catalog and exact Top-K math | Games CF hardness is an added condition |
| Data preparation | upstream preference builder | Train pairs and hardness use only train rows/items; held-out rows are inference-only | True next-item Games targets |
| Evaluation | README (`AUC` for native hit-1) | Frozen SASRec candidate set; report NDCG@10/Recall@10 and candidate recall | Deliberate Games reranking adaptation |

### Data contract

The train item universe is derived before validation/test rows are consulted. Negatives exclude padding, the positive, every observed item in the user row, and train-known future items. Evaluation targets and candidates can add inference assets but cannot change the training pair, hardness, or shuffle hashes. Selection is seeded at user-row level; no fixed prefix is used.

The shuffle control is a deterministic fixed-point-free bijection within frequency strata. Singleton strata are merged with adjacent strata before derangement. The same mapping is applied to SFT prompts, DPO prompts, hardness visual inputs, and evaluation prompts for the shuffle condition.

### Training contract

The implementation uses real micro-batches, `drop_last`, gradient accumulation, partial-accumulation scaling, AdamW, cosine schedule, warmup zero, gradient clipping one, and deterministic seeds for Python/NumPy/Torch CPU/CUDA/data/noise generators. Checkpoints carry model, optimizer, scheduler, RNG, stage, and count state. `COMPLETE` requires every expected stage and exact observed counts.

A correctness smoke is intentionally smaller than the five-epoch research recipe and is stamped as non-signal. Full signal execution is quota-gated after smoke timings and must use seeds 2024/2025/2026, real/shuffle parents, weights 0/0.5/1, validation selection, and untouched confirmation scoring.

### Withdrawn historical evidence

The old 530-pair pilot (the 265-user six-arm run) remains stored under `research/hanorec_cf_hardness/results/full_run_265/`, and its results and bootstrap audit (sections 1–2 of `v11-hanorec-cf-hardness.md`) remain historical artifacts. Its mechanism acceptance is withdrawn because it leaked held-out catalog items, admitted training-history negatives, used an unmatched shuffle, used base-model reference semantics, and executed two full-dataset updates instead of the configured mini-batch recipe.

## 5. Báo cáo calibration


### Executive summary

- **Issue:** the calibration push (`hanorec-cf-hardness-calibration`, 40 train
  pairs, 20 eval users, 6 arms) ran far longer than the 3000s internal safety
  cap and ended in `KernelWorkerStatus.CANCEL_ACKNOWLEDGED` — this repository
  never issued a cancel; someone/something on the Kaggle side stopped it.
- **Impact:** no clean sizing numbers yet. 3 of 6 arms produced real
  checkpoints before cancellation; the final `hanorec_cf_hardness_result.json`
  (which would carry the real `stage_timings`) was never written because
  cancellation happened before that write.
- **Root cause (code-side, confirmed):** the pre-fix budget check
  (`remaining() <= 0`) was only evaluated between arms/steps, not inside the
  per-example loops. At 5x scale (40 pairs/20 users vs. the smoke test's 8/4),
  a single arm's stats+train pass over 80 DPO examples ran for several
  hundred seconds without any chance to bail early, so the 3000s cap could
  only take effect at the *next* arm boundary — by which point real wall time
  had already run well past it.
- **Status:** fixed in code (`train.py` now checks `remaining()` inside the
  SFT loop, both DPO per-example loops, and the reranking per-row loop — see
  `plan.md` Phase 2 Section 3). Not yet re-verified with a fresh push.
- **Fix:** re-run calibration with the fixed code, and get user confirmation
  on whether they cancelled the prior run before spending more quota.

### Real evidence recovered

Downloaded via `kaggle_kernel_output` from the cancelled kernel
(the downloaded cancelled-run output (workspace-only, not included in this repository)):

| Artifact | Real, present | Note |
|---|---|---|
| `experiment.json` | Yes | Calibration config: `train_pairs=40`, `eval_users=20`, same 6 arms |
| `prepared_manifest.json` | Yes, 102.5KB | Real `prep.prepare()` output — 40 pairs, 20 eval rows built successfully |
| `images/` | Yes, 334+ real downloaded JPGs | Confirms catalog grew to the real logged `catalog_items: 340` |
| `sft_lora_state.pt` | Yes, 14.2MB | Real SFT completed |
| `arm_w1.0_real.pt` | Yes, 14.2MB | Real arm 1/6 completed |
| `arm_w1.0_shuffle.pt` | Yes, 14.2MB | Real arm 2/6 completed |
| `arm_w0.0_real.pt` | Yes, 14.2MB | Real arm 3/6 completed |
| `arm_w0.0_shuffle.pt`, `arm_w0.5_real.pt`, `arm_w0.5_shuffle.pt` | **Missing** | Arms 4-6 never ran; cancellation occurred during or before arm 4 |
| `hanorec_cf_hardness_result.json` | **Missing** | Only written at the very end of `train_and_evaluate`; never reached |
| `.log` | Present but truncated at t≈133s | Kaggle's output download for a `CANCEL_ACKNOWLEDGED` kernel does not appear to return the full accumulated stdout — the `log_stage()` prints added for exactly this purpose are not visible in the retrieved log despite real work clearly having happened downstream (3 real checkpoints exist) |

Log's last real line: `{'train_pairs': 40, 'eval_rows': 20, 'catalog_items': 340}` at t=103.8s (right after `prep.prepare()` returns), followed by the Qwen2.5-VL slow-processor warning at t=133.1s (model load). Nothing after that was retrievable.

### What this does tell us, honestly

- The pipeline is real and correct at 5x scale: `prep.py` built 40 real pairs
  and 20 real eval rows from a 340-item real catalog with real downloaded
  images, and 3 full real SFT+DPO+checkpoint cycles completed.
- **It does not tell us precise seconds/unit.** File download timestamps are
  local (download time), not Kaggle-side creation time, so per-arm duration
  cannot be reconstructed from file metadata. The one piece of real
  timing evidence (t=103.8s for `prepare()`, t=133.1s for model load) only
  covers setup, not the expensive stages.
- Total wall time from push to observed cancellation was somewhere between
  ~4,400s and ~6,200s (bounded by my own poll intervals, not by an exact
  Kaggle timestamp) for 3 of 6 arms plus setup — order-of-magnitude **1.2 to
  1.7 hours for half the arms**, i.e. plausibly 2.5-3.5 hours for the full 6
  at this scale, well past both my 3000s intended cap and a comfortable
  calibration budget.

### Correction after re-investigation

The "ran far longer than the 3000s internal safety cap" framing above was
wrong. Re-reading the v1 notebook's orchestration cell before re-pushing
revealed the config's declared `budget_seconds: 3000` was silently
overridden: `EXPERIMENT_CONFIG["budget_seconds"] = max(60.0, DEADLINE -
time.monotonic())` where `DEADLINE = time.monotonic() + 6600.0` (set in the
setup cell). The real intended cap was **~6600s (110 minutes)**, not 3000s.
The observed 4,400-6,200s cancellation happened *before* that real deadline,
meaning v1 was cancelled by an external actor mid-run, not because it
overran its own budget check — the coarse between-arm-only check just never
got a chance to matter either way. Fixed in the v2 re-sync (Section below):
`DEADLINE` lowered to 2400s and the config's stated `budget_seconds` made
consistent with it.

### v2 re-run (COMPLETE, real numbers)

Re-synced `train.py` cell (index 5, exact-index + prefix-assert match, not
substring search) from the fixed disk file — fine-grained `remaining() <= 0`
checks now sit inside the SFT loop, both DPO per-example loops (stats pass
and train pass), and the reranking per-row loop, plus `stage_timings`
instrumentation via `log_stage()`. Lowered `DEADLINE` (cell 1) from 6600s to
**2400s** to keep this calibration spend small, and added a print of the
dynamically-computed effective `budget_seconds` for transparency. Pushed as
version 2, `kaggle_kernel_push` returncode 0.

Polled every 60s; reached `KernelWorkerStatus.COMPLETE` after **2543.5s**
total kernel wall time (0.706 GPU-hours) — no cancellation, no timeout,
clean self-termination.

Downloaded via `kaggle_kernel_output` to
the downloaded v2 output (workspace-only, not included in this repository). Real artifacts confirmed present:

| Artifact | Present | Note |
|---|---|---|
| `hanorec_cf_hardness_result.json` | Yes, 58,020 bytes, valid JSON | Full 8 top-level keys including `stage_timings` (18 real entries) and `arms` (6 entries) |
| `arm_w1.0_real.pt`, `arm_w1.0_shuffle.pt`, `arm_w0.0_real.pt`, `arm_w0.0_shuffle.pt`, `arm_w0.5_real.pt` | Yes, 14.2-14.9MB each | 5/6 arms `COMPLETE` |
| `arm_w0.5_shuffle.pt` | Correctly absent | 6th arm cleanly `SKIPPED_BUDGET` per `result.json["arms"][5]`, not a download failure — confirmed by the arm's own status field, not inferred from the missing file |
| `images/` | 213 real downloaded JPGs | Client-side download of this directory did not fully mirror the kernel's 340-item catalog before the wrapping shell command's own 180s timeout was hit; irrelevant to throughput analysis (checkpoints + `result.json` were already fully materialized on disk before that timeout) |

`reference_reproducibility_check`: `first`/`second` both `-0.9375150203704834`
— exact match, confirms `disable_adapter()` frozen-reference fix holds at 5x
scale.

Real per-stage timings (seconds), used to derive Section 5's sizing
coefficients in `plan.md`'s Phase 2 file:

| Stage | Seconds | Units | s/unit |
|---|---|---|---|
| `model_load` | 41.90 | — | — |
| `semantic_embedding` | 29.02 | 340 items | 0.0853 |
| `sft_total` | 49.59 | 80 examples | 0.6198 |
| `dpo_stats_pass_*` (5 arms) | 132.8-134.9 | 160 examples each | 0.830-0.843 |
| `dpo_train_pass_*` (5 arms) | 171.4-174.2 | 160 examples each | 1.071-1.089 |
| `rerank_*` (4 full arms + 1 partial) | 172.8-175.0 (full), 34.4 (partial, 80/400 units) | 400 (full), 80 (partial) | 0.429-0.435 |

Per-unit costs are consistent within ~2% across all 5 completed arms — a
trustworthy basis for sizing. Full sizing model and 2-3 concrete `(P, U)`
options: `the workspace plan (phase 2, full-dataset scaling)`
Sections 5-6.

## 6. Tài liệu tham chiếu ngoài

- [HaNoRec repository](https://github.com/wangyu0627/HaNoRec) — upstream hardness-aware multimodal preference optimization implementation.
- [HaNoRec paper](https://arxiv.org/abs/2511.18740) — HaRS/NoDO motivation and preference optimization design.
- [Qwen2.5-VL documentation](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl) — multimodal model interface and supported inputs.
