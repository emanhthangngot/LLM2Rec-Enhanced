# Tổng hợp cuối cùng về các hướng đã triển khai với LLM2Rec

**Mục đích:** tài liệu báo cáo với giáo viên về pipeline LLM2Rec, các hướng multimodal đã thử, quy trình train/evaluate, kết quả định lượng, nguyên nhân thất bại và hướng còn tiềm năng nhất.

**Trạng thái bằng chứng:** các con số trong tài liệu lấy từ những report và artifact đã lưu trong repository. Cần phân biệt ba loại kết quả:

1. **Compatibility-profile:** tái dựng pipeline LLM2Rec dưới giới hạn Kaggle T4, không phải reproduction tuyệt đối paper.
2. **Kết quả thực nghiệm có control:** visual arm được so sánh với text-only, frequency-matched shuffle, Gaussian random code, text-PCA và constant control.
3. **Kết quả exploratory hoặc đã supersede:** một số headline gain ban đầu thay đổi sau khi loại immediate-repeat rows, sửa bidirectional mask hoặc thay đổi checkpoint provenance.

---

## 1. Executive summary

LLM2Rec được tái dựng theo pipeline:

```text
AmazonMix-6
    -> CSFT trên Qwen2-0.5B
    -> MNTP
    -> SimCSE
    -> title embedding table
    -> adapter + sequential recommender
    -> full-catalog ranking
```

Các recommender đã dùng:

- SASRec;
- GRU4Rec;
- BERT4Rec.

Các hướng đưa visual information vào model gồm:

1. cộng visual branch trực tiếp vào item embedding;
2. đưa visual vào sequence/history representation;
3. cộng visual score residual ở ranking boundary;
4. gating residual theo item exposure/frequency;
5. policy học để chọn text hay visual theo từng context.

Kết luận tổng quát:

> Hướng có tiềm năng nhất hiện tại là **frozen score-level visual residual**: giữ nguyên text-based recommender, giữ nguyên candidate table, chỉ cộng một visual similarity residual đã chuẩn hóa vào score cuối cùng.

Công thức:

```text
score_final(u, i)
    = score_text(u, i)
    + alpha * z_visual(u, i)
```

Trong đó:

- `score_text` là score từ LLM2Rec text path;
- `z_visual` là visual similarity residual giữa user visual history profile và item CLIP feature;
- `alpha` là một scalar chọn trên validation;
- recommender và text candidate table được freeze trong frozen protocol.

Hướng này tiềm năng vì:

- bảo toàn được text baseline tại `alpha = 0`;
- không làm thay đổi candidate embedding table;
- không cần train thêm multimodal architecture lớn;
- giảm nguy cơ visual branch phá hỏng text geometry;
- có thể kiểm tra causal contribution bằng matched shuffle;
- đã cho bằng chứng visual-specific trên Games với SASRec và BERT4Rec trong một protocol cụ thể.

Tuy nhiên, chưa được gọi là method tổng quát hoặc dataset-independent. Các giới hạn chính:

- Sports transfer không vượt qua toàn bộ control gate;
- GRU4Rec có baseline text gần như degenerate và text-PCA còn mạnh hơn visual;
- patched-IEM frozen rerun tăng Recall nhưng không đạt matched-control NDCG gate;
- learned selective policy không thắng `always-visual`;
- toàn bộ method expansion hiện đã dừng ở `STOP2 — Method Exhausted`.

Kết luận khoa học an toàn nhất:

> Aligned image features có thể mang lại visual-specific offline ranking utility trên Games khi kết hợp với một text backbone đủ mạnh và đưa vào ở score boundary. Nhưng bằng chứng hiện tại chưa đủ để kết luận utility này tổng quát qua dataset, checkpoint pipeline và deployment policy.

---

## 2. Pipeline LLM2Rec đã tái dựng

### 2.1. Stage 1 — CSFT

Base model là `Qwen2-0.5B`. Model được continual/supervised fine-tune trên dữ liệu AmazonMix-6 để thích nghi với dữ liệu item và interaction của recommender system.

Paper profile:

- full fine-tuning;
- `use_lora=False`;
- `train_from_scratch=False`;
- batch hiệu dụng: `128`;
- micro-batch: `4`;
- learning rate: `3e-4`;
- cutoff length: `1024`;
- số bước chính thức: khoảng `10,000`.

Pipeline thực thi trên Kaggle bị giới hạn bởi một T4 nên sử dụng compatibility profile:

- một GPU thay vì hai GPU;
- FP16 thay vì BF16;
- SDPA thay vì FlashAttention2;
- khoảng `1,000` CSFT steps thay vì `10,000`;
- micro-batch và gradient accumulation để giữ batch hiệu dụng gần profile mục tiêu.

Do paper không cung cấp checkpoint đầy đủ và upstream repository không có LICENSE rõ ràng cho việc redistribute, model phải được train lại. Vì vậy đây không phải reproduction binary-identical.

### 2.2. Stage 2 — IEM bằng MNTP

MNTP là bước tiếp tục pretrain model trên sequence/title-related data:

- batch size: `32`;
- max sequence length: `512`;
- masking probability: `0.2`;
- blank mask;
- khoảng `1,000` training steps;
- gradient checkpointing.

Một lỗi quan trọng được phát hiện trong quá trình audit là bidirectional behavior của Qwen2 path. Sau khi patch, preflight kiểm tra future-token sensitivity cho thấy model thực sự nhìn được thông tin hai chiều trong IEM path.

Patched preflight ghi nhận:

```text
Qwen2BiModel
attention backend: SDPA
is_causal: false
future-token prefix delta: 20.46875
```

### 2.3. Stage 3 — SimCSE

SimCSE được dùng để cải thiện chất lượng sentence/title embedding:

- learning rate: `2e-4`;
- batch profile chính thức: `256`;
- thực thi Kaggle: per-device batch `32`;
- epochs: `5`;
- warmup: `300`;
- dropout: `0.2`;
- loss scale: `10`;
- seed: `42`;
- mean pooling;
- bidirectional representation;
- evaluate checkpoint `500` và `1,000`.

### 2.4. Stage 4 — Extract title embedding

Sau IEM, title của mỗi item được đưa qua LLM để tạo item embedding table.

Embedding table này là đầu vào cho downstream recommender. Với visual experiments, title table thường được freeze để tách ảnh khỏi tác động của text training.

### 2.5. Stage 5 — Downstream recommender

SASRec:

- sequential self-attention;
- cross-entropy loss;
- learning rate `1e-3`;
- weight decay `1e-4`;
- dropout `0.3`;
- full-catalog scoring.

BERT4Rec:

- title-conditioned input;
- adapter bias-free `896 -> 128`;
- hai bidirectional Transformer layers;
- hai attention heads;
- inner size `512`;
- dropout `0.2`;
- dynamic Cloze masking `20%`;
- masked-position cross-entropy;
- final-mask full-catalog inference.

GRU4Rec được dùng như cross-backbone stress test, nhưng text baseline của nó gần như degenerate nên không thể dùng để xác nhận transfer mạnh.

### 2.6. Evaluation protocol

Các metric chính:

- `NDCG@10`;
- `Recall@10`;
- `NDCG@20`;
- `Recall@20`.

Các evaluation arms được chạy trên ba seed:

```text
2024, 2025, 2026
```

Primary analysis tách:

- immediate-repeat targets;
- novel targets.

Điểm này rất quan trọng vì các Amazon 5-core split có growing prefixes. Một aggregate metric có thể trộn nhiệm vụ repeat prediction với novel-item recommendation.

---

## 3. Text-only reproduction và baseline fidelity

### 3.1. Kết quả current compatibility baseline

Sau khi audit lại raw ranks và patched bidirectional path, kết quả compatibility baseline hiện tại là:

| IEM checkpoint | NDCG@10 | Recall@10 |
|---|---:|---:|
| 500 | 0.049752161 | 0.082077051 |
| 1000 | 0.050377072 | 0.082055299 |

Paper reference trên Games:

```text
NDCG@10 = 0.0521
Recall@10 = 0.0865
```

Gap hiện tại:

| Checkpoint | Gap NDCG@10 | Gap Recall@10 |
|---|---:|---:|
| 500 | -4.51% | -5.11% |
| 1000 | -3.31% | -5.14% |

Raw rank recomputation trên `15,323` evaluation rows cho agreement trong khoảng `1e-8`.

### 3.2. Patched-IEM retraining

Patched IEM được train lại từ trước MNTP và SimCSE. Kết quả mới hơn, dùng làm comparator cho patched visual rerun:

| IEM checkpoint | NDCG@10 | Recall@10 |
|---|---:|---:|
| 500 | 0.046365920 | 0.078183124 |
| 1000 | 0.047502026 | 0.079444841 |

Đây là lý do phải tách hai nhóm số liệu:

- các visual report cũ dựa trên v8/v5 compatibility table;
- patched-IEM visual rerun dùng table được train lại.

Không được lấy visual gain trên table cũ rồi gắn trực tiếp vào patched baseline mới.

Nguồn: `plans/reports/report-260909-llm2rec-text-baseline-fidelity-audit.md` và `plans/260827-1532-llm2rec-baseline-preserving-visual-fusion/reports/patched-i1a-frozen-alpha-report.md`.

---

## 4. Hướng 1 — Additive visual item embedding, v9

### Method

CLIP image feature được đưa qua một MLP rồi cộng trực tiếp vào item embedding:

```text
item_embedding_final
    = text_item_embedding
    + MLP(CLIP_image_embedding)
```

Đặc điểm:

- CLIP feature dimension: `512`;
- feature table khoảng `9,514` item có ảnh hợp lệ;
- visual MLP khoảng `575,360` trainable parameters;
- text adapter chỉ khoảng `114,816` parameters;
- visual branch lớn hơn text adapter khoảng `5.01` lần.

### Kết quả

| Arm | Mean NDCG@10 |
|---|---:|
| T-null | 0.048886 |
| T-mask | 0.048247 |
| L-shuffle | 0.042679 |
| L-real | 0.040156 |

Real visual so với shuffle:

| Seed | Relative delta |
|---|---:|
| 2024 | -1.104% |
| 2025 | -16.315% |
| 2026 | +1.789% |

`L-shuffle` cũng giảm mạnh so với text control. Điều đó cho thấy branch visual ngẫu nhiên đã làm thay đổi representation và optimization trajectory, nên real-vs-shuffle không còn là phép đo visual semantics sạch.

### Kết luận

V9 bác bỏ thiết kế **late additive item-embedding fusion** cụ thể. Nó không bác bỏ multimodal recommendation nói chung.

Nguyên nhân chính:

1. visual branch quá lớn;
2. candidate item geometry bị thay đổi;
3. history representation cũng bị thay đổi;
4. random visual input đã gây ranking damage;
5. khó tách visual information khỏi optimization noise.

Nguồn: `plans/reports/report-2026-08-25-llm2rec-visual-signal-v9.md`.

---

## 5. Hướng 2 — Sequence-side visual fusion S1/S2

### 5.1. S1: đưa visual vào sequence/history state

S1 đưa visual information vào quá trình hình thành user sequence representation.

Kết quả aggregate ban đầu có vẻ dương, nhưng audit theo từng loại target cho thấy:

- `740 / 15,323` test rows là immediate-repeat;
- các rows này chiếm khoảng `92%` aggregate `q-real - q-shuffle` NDCG gain;
- trên `14,583` novel-target rows, real visual gần như tie hoặc thấp hơn text control.

Một recency reranker không dùng ảnh lại thắng q-real trên cả ba seed:

| Seed | Recency thắng q-real |
|---|---:|
| 2024 | +11.77% |
| 2025 | +8.77% |
| 2026 | +24.59% |

Điều này cho thấy S1 học copy/recency behavior nhiều hơn là visual semantics.

### 5.2. S2: repeat-debiased training

S2 loại bỏ `3,699` immediate-repeat training rows, giữ checkpoint budget cố định và đánh giá novel targets.

Tất cả các contrast chính đều có confidence interval chứa zero:

- real - null;
- real - text-PCA;
- real - random-code;
- real - mean shuffle.

Immediate-repeat NDCG giảm mạnh sau khi loại repeat rows:

| Condition | q-null | q-real |
|---|---:|---:|
| S1, repeat có trong training | 0.510240 | 0.611478 |
| S2, epoch 25 | 0.257375 | 0.260284 |
| S2, epoch 50 | 0.169689 | 0.157638 |

### Kết luận

S1/S2 không chứng minh sequence-side visual utility. Chúng cung cấp một phát hiện phương pháp luận quan trọng:

> Aggregate NDCG trên Amazon 5-core có thể trộn repeat prediction và novel-item recommendation. Nếu không tách hai loại target, multimodal gain có thể bị đánh giá sai.

Nguồn: `reports/s1-sequence-side-deep-report.md` và `reports/s2-repeat-debiased-report.md`.

---

## 6. Hướng 3 — Frozen score-level visual residual

Đây là hướng tiềm năng nhất hiện tại.

### 6.1. Ý tưởng method

Text recommender là authority. Visual không được phép sửa trực tiếp item table hoặc toàn bộ sequence model. Visual chỉ tạo một score residual:

```text
score_text(u, i)
    = text user state dot text item representation

z_visual(u, i)
    = standardized cosine(
          recency_weighted_visual_history(u),
          CLIP_item(i)
      )

score_final(u, i)
    = score_text(u, i) + alpha * z_visual(u, i)
```

Các invariant quan trọng:

1. `alpha = 0` phải tạo ranking y hệt text-only.
2. Text candidate table không được thay đổi.
3. Text recommender checkpoint phải freeze trong frozen protocol.
4. Alpha chỉ chọn trên validation.
5. Test labels chỉ dùng để báo cáo cuối.
6. Real image phải được so với các control có cùng shape và cùng frequency distribution.

### 6.2. G0 exact parity

G0 chạy `text-exact` và `s-null` trên ba seed.

| Seed | Rows | Rank churn text-exact vs s-null |
|---:|---:|---:|
| 2024 | 15,323 | 0 |
| 2025 | 15,323 | 0 |
| 2026 | 15,323 | 0 |

Tất cả top-20 arrays giữa text-only, null residual và raw baseline đều giống nhau hoàn toàn.

Đây là điều kiện cần để biết visual branch không tự làm thay đổi baseline khi tắt.

Nguồn: `reports/g0-parity-audit.md`.

### 6.3. I1-A frozen alpha fit

I1-A freeze SASRec và text candidate table, sau đó chỉ tìm scalar alpha.

Arms:

- real visual;
- frequency-matched shuffle M1;
- shuffle M2;
- shuffle M3.

Alpha selection dùng validation Recall@20. Test labels không tham gia chọn alpha.

Trên comparator cũ, real visual thắng text trong cả ba seed. Tuy nhiên NDCG real-vs-shuffle chưa đạt strict gate:

| Seed | Real vs mean shuffle NDCG@10 |
|---|---:|
| 2024 | -0.785% |
| 2025 | +1.013% |
| 2026 | +2.913% |

Recall@10 lại dương trong cả ba seed:

| Seed | Real vs mean shuffle Recall@10 |
|---|---:|
| 2024 | +3.864% |
| 2025 | +7.316% |
| 2026 | +9.921% |

Kết quả này gợi ý visual contribution rõ hơn ở retrieval/Recall so với thứ hạng chính xác NDCG, nhưng chưa đủ cho claim visual-specific NDCG ổn định.

### 6.4. Patched-IEM frozen rerun

Sau khi retrain patched IEM, frozen residual được chạy lại trên comparator mới:

| Arm | Mean NDCG@10 | Mean Recall@10 |
|---|---:|---:|
| Patched text | 0.046365920 | 0.078183123 |
| Mean matched shuffle | 0.056748227 | 0.083396782 |
| Real visual | 0.057411783 | 0.090604538 |

Real so với text:

```text
NDCG@10: +23.82%
Recall@10: +15.89%
```

Real so với matched shuffle:

```text
NDCG@10: +1.17%
Recall@10: +8.64%
```

Per-seed NDCG real-minus-shuffle:

| Seed | Delta | Relative |
|---:|---:|---:|
| 2024 | +0.001709851 | +3.10% |
| 2025 | +0.000303573 | +0.51% |
| 2026 | -0.000022755 | -0.04% |

Hierarchical 95% interval:

```text
[-0.000628264, +0.001962107]
```

Kết luận strict:

```text
FAIL_FROZEN_MATCHED_ALPHA_NDCG_GATE
```

Kết quả vẫn giữ một finding hẹp:

> Real visual cải thiện Recall@10 so với matched shuffle trong patched frozen protocol, nhưng visual-specific NDCG improvement chưa khác biệt có ý nghĩa so với matched controls.

Không được retune alpha sau khi xem test result.

Nguồn: `reports/patched-i1a-frozen-alpha-report.md`.

---

## 7. Joint-trained score residual

### Method

Joint-trained path vẫn dùng:

```text
score_final = score_text + alpha * standardized_visual_similarity
```

Nhưng text adapter, SASRec và alpha được train chung trong mỗi arm.

### Kết quả patched-IEM

| Arm | Mean NDCG@10 |
|---|---:|
| Text-only | 0.046365920 |
| Mean shuffle | 0.048225738 |
| Real visual | 0.051065679 |

Real visual:

```text
+10.14% vs text
+5.89% vs mean shuffle
```

Hierarchical interval real-minus-shuffle:

```text
[+0.001227870, +0.004619737]
```

### Giới hạn diễn giải

Mean shuffle cũng thắng text:

```text
+0.001859818 NDCG@10
```

Noise floor do mỗi arm có training trajectory riêng chiếm khoảng `65.49%` headline real-minus-shuffle delta.

Alpha của real arm khoảng:

```text
0.552 / 0.575 / 0.559
```

Trong khi shuffle alpha gần `0.05`.

Do alpha và model state không được matched, kết quả này chỉ chứng minh:

> Có một joint-trained association effect dương trong protocol này.

Nó không chứng minh frozen scalar residual một cách causal.

Nguồn: `reports/patched-iem-score-residual-rerun-report.md`.

---

## 8. Hướng 4 — R1 controls: xác định residual có thật sự là visual không

R1 giữ text model và candidate table freeze, không gradient step, chỉ so sánh các direction table:

- `s-real`: CLIP visual features thật;
- `s-gauss`: Gaussian random unit vectors;
- `s-textpca`: PCA-64 từ chính title embedding table;
- `s-pop`: popularity direction;
- `s-const`: constant vector.

Kết quả aggregate historical:

| Arm | NDCG@10 relative gain |
|---|---:|
| Real visual | +18.25% |
| Gaussian | +21.44% |
| Text-PCA | **+23.78%** |
| Popularity | -11.40% |
| Constant | 0.00% |

Diễn giải:

- `s-const = 0`: không phải global score shift.
- `s-pop` thất bại: không chỉ là popularity reranking.
- `s-gauss` dương ở một protocol: residual form có thể tự tạo gain.
- `s-textpca` mạnh hơn real visual: text geometry re-projection đã đủ tạo nhiều gain.

Finding quan trọng:

> Một phần headline gain đến từ hình thức của residual — standardized similarity ở score boundary — chứ không thể mặc định quy toàn bộ gain cho visual modality.

Tuy nhiên R1 cũng phát hiện một vùng hẹp có visual-specific signal:

| Slice | Real vs text-PCA NDCG | Real vs text-PCA Recall |
|---|---:|---:|
| Cold zero-exposure | -13.89% | -10.82% |
| Warm low | -11.07% | -8.15% |
| Warm mid | -2.20% | -0.28% |
| Warm high | **+15.55%** | **+15.10%** |

Trong R1, real visual chỉ vượt cả control trên vùng `pop_warm_high`.

Cần lưu ý: R1 headline aggregate đã được phân tích lại bằng repeat-debiased protocol và không được dùng một mình để khẳng định method.

Nguồn: `reports/r1-mechanism-report.md`.

---

## 9. Hướng 5 — Exposure-gated residual

### Method

R2a quan sát rằng tác động visual thay đổi theo item training exposure. G2 thử dùng exposure/frequency để điều chỉnh alpha theo candidate:

```text
score(u, i)
    = score_text(u, i)
    + alpha(exposure_i) * z_visual(u, i)
```

### Kết quả trên SASRec G2, Games novel targets

| Arm | NDCG@10 | vs text |
|---|---:|---:|
| Text-only | 0.022756 | — |
| Constant | 0.022756 | 0.00% |
| Gaussian | 0.022109 | -2.85% |
| Text-PCA | 0.022835 | +0.35% |
| Real visual | **0.024127** | **+6.03%** |
| Gated real-low | 0.024039 | +5.64% |
| Gated hybrid | 0.024282 | +6.70% |

Plain real residual đã thắng text, Gaussian và text-PCA trong protocol G2. Exposure gate chỉ tăng thêm:

```text
+0.000154 NDCG@10 so với plain real
```

Khoảng tin cậy chứa zero và Recall@10 hơi giảm.

### Kết luận

Exposure gate không đáng thêm complexity. Method đơn giản hơn được giữ lại:

```text
s(u, i) = s_text(u, i) + alpha * z_visual(u, i)
```

Tuy nhiên G2 dựa trên comparator Games cũ. Sau patched-IEM rerun, frozen strict NDCG gate không còn pass. Vì thế G2 phải được trình bày như **conditional positive result**, không phải universal confirmation.

Nguồn: `reports/g2-exposure-gate-report.md`.

---

## 10. Hướng 6 — Cross-backbone: GRU4Rec

### Kết quả

| Backbone/protocol | Text | Real visual | Gaussian | Text-PCA |
|---|---:|---:|---:|---:|
| SASRec G2 | 0.022756 | 0.024127 | 0.022109 | 0.022835 |
| GRU4Rec | 0.000653 | 0.014008 | 0.000266 | **0.016648** |

Real visual trên GRU4Rec cao hơn text hơn `2000%`, nhưng text baseline chỉ khoảng `0.00053–0.00070`, tức gần như degenerate.

Text-PCA còn thắng real visual:

```text
real - text-PCA = -0.002640
95% CI = [-0.003554, -0.001719]
```

### Kết luận

Không thể claim backbone-independent visual method. GRU4Rec là stress test cho thấy residual có thể bù đắp một text backbone yếu, nhưng không chứng minh visual semantics.

Nguồn: `reports/cross-backbone-report.md`.

---

## 11. Hướng 7 — BERT4Rec competitive protocol

### Method

BERT4Rec được xây để tránh vấn đề GRU baseline quá yếu:

- title-conditioned;
- bias-free adapter `896 -> 128`;
- 2 bidirectional Transformer layers;
- 2 attention heads;
- inner size `512`;
- dropout `0.2`;
- dynamic Cloze masking `20%`;
- masked-position cross-entropy;
- full-catalog final-mask inference;
- text checkpoint freeze trước khi đưa visual vào.

### Competitive baseline gate

| Seed | Text-only novel NDCG@10 |
|---:|---:|
| 2024 | 0.020588 |
| 2025 | 0.019829 |
| 2026 | 0.018166 |
| Mean | **0.019528** |

Mean baseline vượt floor yêu cầu `0.018205`.

### Kết quả

| Arm | Mean novel NDCG@10 |
|---|---:|
| Text-only | 0.019528 |
| Gaussian | 0.019171 |
| Text-PCA | 0.019777 |
| Real visual | **0.020935** |
| Constant | 0.019528 |

Real visual thắng:

```text
+7.21% vs text
+0.001764 vs Gaussian
+0.001158 vs text-PCA
```

Tất cả contrast đều thắng `3/3` seeds và các hierarchical intervals đều nằm trên zero.

### Kết luận

Đây là bằng chứng tốt nhất cho hướng score residual trên Games:

> Khi text backbone đủ mạnh và evaluation được repeat-debias, frozen visual residual có thể vượt text-only, Gaussian và text-PCA trên cả SASRec và BERT4Rec.

Nhưng nó vẫn là claim có điều kiện theo dataset/protocol.

Nguồn: `reports/bert4rec-protocol-report.md`.

---

## 12. Hướng 8 — Transfer sang Sports

### Method

- frozen checkpoint-500 title table;
- không retrain CSFT/IEM;
- frozen Sports CLIP table;
- BERT4Rec transfer;
- real/shuffle/Gaussian/text-PCA controls;
- three seeds;
- novel-target NDCG@10 primary.

### Kết quả

| Seed | Text | Real visual | Shuffle |
|---:|---:|---:|---:|
| 2024 | 0.006591902 | 0.006520674 | 0.006410032 |
| 2025 | 0.007659990 | 0.007558745 | 0.007429713 |
| 2026 | 0.007017651 | 0.007239187 | 0.007000550 |

Contrast:

| Contrast | Mean delta | Verdict |
|---|---:|---|
| real - text | +0.000016355 | Fail |
| real - shuffle | +0.000159437 | Pass |
| real - Gaussian | +0.000106735 | Fail |
| real - text-PCA | -0.000038088 | Fail |

Recall@10 cũng giảm quá ngưỡng cho phép ở hai seed.

### Kết luận

Sports không xác nhận external validity. Visual residual có thể tách khỏi shuffle nhưng không thắng ổn định text-only và specificity controls.

Không được claim method dataset-independent.

Nguồn: `reports/two-dataset-transfer-report.md`.

---

## 13. Hướng 9 — C0.5 matched-shuffle reconstruction

### Mục tiêu

C0.5 không train recommender mới. Nó dùng frozen checkpoints và kiểm tra visual-specificity bằng matched shuffle:

- giữ phân phối CLIP PCA-64;
- giữ train-frequency bins;
- phá item-image association;
- align rows theo:
  ```text
  (evaluation_row_id, user_id, target_item_id, immediate_repeat, seed)
  ```
- bootstrap theo seed rồi theo source user.

### Kết quả

| Backbone | ATE real | ATE shuffle | Real - shuffle | 95% CI |
|---|---:|---:|---:|---|
| SASRec G2 | 0.001371139 | -0.000306537 | +0.001677676 | [0.001258085, 0.002196482] |
| BERT4Rec | 0.001407279 | -0.000103976 | +0.001511255 | [0.000859907, 0.002138483] |

Cả hai backbone:

- real thắng `3/3` seeds;
- interval trên zero;
- alignment bị phá trong shuffle;
- không train thêm recommender, LLM2Rec embedding model hoặc visual encoder.

Verdict:

```text
PASS_VISUAL_SPECIFICITY
```

Đây là bằng chứng visual-specific mạnh nhất trong toàn bộ chương trình. Nhưng phạm vi phải ghi chính xác:

> C0.5 chứng minh visual-specific offline contrast trong Games và các frozen checkpoint cụ thể. Nó không chứng minh transfer sang mọi dataset, không chứng minh online preference causality và không thay thế patched-IEM cascade mới.

Nguồn: `reports/c05-matched-shuffle-report.md`.

---

## 14. Hướng 10 — C1 selective visual policy

### Mục tiêu

Thay vì luôn dùng visual, học một policy chọn giữa:

```text
always text
hoặc
always visual
```

Inputs chỉ dùng feature có thể có trước khi biết held-out outcome:

- text score statistics;
- visual residual statistics;
- history features;
- exposure features;
- uncertainty features.

### Kết quả

SASRec:

| Policy | NDCG@10 |
|---|---:|
| always-text | 0.02275622 |
| always-visual | 0.02412736 |
| selected learned | 0.02412736 |

Policy học được chỉ tie với always-visual.

BERT4Rec:

| Policy | NDCG@10 |
|---|---:|
| always-text | 0.01952775 |
| always-visual | 0.02093503 |
| selected learned | 0.02062535 |

Selected learned thấp hơn always-visual `0.00030967`.

Corrected independent audit vẫn giữ verdict:

```text
STOP_C1_POLICY
```

### Ý nghĩa

Oracle headroom vẫn tồn tại, nhưng oracle biết outcome nên không deploy được. Các feature pre-treatment hiện tại chưa đủ để học một selective policy có giá trị hơn `always-visual`.

Không được nói rằng visual utility không có heterogeneity. Kết luận đúng là chưa chứng minh được heterogeneity có thể được khai thác trước outcome.

Nguồn: `reports/c1-selective-policy-report-corrected.md`.

---

## 15. Hướng tiềm năng nhất hiện tại: frozen score-level visual residual

### 15.1. Vì sao hướng này tốt hơn các hướng khác?

#### So với additive item fusion

Additive item fusion làm thay đổi toàn bộ candidate geometry. Frozen score residual chỉ thay đổi scalar score cuối cùng và giữ text candidate table cố định.

#### So với sequence-side fusion

Sequence-side fusion có nguy cơ học repeat/copy shortcut và làm thay đổi user state. Score residual dễ audit hơn vì text user state giữ nguyên.

#### So với joint training

Joint training tạo confounding giữa visual information và training trajectory. Frozen alpha fit tách được visual direction khỏi optimizer state.

#### So với exposure gate

Exposure gate thêm complexity nhưng không cải thiện đáng kể plain residual. Một scalar global dễ giải thích, dễ benchmark và dễ kiểm soát.

#### So với selective policy

Selective policy hiện chưa thắng always-visual. Score residual vẫn có bằng chứng identification tốt hơn policy exploitation.

### 15.2. Method đề xuất để mô tả với giáo viên

Input:

- title embedding từ LLM2Rec;
- CLIP item image embedding;
- user history;
- item training exposure chỉ dùng trong diagnostic/control hoặc một protocol được định nghĩa rõ.

Text path:

```text
history text embeddings
        -> SASRec/BERT4Rec
        -> text user representation
        -> text candidate scores
```

Visual path:

```text
history image embeddings
        -> recency-weighted visual user profile
        -> cosine với CLIP candidate embedding
        -> standardize theo user/candidate score distribution
        -> visual residual z_visual
```

Fusion:

```text
final_score(u, i)
    = text_score(u, i)
    + alpha * z_visual(u, i)
```

Training/evaluation protocol:

1. train LLM2Rec text pipeline;
2. freeze title embedding table;
3. train text recommender;
4. freeze recommender và text candidate table;
5. compute CLIP feature table;
6. compute visual history profile;
7. chọn alpha trên validation, không dùng test labels;
8. chạy real visual và frequency-matched shuffle;
9. giữ constant control để kiểm tra null parity;
10. report novel-target metrics riêng với immediate-repeat metrics;
11. dùng seed-then-user-chain bootstrap;
12. chỉ kết luận visual-specific nếu real thắng shuffle/control theo gate đã định trước.

### 15.3. Điểm mới có thể bảo vệ về mặt nghiên cứu

Không nên nói novelty là “thêm hình ảnh vào recommender”, vì hướng đó đã phổ biến.

Đóng góp có thể bảo vệ hơn gồm:

1. **Score-boundary insertion:** visual được đưa vào tại ranking boundary để bảo toàn text baseline.
2. **Repeat-debiased evaluation:** tách immediate-repeat khỏi novel-item recommendation.
3. **Matched non-semantic controls:** frequency-matched shuffle, Gaussian và text-PCA để phân biệt visual semantics với residual calibration.
4. **Backbone qualification:** visual-specific effect chỉ đáng tin khi text baseline đủ mạnh.
5. **Negative evidence:** additive item fusion và sequence-side fusion có thể tạo gain giả do representation drift hoặc repeat copying.
6. **Scoped visual utility:** evidence cho thấy visual effect phụ thuộc benchmark/backbone/protocol, không nên claim universal multimodal improvement.

### 15.4. Cách diễn giải kết quả hiện tại

Cách nói quá mạnh, không nên dùng:

> CLIP visual features luôn cải thiện LLM2Rec.

> Method visual residual generalizes across datasets.

> Visual features giải quyết cold-start.

> Learned gate đã biết khi nào cần dùng visual.

Cách nói chính xác hơn:

> Trong Games_5core, với frozen title-conditioned SASRec/BERT4Rec và full-catalog novel-target evaluation, score-level visual residual cho thấy aligned image association có giá trị offline dương so với matched shuffle trong một số frozen comparator protocols. Tuy nhiên kết quả không transfer rõ sang Sports, không ổn định trên backbone yếu và chưa tạo được deployable selective policy.

---

## 16. Các phát hiện khoa học phụ quan trọng

### 16.1. Immediate-repeat confound

Test split có tỷ lệ immediate repeat cao hơn train:

| Split | Immediate-repeat share |
|---|---:|
| Train | 3.02% |
| Validation | 3.77% |
| Test | 4.83% |

Một method có khả năng copy last item có thể tăng aggregate NDCG mà không tăng novel-item recommendation.

### 16.2. Residual form và modality là hai yếu tố khác nhau

Text-PCA đôi khi mạnh hơn visual. Vì vậy cần tách:

```text
gain do residual form
vs
 gain do semantic visual information
```

Nếu không có text-PCA/Gaussian control, không thể biết gain thật sự đến từ ảnh.

### 16.3. Backbone quality là điều kiện quan trọng

GRU4Rec text baseline quá yếu. Một residual bất kỳ cũng có thể tạo large relative gain. BERT4Rec có baseline cạnh tranh hơn và là kiểm tra có giá trị hơn.

### 16.4. Identification khác exploitation

- Identification: visual có thêm information không?
- Characterization: visual giúp trong điều kiện nào?
- Exploitation: có thể dự đoán trước khi nào visual sẽ giúp không?

Hiện tại:

- identification: có bằng chứng hẹp trên Games;
- characterization: có dấu hiệu exposure-dependent nhưng cần protocol sạch hơn;
- exploitation: C1 fail.

---

## 17. Final verdict

### Những gì đã được chứng minh tương đối tốt

1. Có thể tái dựng LLM2Rec end-to-end dưới compatibility profile.
2. Có thể kiểm tra exact null parity ở score boundary.
3. Additive item fusion v9 không an toàn và đã bị bác bỏ.
4. Sequence-side gain ban đầu chủ yếu do immediate-repeat/copy behavior.
5. Repeat-debiased evaluation là bắt buộc.
6. BERT4Rec là competitive second backbone tốt hơn GRU4Rec.
7. C0.5 cho visual-specific positive contrast trên Games với SASRec và BERT4Rec.
8. Score-level residual là insertion point đáng nghiên cứu nhất.

### Những gì chưa được chứng minh

1. Strict reproduction đúng toàn bộ paper LLM2Rec.
2. Visual residual generalizes sang Sports.
3. Visual residual luôn thắng text-PCA trên mọi comparator.
4. Visual utility có quan hệ nhân quả với user preference thực tế.
5. Learned policy có thể chọn đúng khi nào nên dùng visual.
6. Visual features giải quyết cold-start nói chung.
7. Method đạt production/online value.

### Trạng thái cuối

```text
STOP2 — Method Exhausted
```

Không nên tiếp tục thêm architecture fusion mới trong cùng protocol Games-only. Kết quả hiện tại nên được đóng gói theo hướng:

> **Evidence-first evaluation of LLM2Rec visual residuals: baseline preservation, repeat-debiased ranking, matched controls and conditional visual specificity.**

Đây là framing mạnh và trung thực hơn một claim chung rằng “thêm ảnh giúp recommender tốt hơn”.

---

## 18. Một đoạn tóm tắt ngắn để nói trực tiếp với giáo viên

> Em đã tái dựng pipeline LLM2Rec bằng Qwen2-0.5B. Đầu tiên model được CSFT trên AmazonMix-6, sau đó train MNTP và SimCSE để tạo title embedding cho item. Embedding này được đưa vào các sequential recommender như SASRec, GRU4Rec và BERT4Rec. Do giới hạn Kaggle T4, kết quả của em là compatibility-profile chứ không phải reproduction tuyệt đối paper.
>
> Sau đó em thử nhiều cách đưa ảnh CLIP vào recommender. Cộng ảnh trực tiếp vào item embedding làm hỏng text geometry. Đưa ảnh vào sequence representation ban đầu có gain, nhưng audit cho thấy gain chủ yếu do immediate-repeat copying. Vì vậy em chuyển sang score-level residual: giữ nguyên text model và candidate table, chỉ cộng một visual similarity residual đã chuẩn hóa vào score cuối cùng. Công thức là `score_final = score_text + alpha * z_visual`, trong đó alpha chỉ chọn trên validation.
>
> Đây là hướng tốt nhất vì nó bảo toàn text baseline tại alpha bằng không và cho phép dùng matched image shuffle để kiểm tra visual-specificity. Trên Games, với SASRec và BERT4Rec, real image alignment thắng frequency-matched shuffle trong C0.5 audit. Tuy nhiên kết quả chưa tổng quát: Sports transfer không vượt qua toàn bộ control gate, GRU4Rec có text baseline quá yếu, patched-IEM frozen rerun chỉ tăng Recall chứ chưa đạt strict NDCG gate, và learned policy chưa thắng always-visual. Vì vậy kết luận hiện tại là visual score residual có tiềm năng offline có điều kiện trên Games, nhưng chưa thể gọi là multimodal method tổng quát.

---

## 19. Source map

### Text reproduction

- `plans/260820-1454-llm2rec-text-reproduction-kaggle/plan.md`
- `plans/reports/report-260909-llm2rec-text-baseline-fidelity-audit.md`
- `plans/reports/report-260909-llm2rec-mask-arm-measurement.md`

### Visual fusion

- `plans/reports/report-2026-08-25-llm2rec-visual-signal-v9.md`
- `plans/260827-1532-llm2rec-baseline-preserving-visual-fusion/plan.md`
- `plans/260827-1532-llm2rec-baseline-preserving-visual-fusion/reports/g0-parity-audit.md`
- `plans/260827-1532-llm2rec-baseline-preserving-visual-fusion/reports/g1-visual-screen-report.md`
- `plans/260827-1532-llm2rec-baseline-preserving-visual-fusion/reports/i1-score-residual-report.md`
- `plans/260827-1532-llm2rec-baseline-preserving-visual-fusion/reports/patched-i1a-frozen-alpha-report.md`
- `plans/260827-1532-llm2rec-baseline-preserving-visual-fusion/reports/patched-iem-score-residual-rerun-report.md`
- `plans/260827-1532-llm2rec-baseline-preserving-visual-fusion/reports/r1-mechanism-report.md`
- `plans/260827-1532-llm2rec-baseline-preserving-visual-fusion/reports/g2-exposure-gate-report.md`
- `plans/260827-1532-llm2rec-baseline-preserving-visual-fusion/reports/bert4rec-protocol-report.md`
- `plans/260827-1532-llm2rec-baseline-preserving-visual-fusion/reports/cross-backbone-report.md`
- `plans/260827-1532-llm2rec-baseline-preserving-visual-fusion/reports/two-dataset-transfer-report.md`
- `plans/260827-1532-llm2rec-baseline-preserving-visual-fusion/reports/c05-matched-shuffle-report.md`
- `plans/260827-1532-llm2rec-baseline-preserving-visual-fusion/reports/c1-selective-policy-report-corrected.md`

### Sequence-side and repeat audit

- `plans/260827-1532-llm2rec-baseline-preserving-visual-fusion/reports/s1-sequence-side-deep-report.md`
- `plans/260827-1532-llm2rec-baseline-preserving-visual-fusion/reports/s2-repeat-debiased-report.md`
- `plans/260827-1532-llm2rec-baseline-preserving-visual-fusion/reports/r2a-recall-scope-report.md`

---

**Final status:** this document is a synthesis report for discussion with the teacher. It does not create a new experiment, change any baseline designation, or promote the visual residual to a universal method claim.
