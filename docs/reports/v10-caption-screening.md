# Hướng v10 — Sàng lọc caption theo domain, không tốn GPU (hướng A)

> Trạng thái: **ĐÃ ĐÓNG (2026-09-30), kết quả âm.** Sàng lọc theo domain cho `PROXY_INVALID`; thử ghép muộn (mục 7) cho `FAIL`. Hướng caption v10 dừng tại đây.
> Giao thức và quy tắc quyết định được ghi **trước** khi tính số của các domain khác Video_Games: `research/caption_augmentation/screening/protocol.md`.

## 1. Câu hỏi

Ở mỗi domain của AmazonMix-6, caption ảnh có mang thông tin về item được mua tiếp theo, **ngoài những gì title đã nói**, hay không? Nếu có, domain nào đáng bỏ GPU để chạy pilot paired?

## 2. Cách đo

- **Dữ liệu:** 960,757 transition train và 120,099 transition test. Kernel CPU `trixuanle/llm2rec-caption-screening-pairs-v1` xuất các cặp này. Mọi file đều có SHA-256; không có target nào lệch title. Caption lấy từ Florence-2 v10 (108,753 item); riêng Video_Games đo thêm bằng caption Qwen3-VL. Ghép theo ASIN: 108,733/108,733 item khớp, ID CSV trùng với global ID.
- **Chỉ số chính M:**
  - So sánh cặp thật (item trước → item mua tiếp) với 5 item âm. Item âm được ghép cho cùng nhóm popularity và cùng nhóm độ dài caption với item thật.
  - Chỉ xét các bộ ba mà **title không phân biệt được** (độ giống title bằng nhau).
  - Độ giống caption tính trên phần caption đã **bỏ mọi từ có trong hai title**.
  - AUC = 0.5 nghĩa là không có tín hiệu.
- **M_cf (amendment 1, đăng ký sau khi calibrate Games):** giống M, nhưng dùng cặp của tập **test**, và chỉ giữ các bộ ba mà cả title lẫn đồng xuất hiện trong train (CF) đều không phân biệt được.
- **Kiểm soát:**
  - G1: caption bị xáo giữa các item phải cho ≈ 0.5.
  - G2: title phải có tín hiệu, AUC ≥ 0.55.
  - G3: Video_Games (đã biết là không có lợi khi train) không được nằm trong top 2.
  - C_cf: M_cf của Games tính bằng caption Qwen3-VL phải ≤ 0.52.

## 3. Kết quả (caption Florence-2)

| Domain | AUC title | M | M 95% CI | Shuffle | M_cf |
|---|---|---|---|---|---|
| Video_Games | 0.633 | 0.519 | [0.518, 0.520] | 0.501 | 0.519 |
| Movies_and_TV | 0.593 | 0.517 | [0.516, 0.519] | 0.500 | 0.519 |
| Arts_Crafts_and_Sewing | 0.638 | 0.510 | [0.509, 0.511] | 0.503 | 0.505 |
| Electronics | 0.572 | 0.505 | [0.504, 0.506] | 0.501 | 0.504 |
| Tools_and_Home_Improvement | 0.561 | 0.504 | [0.504, 0.505] | 0.500 | 0.505 |
| Home_and_Kitchen | **0.542** | 0.503 | [0.502, 0.503] | 0.500 | 0.503 |

Calibrate Video_Games bằng caption **Qwen3-VL** (loại caption đã dùng để train): M = 0.580, M_cf = 0.575.

**Gate:**

| Gate | M | M_cf | Lý do |
|---|---|---|---|
| G1 shuffle ≈ 0.5 | FAIL | PASS | Arts 0.503 [0.501, 0.504]: CI không chứa 0.5 |
| G2 title ≥ 0.55 | FAIL | FAIL | Home_and_Kitchen 0.542 |
| G3 Games không top 2 | FAIL | FAIL | Games đứng thứ 1 theo M, thứ 2 theo M_cf |
| C_cf ≤ 0.52 | — | FAIL | 0.575 |

Theo quy tắc đã đăng ký, cả hai chỉ số đều cho `PROXY_INVALID`. Quan trọng nhất là G3 và C_cf: chỉ số xếp Video_Games cao nhất, mà đó lại là domain train ra kết quả âm. Nghĩa là "caption có thêm tín hiệu" **không dự đoán được** "caption giúp khi train".

## 4. Phát hiện phụ: thông tin có sẵn, nhưng mô hình không dùng được

Trên Video_Games, lấy các cặp test và so với hai embedding đã train của pilot (`screening/games_baseline_conditioned.py`, `results/caption_screening/games_baseline_conditioned.json`):

| Tập bộ ba | Số bộ ba | Caption residual (Qwen3-VL) | Embedding title-only | Embedding `real` |
|---|---|---|---|---|
| Tất cả | 65,765 | 0.610 | 0.607 | 0.582 |
| Title-only đoán **sai** | 25,875 | **0.554** | 0.000 | **0.490** |
| Title-only đoán đúng | 39,890 | 0.647 | 1.000 | 0.641 |

Ở những chỗ baseline sai, caption vẫn phân biệt được item thật (0.554). Embedding `real` được train với chính caption đó nhưng không khai thác được (0.490, gần như ngẫu nhiên). Vậy trên Games, điểm nghẽn là **cách đưa caption vào LLM2Rec**, không phải việc caption thiếu thông tin. *(Suy luận: lý do có thể là CSFT chỉ dạy sinh title, còn IEM ép toàn bộ văn bản vào một vector bằng mean pooling. Chưa kiểm chứng riêng.)*

## 5. Kết luận và hướng tiếp theo

- **Không đổi domain để cứu hướng caption.** Không có bằng chứng tin cậy rằng domain khác tốt hơn Video_Games. Chỉ số lexical không dự đoán được lợi ích khi train.
- **Đóng góp khả thi hơn:** một kết quả âm có kiểm soát (Mục 3), cộng với bằng chứng ở Mục 4 rằng caption có tín hiệu bổ sung mà LLM2Rec không hấp thụ được.
- **Nếu muốn tiếp tục, hãy đổi cách tích hợp chứ không đổi dataset:** ví dụ đưa đặc trưng caption thành một kênh riêng, hoặc late fusion với embedding title. Có thể kiểm tra rẻ trên Games bằng các embedding hiện có trước khi train lại. Việc này chưa được lên kế hoạch.

## 6. Giới hạn

- Chỉ số dựa trên trùng từ (lexical), nên có thể bỏ sót tín hiệu ngữ nghĩa.
- Ở 5 domain, Florence caption ngắn hơn và khác phong cách Qwen3-VL. Chỉ Games có Qwen3-VL (M: 0.519 với Florence so với 0.580 với Qwen3-VL).
- M_cf là amendment đăng ký sau khi calibrate Games, nên chỉ mang tính thăm dò.
- Chênh lệch giữa các domain rất nhỏ (≤ 0.016 AUC); CI hẹp chủ yếu nhờ số cặp rất lớn.

## 7. Thử ghép muộn (late fusion) — amendment 2 và 3

**Bước 1, không tốn GPU (amendment 2, đạt).** Trên các cặp test của Games, điểm `cos(e_title) + λ·(độ giống caption)` cho AUC 0.640, so với 0.608 của riêng title: +0.032, CI [0.027, 0.036]. Caption bị xáo chỉ thêm +0.0005. λ = 2.0 được chọn trên tập valid và nằm đúng mép lưới. Kết quả: `results/caption_screening/fusion_headroom.json`.

**Bước 2, GPU (amendment 3, FAIL).**
- Kernel `trlxun/llm2rec-vg-late-fusion-v1` chạy trên account thứ hai, dùng dataset private `trlxun/llm2rec-vg-late-fusion-inputs-v1`; mọi input đều được pin SHA.
- GPU Tesla T4, 3,307 s ≈ 0.92 GPU-h. Tính cả lần này, toàn dự án đã dùng khoảng 31.7 GPU-h trên hai account.
- Ma trận đặc trưng cố định:
  - `[e_title ; s·e_caption]`, trong đó `e_caption` là TF-IDF của phần caption khác title, giảm về 256 chiều bằng SVD không giám sát, và `s` bằng norm trung bình của `e_title`.
  - Đặc trưng được đưa vào adapter tuyến tính sẵn có của SASRec.

| Arm | Recall@10 (2024 / 2025 / 2026) | Recall@10 TB | NDCG@10 TB |
|---|---|---|---|
| `title` (tái lập pilot, lệch 0.0) | 0.0837 / 0.0829 / 0.0846 | 0.0838 | 0.0499 |
| `fused_real` | 0.0832 / 0.0739 / 0.0705 | 0.0759 | 0.0433 |
| `fused_shuffle` | 0.0679 / 0.0666 / 0.0734 | 0.0693 | 0.0394 |

Kết luận theo quy tắc: real thắng shuffle ở 2/3 seed (yêu cầu 3/3), và real chỉ đạt 0.905 lần baseline (yêu cầu ≥ 1.02). Verdict: **FAIL**.

**Đọc kết quả.**
- Thêm một khối 256 chiều vào đầu vào adapter làm SASRec kém đi, kể cả khi khối đó là nhiễu: shuffle giảm 17%.
- Caption thật lấy lại được một phần so với nhiễu (+0.0066 Recall@10 trung bình), nhưng vẫn thấp hơn baseline 9.5%.
- Như vậy tín hiệu thấy ở bước 1 **không chuyển thành lợi ích** khi dùng cách ghép này.
- Sai lệch so với thiết kế ban đầu: tôi dùng adapter tuyến tính sẵn có để không phải sửa code LLM2Rec, nên cổng `g` **không** được khởi tạo bằng 0. Ngay từ đầu, khối caption đã có cùng độ lớn với khối title. *(Suy luận: đây có thể là nguyên nhân chính làm shuffle tụt 17%.)*

Kết quả: `results/caption_screening/late_fusion_v1/late_fusion_artifact.json`.
