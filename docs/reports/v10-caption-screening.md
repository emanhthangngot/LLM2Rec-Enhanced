# Hướng v10 — Sàng lọc caption theo domain, không tốn GPU (hướng A)

> Trạng thái: **đã đo; theo quy tắc đăng ký trước, kết quả là `PROXY_INVALID`**. Không domain nào được chọn để chạy paired GPU.
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
