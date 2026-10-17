# Hướng dẫn và báo cáo nghiên cứu bóc tách phạm vi đầu vào (Task M03A)

Báo cáo này giải quyết **câu hỏi nghiên cứu RQ3** trong tài liệu dự án: *"Title-only và Title+Content khác nhau thế nào đối với bài toán phát hiện chủ đề tin tức tiếng Việt?"*.

---

## 1. Thiết lập thực nghiệm kiểm soát biến

Để đo lường chính xác giá trị gia tăng của phần nội dung bài báo mà không bị nhiễu bởi các yếu tố khác, thực nghiệm áp dụng nguyên tắc kiểm soát biến duy nhất:
- **Tập dữ liệu nền:** Cố định biến thể tiền xử lý tối ưu nhất từ thực nghiệm M03 là **P3** (`data/processed/p3_no_stopwords.csv`), nơi văn bản đã được tách từ ghép tiếng Việt bằng PyVi và loại bỏ 388 từ dừng.
- **Phân chia dữ liệu:** Cùng tập train (6.493 bài) và validation (1.391 bài) từ `data/splits/split_manifest.csv`. Toàn bộ vectorizer và mô hình chỉ fit trên train, đánh giá độc lập trên validation.
- **Cố định siêu tham số TF-IDF:** `ngram_range = (1, 1)`, `min_df = 2`, `max_df = 0.85`, `max_features = 20.000`, `sublinear_tf = True`, `norm = 'l2'`, `token_pattern = r"(?u)\b\w\w+\b"`, giữ `lowercase: false` và `stopwords_file: null`.
- **Cố định thuật toán phân cụm:** K-Means với $K=9$ làm mốc khảo sát chính và $K=7$ làm mốc đối chiếu độ nhạy, đo độ ổn định qua 5 seed ngẫu nhiên (42, 43, 44, 45, 46).
- **Biến độc lập duy nhất:** Phạm vi văn bản đầu vào:
  1. **Chỉ tiêu đề (title_only):** Sử dụng cột `processed_title` (độ dài trung bình 9.98 từ/bài).
  2. **Tiêu đề và nội dung (title_content):** Sử dụng cột `text` (độ dài trung bình 466.6 từ/bài).

---

## 2. Bảng tổng hợp kết quả định lượng trên tập validation ($K = 9$)

Toàn bộ số liệu được lưu trữ độc lập tại `results/kmeans/m03a_title_vs_content_metrics.csv`.

| Chỉ số đo lường | Chỉ tiêu đề (`title_only`) | Tiêu đề và nội dung (`title_content`) | Mức chênh lệch | Đánh giá xu hướng |
|---|---|---|---|---|
| **Kích thước từ vựng (train)** | 4.437 từ | 20.000 từ | +15.563 từ (+350.8%) | Nội dung làm giàu từ vựng vượt bậc |
| **Độ thưa ma trận validation** | 99.83% | 99.05% | -0.78% | Tiêu đề bị thưa thớt cực độ |
| **Tỷ lệ từ chưa biết (OOV)** | 10.45% | 2.56% | -7.89% (giảm 4 lần) | Tiêu đề gặp rủi ro OOV rất cao |
| **Số từ hoạt động trung bình/bài** | 7.7 từ | 190.7 từ | +183.0 từ | Lượng thông tin tăng gấp 25 lần |
| **Số bài vector rỗng (zero-vector)** | 0 bài | 0 bài | 0 | Đảm bảo 100% bài có tín hiệu |
| **Silhouette Score** $\uparrow$ | +0.0063 | **+0.0190** | +0.0127 | Toàn bài tách biệt rõ hơn |
| **Davies-Bouldin Index** $\downarrow$ | **5.8107** | 6.1566 | +0.3459 | Nghịch lý chỉ số nội tại tái diễn |
| **Calinski-Harabasz Score** $\uparrow$ | 8.87 | **11.38** | +2.51 | Toàn bài phân tán cụm tốt hơn |
| **Purity** $\uparrow$ | 30.77% | **87.28%** | **+56.51%** | Toàn bài vượt trội áp đảo |
| **NMI** $\uparrow$ | 0.1637 | **0.7004** | **+0.5367** | Tương đồng thông tin tăng vọt |
| **ARI** $\uparrow$ | -0.0016 | **0.5925** | **+0.5941** | Tiêu đề tương đương đoán ngẫu nhiên |
| **Độ ổn định đa seed (ARI)** $\uparrow$ | 0.5496 | **0.8211** | +0.2715 | Toàn bài hội tụ rất ổn định |
| **Tỷ lệ từ ghép tâm cụm** $\uparrow$ | 17.04% | **43.70%** | +26.66% | Từ khóa toàn bài giàu nghĩa chuyên sâu |
| **Tỷ lệ nhiễu từ dừng** $\downarrow$ | **0.0%** | **0.0%** | 0.0% | Kế thừa sạch sẽ từ P3 |
| **Đa dạng từ khóa top-10** $\uparrow$ | 0.7556 | **0.9556** | +0.2000 | Các cụm toàn bài ít trùng lặp từ |
| **Tổng thời gian chạy** $\downarrow$ | **8.12 giây** | 22.45 giây | +14.33 giây | Tiêu đề chạy nhanh hơn 2.8 lần |

---

## 3. Phân tích hiện tượng và phát hiện chuyên sâu

### 3.1. Hiện tượng sụp đổ cụm ở biến thể chỉ tiêu đề (Cluster Collapse)
Phát hiện thực nghiệm quan trọng nhất là sự **sụp đổ hoàn toàn về cấu trúc phân cụm** khi chỉ dựa vào tiêu đề bài báo:
- **Tình trạng phân bổ bài viết:** Trong 1.391 bài báo validation, biến thể chỉ tiêu đề dồn tới **982 bài (chiếm 70.6%) vào duy nhất Cụm 2**. Tám cụm còn lại chỉ nhận số lượng bài ít ỏi, biến Cụm 2 thành một cụm khổng lồ chứa lẫn lộn mọi chủ đề.
- **Bản chất toán học:** Tiêu đề tin tức rất ngắn (trung bình 10 từ). Với độ thưa ma trận lên đến 99.83%, đa số các vector bài viết rơi vào trạng thái trực giao với nhau (tích vô hướng bằng 0) do không có từ vựng giao thoa. Thuật toán K-Means trong không gian khoảng cách Euclid không tìm đủ lực kéo để phân bổ bài viết vào các tâm cụm chuyên biệt, dẫn đến việc phần lớn các điểm dữ liệu bị rơi vào một vùng trũng trung tâm.
- **Hậu quả phân loại:** Điểm Purity rơi tự do xuống **30.77%** và ARI tụt xuống **-0.0016**. Điểm ARI xấp xỉ 0 chứng minh việc phân cụm bằng tiêu đề đơn lẻ trên mô hình túi từ (Bag-of-Words) không đem lại kết quả tốt hơn việc gán nhãn ngẫu nhiên.

### 3.2. Nghịch lý chỉ số nội tại tiếp tục được khẳng định
Tương tự như phát hiện ở thực nghiệm M03:
- Biến thể chỉ tiêu đề có chỉ số Davies-Bouldin tốt hơn (5.8107 so với 6.1566 của toàn bài).
- Nếu một người làm khoa học chỉ nhìn vào chỉ số nội tại hình học mà không đối chiếu ngoại vi, họ có thể đưa ra kết luận sai lầm rằng *"chỉ dùng tiêu đề gom cụm chặt chẽ hơn dùng toàn bài"*.
- Trên thực tế, sự chặt chẽ này chỉ là ảo giác hình học: do vector tiêu đề chứa rất ít chiều khác 0, các điểm dữ liệu nằm co cụm quanh gốc tọa độ. Khi bổ sung toàn bộ nội dung bài viết, không gian đặc trưng mở rộng ra 20.000 chiều ngữ nghĩa thực thụ, làm cụm nở rộng hơn về mặt hình học nhưng phân tách chính xác tuyệt đối các chủ đề với Purity đạt **87.28%**.

### 3.3. Phân tích chi tiết mức độ tổn thương theo 7 chuyên mục
Dữ liệu trích xuất từ `results/kmeans/m03a_category_breakdown.csv` cho thấy sự khác biệt rõ rệt giữa hai biến thể:

| Chuyên mục gốc | Số bài validation | Độ thuần khiết cụm (`title_only`) | Độ thuần khiết cụm (`title_content`) | Mức cải thiện khi có nội dung |
|---|---|---|---|---|
| **Giáo dục** | 156 bài | 13.24% | **97.67%** (Cụm 7) | **+84.43%** |
| **Sức khỏe** | 185 bài | 16.19% | **98.74%** (Cụm 3) | **+82.55%** |
| **Khoa học công nghệ** | 105 bài | 7.03% | **92.11%** (Cụm 2) | **+85.08%** |
| **Pháp luật** | 177 bài | 14.66% | **87.23%** (Cụm 5) | **+72.57%** |
| **Thể thao** | 293 bài | 14.26% | **100.00%** (Cụm 4) | **+85.74%** |
| **Kinh doanh** | 297 bài | 17.92% | **74.67%** (Cụm 6) | **+56.75%** |
| **Giải trí** | 178 bài | 16.70% | **68.13%** (Cụm 0) | **+51.43%** |

Ở biến thể `title_only`, cả 7 chuyên mục đều có cụm chiếm ưu thế là Cụm 2 với độ thuần khiết lẹt đẹt chỉ từ 7% đến 18%. Ngược lại, khi có toàn bài, mỗi chuyên mục tự tìm thấy một cụm riêng biệt với độ thuần khiết gần như tuyệt đối (Thể thao đạt 100%, Sức khỏe đạt 98.7%, Giáo dục đạt 97.7%).

---

## 4. Kiểm tra độ nhạy với mốc đối chiếu $K = 7$

Bảng đối chiếu từ `results/kmeans/m03a_title_vs_content_k7_sensitivity.csv` khẳng định kết luận không phụ thuộc vào việc chọn $K=7$ hay $K=9$:

| Biến thể đầu vào | Số cụm $K$ | Purity | NMI | ARI | Davies-Bouldin | Độ ổn định đa seed |
|---|---|---|---|---|---|---|
| **Chỉ tiêu đề** | $K=7$ | 29.55% | 0.1047 | -0.0046 | 6.7098 | 0.5983 |
| **Chỉ tiêu đề** | $K=9$ | 30.77% | 0.1637 | -0.0016 | 5.8107 | 0.5496 |
| **Tiêu đề và nội dung** | $K=7$ | 82.10% | 0.6889 | 0.6239 | 6.6178 | 0.8277 |
| **Tiêu đề và nội dung** | $K=9$ | **87.28%** | **0.7004** | **0.5925** | **6.1566** | **0.8211** |

Khoảng cách hiệu năng giữa hai biến thể luôn duy trì ở mức chênh lệch khổng lồ (~55% về Purity và ~0.55 về NMI/ARI) bất kể số cụm $K$.

---

## 5. Trả lời câu hỏi nghiên cứu RQ3

Dựa trên các bằng chứng định lượng và định tính:
1. **Tiêu đề đơn lẻ không đủ năng lực đại diện cho chủ đề tin tức trong không gian TF-IDF:** Do hạn chế về độ dài (khoảng 10 từ), các tiêu đề không cung cấp đủ tần suất từ khóa để thuật toán hình học K-Means thiết lập ranh giới phân tách. Mô hình bị sụp đổ cụm, gom phần lớn tài liệu vào một cụm rác duy nhất.
2. **Nội dung bài viết là điều kiện tiên quyết cho mô hình truyền thống:** Phần nội dung giúp tăng số từ hoạt động trung bình từ 7.7 lên 190.7 từ/bài, hạ tỷ lệ OOV từ 10.45% xuống 2.56%, và nâng độ chính xác phân cụm từ 30.77% lên 87.28%.
3. **Đánh đổi về chi phí tính toán là hoàn toàn xứng đáng:** Dù biến thể toàn bài tiêu tốn thời gian gấp khoảng 2.8 lần biến thể tiêu đề (22.5 giây so với 8.1 giây), mức tăng vượt bậc về chất lượng phân cụm (+56.5% Purity) khẳng định việc đưa toàn bộ nội dung vào huấn luyện là bắt buộc đối với phương pháp phân cụm K-Means.

---

## 6. Danh mục tài nguyên và hướng dẫn tái hiện

Toàn bộ thực nghiệm được đóng gói hoàn chỉnh:
- **Cấu hình:** `configs/m03a_title_vs_content.yaml`
- **Mã nguồn thực nghiệm:** `experiments/kmeans/run_m03a_title_vs_content.py`
- **Bộ kiểm thử tự động:** `tests/test_m03a_title_vs_content.py`
- **Sổ tay tương tác:** `notebooks/07_m03a_title_vs_content.ipynb`
- **Biểu đồ nghiên cứu (300 DPI):**
  - `results/kmeans/figures/m03a_metrics_comparison.png`
  - `results/kmeans/figures/m03a_vocab_and_runtime.png`
  - `results/kmeans/figures/m03a_category_purity_comparison.png`
  - `results/kmeans/figures/m03a_k7_vs_k9_sensitivity.png`

Lệnh chạy tái hiện:
```powershell
python experiments/kmeans/run_m03a_title_vs_content.py
pytest tests/test_m03a_title_vs_content.py -v
```

