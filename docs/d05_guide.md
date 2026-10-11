# Hướng dẫn Sử dụng Split IDs Task D05 Guide
---

## 1. Kết quả Phân chia Tập Dữ liệu

Tập dữ liệu đã làm sạch và khử trùng lặp (**9,274 bài viết**) được chia tách thành 3 tập độc lập:

| Tập dữ liệu | File ID | Số lượng bài | Tỷ lệ (%) | Mục đích sử dụng |
| :--- | :--- | :--- | :--- | :--- |
| **Train (Huấn luyện)** | `data/splits/train_ids.csv` | **6,493** | **70.01%** | Huấn luyện mô hình chủ đề, trích xuất từ vựng, học vector nhúng (embeddings). |
| **Validation (Kiểm định)** | `data/splits/validation_ids.csv` | **1,391** | **15.00%** | **Tinh chỉnh tham số, chọn số cụm / số chủ đề $K$**, chọn ngưỡng phân cụm, so sánh mô hình sơ bộ. |
| **Test (Kiểm thử)** | `data/splits/test_ids.csv` | **1,390** | **14.99%** | **Đánh giá hiệu năng cuối cùng (Final Benchmark)**. ĐƯỢC KHÓA CHẶT CHẼ. |
| **Tổng cộng** | `data/splits/split_manifest.csv` | **9,274** | **100.0%** | Bảng chỉ mục tổng hợp đầy đủ metadata và cụm sự kiện. |

---

## 2. Đoạn Mã Mẫu Nạp Dữ liệu Chuẩn

Vì `data/splits/` chỉ lưu `article_id` để tối ưu hóa dung lượng lưu trữ, các bạn hãy sử dụng đoạn mã Python chuẩn sau để nạp dữ liệu vào mô hình của mình:

```python
from pathlib import Path
import pandas as pd

# 1. Đọc danh sách IDs của từng tập
train_ids = pd.read_csv("data/splits/train_ids.csv")["article_id"]
val_ids = pd.read_csv("data/splits/validation_ids.csv")["article_id"]
test_ids = pd.read_csv("data/splits/test_ids.csv")["article_id"]

# 2. Đọc biến thể tiền xử lý phù hợp với mô hình của bạn
# Ví dụ: LDA / NMF dùng p2_segmented hoặc p3_no_stopwords
# Ví dụ: BERTopic / PhoBERT dùng p1_normalized hoặc p0_minimal
df_variant = pd.read_csv("data/processed/p2_segmented.csv")

# 3. Lọc dữ liệu theo split IDs
df_train = df_variant[df_variant["article_id"].isin(train_ids)].reset_index(drop=True)
df_val = df_variant[df_variant["article_id"].isin(val_ids)].reset_index(drop=True)
df_test = df_variant[df_variant["article_id"].isin(test_ids)].reset_index(drop=True)

print(f"Train: {len(df_train)} bài | Val: {len(df_val)} bài | Test: {len(df_test)} bài")

# 4. Trích xuất văn bản đầu vào cho mô hình (cột 'text')
X_train = df_train["text"].tolist()
y_train = df_train["category_gold"].tolist()  # (nếu cần nhãn tham chiếu)
```

---

## 3. Chính sách khóa tập Test

Để đảm bảo tính khoa học, độ tin cậy và không bị học vẹt khi báo cáo đồ án, toàn bộ thành viên bắt buộc phải tuân thủ quy trình 3 bước:

```
[Tập TRAIN] ──────> Huấn luyện Mô hình ban đầu
                           │
                           ▼
[Tập VALIDATION] ──> Tinh chỉnh Siêu tham số & Chọn số cụm K
                     (Coherence Score Cv, Silhouette, Davies-Bouldin)
                           │
                           ▼
[Tập TEST] ────────> ĐÁNH GIÁ CUỐI CÙNG (Chỉ chạy đúng 1 lần duy nhất để báo cáo)
```

### Các điều không trên tập Test:
1. **Không** dùng tập Test để chạy vòng lặp tìm số chủ đề $K$ tối ưu (ví dụ quét $K \in [5, 20]$ bằng điểm Coherence trên Test là vi phạm).
2. **Không** dùng tập Test để chọn mô hình tốt nhất (ví dụ so sánh LDA vs BERTopic trên Test để chọn model rồi mới tuning lại là sai quy trình).
3. **Không** fit bất kỳ transformer, scaler, vectorizer (`TfidfVectorizer`, `CountVectorizer`) nào trên tập Test. Vectorizer chỉ được `fit` trên tập Train và `transform` sang Val/Test.

---

## 4. Bảng Kiểm Định Tỷ Lệ Phân Tầng Chuyên Mục 


| Chuyên mục | Tỷ lệ Toàn bộ (%) | Tỷ lệ Train (%) | Tỷ lệ Val (%) | Tỷ lệ Test (%) | Độ lệch tối đa |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Giáo dục** | 11.15% | 11.14% | 11.21% | 11.15% | 0.06% |
| **Giải trí** | 12.72% | 12.71% | 12.80% | 12.73% | 0.08% |
| **Khoa học công nghệ** | 7.55% | 7.55% | 7.55% | 7.55% | **0.00%** |
| **Kinh doanh** | 21.37% | 21.38% | 21.35% | 21.37% | 0.02% |
| **Pháp luật** | 12.77% | 12.77% | 12.72% | 12.81% | 0.05% |
| **Sức khỏe** | 13.32% | 13.32% | 13.30% | 13.31% | 0.02% |
| **Thể thao** | 21.12% | 21.15% | 21.06% | 21.08% | 0.06% |

