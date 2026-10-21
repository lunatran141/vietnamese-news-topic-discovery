# Bàn giao dữ liệu K-Means + TF-IDF phục vụ giao diện Streamlit

**Thành viên phụ trách:** TV1 — K-Means + TF-IDF  
**Run đại diện:** `kmeans_baseline`  
**Bộ dữ liệu:** `v1.0` (`9.274` bài báo)  
**Biến thể tiền xử lý:** `p2_segmented` (văn bản `title + content`, từ vựng TF-IDF `20.000` đặc trưng)  
**Số cụm chủ đề:** $K = 9$ (`init="k-means++"`, `n_init=10`, `max_iter=300`, `random_state=42`)

---

## 1. Mục tiêu bàn giao

Tài liệu này mô tả bộ common artifact hoàn chỉnh của mô hình **K-Means + TF-IDF** tại `artifacts/common/kmeans/kmeans_baseline/` để tích hợp đồng bộ vào giao diện Streamlit đa mô hình của nhóm.

Phạm vi dữ liệu bàn giao gồm:
- Đủ **8 file** theo chuẩn Common Artifact Schema `1.0` và mở rộng cho giao diện.
- Gán cụm đầy đủ cho cả 3 tập `train` (`6.493` bài), `val` (`1.391` bài) và `test` (`1.390` bài), tổng cộng **`9.274` bài báo**.
- Bảng tách thời gian chạy (`runtime breakdown`) cho từng công đoạn của cấu hình đại diện $K = 9$.
- Bộ chỉ số đánh giá nội tại và ngoại tại trên cả Validation và Test, kèm chỉ số ổn định đa hạt giống (`stability_mean`, `stability_std`) và độ đa dạng từ khóa (`topic_diversity`).
- Bảng bài báo đại diện (`representative_docs.csv`) xếp hạng theo khoảng cách Euclid tới tâm cụm (`distance_to_center`).
- Bảng tọa độ bản đồ 2 chiều (`projection_2d.csv`) trích xuất bằng `TruncatedSVD(n_components=2, random_state=42)` trên không gian TF-IDF.

---

## 2. Cấu trúc thư mục common artifact K-Means

Thư mục chuẩn: `artifacts/common/kmeans/kmeans_baseline/`

| Tên file | Số dòng / Định dạng | Chức năng |
| :--- | :--- | :--- |
| `manifest.json` | JSON | Khai báo phiên bản schema `1.0`, `model_name`, `run_id` và đường dẫn các file chuẩn |
| `model_summary.json` | JSON | Thông tin mô hình, giao thức đánh giá, runtime từng công đoạn, metrics Validation/Test và trạng thái outlier |
| `config_used.yaml` | YAML | Cấu hình thực tế của dữ liệu, TF-IDF và thuật toán K-Means ($K=9$, `seed=42`) |
| `document_topics.csv` | `9.274` dòng | Gán cụm `topic_id` (`0..8`), `distance_to_center` và metadata (`title`, `category_gold`, `source`, `published_at`) cho toàn bộ `train`, `val`, `test` |
| `topic_summary.csv` | `9` dòng | Thống kê số lượng bài, tỷ lệ, 5 từ khóa đầu, 3 tiêu đề đại diện và chuyên mục chiếm ưu thế của từng cụm |
| `topic_keywords.csv` | `135` dòng | Top 15 từ khóa có trọng số TF-IDF lớn nhất tại tâm cụm (`centroid`) của mỗi cụm (`9 × 15`) |
| `representative_docs.csv` | `45` dòng | Top 5 bài báo gần tâm cụm nhất của mỗi cụm (`topic_id,rank,article_id,distance_to_center,title,split,...`) |
| `projection_2d.csv` | `9.274` dòng | Tọa độ 2 chiều (`article_id,split,topic_id,x,y`) từ `TruncatedSVD` chiếu từ ma trận TF-IDF `20.000` chiều |

---

## 3. Giao thức Train / Validation / Test và chống rò rỉ dữ liệu

1. **Huấn luyện và chọn mô hình trên Validation:**
   - Bộ trích xuất đặc trưng `TfidfVectorizer` (`20.000` từ vựng) và mô hình `KMeans` chỉ được `fit` trên tập `train` (`6.493` bài).
   - Quá trình quét lưới $K \in \{3, 5, 7, 9, 11, 14, 18\}$ chỉ sử dụng tập `val` (`1.391` bài) để chọn và khóa cấu hình đại diện $K = 9$, `random_state = 42`.
2. **Suy diễn hậu kiểm một lần trên tập Test:**
   - Tập `test` (`1.390` bài) hoàn toàn không tham gia vào bước học từ vựng TF-IDF, không tham gia chọn $K$, và không dùng để rút trích bài đại diện.
   - Sau khi khóa mô hình `artifacts/kmeans/kmeans_model.joblib`, tập `test` chỉ được đưa qua hàm `predict_kmeans()` đúng một lần để lấy nhãn cụm và khoảng cách tới tâm cụm phục vụ báo cáo cuối cùng.

---

## 4. Tổng hợp chỉ số đánh giá Validation và Test

Nguồn lưu trữ: `artifacts/common/kmeans/kmeans_baseline/model_summary.json` và `results/kmeans/test_metrics.json`.

### 4.1. Các chỉ số theo split (`Validation` và `Test`)

| Chỉ số | Khoá trong `model_summary.json` | Validation (`1.391` bài) | Test (`1.390` bài) | Ghi chú |
| :--- | :--- | ---: | ---: | :--- |
| **Adjusted Rand Index (ARI)** | `validation_ari` / `test_ari` | `0.5925` | `0.5649` | Đối chiếu hậu kiểm với `category_gold` |
| **Normalized Mutual Info (NMI)** | `validation_nmi` / `test_nmi` | `0.7004` | `0.6774` | Đối chiếu hậu kiểm với `category_gold` |
| **Purity** | `validation_purity` / `test_purity` | `0.8728` | `0.8597` | Độ tinh khiết theo nhãn chuyên mục |
| **Silhouette Score** | `silhouette_score` / `test_silhouette_score` | `0.0190` | `0.0201` | Khoảng cách Euclid trên không gian thưa TF-IDF 20.000 chiều |
| **Calinski-Harabasz Score** | `calinski_harabasz_score` / `test_calinski_harabasz_score` | `11.3745` | `11.5909` | Tỷ số phương sai giữa các cụm và trong nội bộ cụm |
| **Davies-Bouldin Score** | `davies_bouldin_score` / `test_davies_bouldin_score` | `6.1566` | `6.1954` | Càng thấp càng tốt; có kèm alias `davies_bouldin_index` |
| **Inertia (Within-cluster SSE)** | `validation_inertia` / `test_inertia` | `1253.4474` | `1251.1536` | `train_inertia = 5844.7727` |

### 4.2. Chỉ số toàn mô hình (`Topic Diversity` & `Stability`)

| Chỉ số | Khoá trong `model_summary.json` | Giá trị | Ý nghĩa |
| :--- | :--- | ---: | :--- |
| **Topic Diversity (Top-10)** | `topic_diversity` | `0.9556` | Tỷ lệ từ khóa duy nhất trên $9 \times 10 = 90$ vị trí từ khóa đầu của 9 tâm cụm (`86 / 90`) |
| **Stability Mean (ARI đa seed)** | `stability_mean` | `0.8224` | Trung bình ARI từng cặp qua 5 hạt giống `{42, 43, 44, 45, 46}` tại $K=9$ |
| **Stability Std (ARI đa seed)** | `stability_std` | `0.0444` | Độ lệch chuẩn ARI qua 5 hạt giống tại $K=9$ |

---

## 5. Bảng tách thời gian chạy (Runtime Breakdown)

Thời gian thực thi được đo tách biệt cho từng công đoạn của cấu hình đại diện $K = 9$, `seed = 42`:

| Công đoạn | Khoá trong `model_summary.json` | Thời gian (giây) | Nội dung đo |
| :--- | :--- | ---: | :--- |
| **Biểu diễn TF-IDF** | `representation_seconds` | `3.9100` | `fit` vectorizer trên `train` + `transform` cho `train`, `val`, `test` (`9.274` bài) |
| **Huấn luyện K-Means** | `fit_seconds` | `2.8408` | `KMeans(n_clusters=9, init='k-means++', n_init=10, random_state=42).fit(train)` |
| **Suy diễn Validation** | `validation_predict_seconds` | `0.0034` | Gán cụm và tính khoảng cách tới tâm cụm cho `1.391` bài `val` |
| **Suy diễn Test** | `test_predict_seconds` | `0.0044` | Gán cụm và tính khoảng cách tới tâm cụm cho `1.390` bài `test` |
| **Đánh giá Metric** | `evaluation_seconds` | `0.9891` | Tính bộ chỉ số nội tại (Silhouette, Davies-Bouldin, Calinski-Harabasz) và ngoại tại (ARI, NMI, Purity) |
| **Tổng thời gian chạy cấu hình $K=9$** | `runtime_seconds` | **`7.7477`** | Tổng 5 công đoạn trên (không gồm đọc/ghi ổ đĩa, quét lưới K-search và vẽ biểu đồ) |
| **Thời gian quét lưới M02 (tham khảo)** | `k_search_pipeline_seconds` | `112.6200` | Tổng thời gian chạy toàn bộ pipeline M02 quét 7 giá trị $K \times 5$ hạt giống và xuất hình |

---

## 6. Lưu ý hiển thị trên giao diện cho K-Means

1. **Trạng thái Outlier (`Không áp dụng`):**
   - K-Means là thuật toán phân hoạch cứng toàn bộ điểm dữ liệu vào $K$ cụm, không có cơ chế từ chối bài nhiễu như HDBSCAN/BERTopic.
   - Trong `model_summary.json`, trường `"outlier_rate"` được đặt là `null` (thay vì `0.0`), kèm `"outlier_applicable": false` và `"outlier_label": "Không áp dụng"`. Giao diện không nên xếp hạng `0%` outlier của K-Means như một ưu thế trước HDBSCAN/BERTopic.
2. **Ý nghĩa khoảng cách tới tâm cụm (`distance_to_center`):**
   - Trong `document_topics.csv` và `representative_docs.csv`, K-Means không sinh phân phối xác suất nên `topic_probability = null`, thay bằng cột `distance_to_center` (khoảng cách Euclid từ vector TF-IDF chuẩn hóa $\ell_2$ của bài báo đến vector trọng tâm cụm). Giá trị **`distance_to_center` càng nhỏ** thì bài báo càng nằm gần tâm cụm và càng tiêu biểu cho cụm đó.
3. **Bản đồ 2D (`projection_2d.csv`):**
   - Tọa độ `x, y` được tạo bằng `TruncatedSVD(n_components=2, random_state=42)` fit trên ma trận TF-IDF tập `train` và chiếu cho cả `train`, `val`, `test`. Bản đồ 2D chỉ phục vụ trực quan hóa khám phá, trong khi K-Means phân cụm trực tiếp trên không gian thưa TF-IDF `20.000` chiều.

---

## 7. Lệnh tái lập và kiểm thử

```powershell
# 1. Cập nhật đánh giá hậu kiểm Test, đo runtime và xuất 8 file common artifacts
.\venv\Scripts\python.exe -m experiments.kmeans.export_common_artifacts

# 2. Chạy bộ kiểm thử tự động cho K-Means common artifacts và loader
.\venv\Scripts\pytest.exe tests/test_kmeans_artifact_export.py tests/test_artifact_schemas.py tests/test_artifact_loader.py -v
```
