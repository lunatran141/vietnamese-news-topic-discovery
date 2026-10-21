# BÀN GIAO DỮ LIỆU LDA PHỤC VỤ GIAO DIỆN STREAMLIT

**Thành viên phụ trách:** TV3 — LDA + Bag-of-Words
**Run đại diện:** `lda_k15_seed42`
**Dataset:** v1.0
**Biến thể tiền xử lý:** `p2_segmented`
**Số chủ đề:** K = 15
**Random seed:** 42
**Trạng thái:** Đã hoàn thiện dữ liệu và kiểm thử tại máy local, chờ commit và tạo Pull Request.

**Human Evaluation LDA/NMF: ĐANG CHỜ — R1/R2 chưa hoàn thành đánh giá.**

---

## 1. Mục tiêu bàn giao

Cung cấp đầy đủ dữ liệu, kết quả thực nghiệm và các chỉ số đánh giá của mô hình Latent Dirichlet Allocation (LDA) để thành viên phụ trách giao diện Streamlit có thể tích hợp.

Phạm vi bàn giao gồm:

- Common Artifact theo Schema 1.0.
- Kết quả đánh giá trên Validation và Test.
- Các chỉ số Runtime, Perplexity và Topic Diversity.
- Kết quả đánh giá Stability theo random seed.
- Thông tin tài liệu đa chủ đề (Multi-topic).
- Dữ liệu phục vụ biểu đồ thực nghiệm và trực quan hóa chủ đề.
- Quy trình tái lập và kiểm thử.

TV3 phụ trách cung cấp và xác minh dữ liệu LDA. **Việc xây dựng và chỉnh sửa giao diện Streamlit do thành viên phụ trách giao diện thực hiện.**

## 2. Common Artifact LDA

Thư mục chuẩn:

`artifacts/common/lda/lda_k15_seed42/`

Gồm 8 file:

| Tên file | Chức năng |
|---|---|
| `manifest.json` | Thông tin định danh run và phiên bản Schema |
| `model_summary.json` | Thông tin mô hình, metrics và runtime |
| `config_used.yaml` | Cấu hình được sử dụng |
| `document_topics.csv` | Kết quả gán chủ đề cho từng bài báo |
| `topic_summary.csv` | Thống kê số lượng và tỷ lệ tài liệu của từng chủ đề |
| `topic_keywords.csv` | Danh sách từ khóa đại diện của các chủ đề |
| `representative_docs.csv` | Các bài báo đại diện cho từng chủ đề |
| `projection_2d.csv` | Tọa độ 2D phục vụ trực quan hóa |

### 2.1. Thống kê dữ liệu

| Tập dữ liệu | Số lượng bài báo |
|---|---:|
| Train | 6.493 |
| Validation | 1.391 |
| Test | 1.390 |
| **Tổng cộng** | **9.274** |

Mô hình LDA Final sử dụng K = 15, seed = 42, huấn luyện trên Train + Validation và đánh giá cuối cùng trên Test.

## 3. Dữ liệu gán chủ đề mở rộng

File `document_topics.csv` gồm 14 cột:

| Cột | Ý nghĩa |
|---|---|
| `article_id` | Mã định danh bài báo |
| `split` | Tập Train, Validation hoặc Test |
| `topic_id` | Chủ đề có xác suất cao nhất |
| `topic_probability` | Xác suất của chủ đề cao nhất |
| `second_topic_id` | Chủ đề có xác suất cao thứ hai |
| `second_topic_probability` | Xác suất của chủ đề cao thứ hai |
| `top_two_share_gap` | Chênh lệch xác suất giữa hai chủ đề cao nhất |
| `topic_entropy` | Mức độ phân tán của phân phối xác suất chủ đề |
| `is_multi_topic` | Đánh dấu tài liệu đa chủ đề |
| `title` | Tiêu đề bài báo |
| `category_gold` | Nhãn danh mục tham chiếu |
| `is_outlier` | Trạng thái ngoại lai |
| `source` | Nguồn bài báo |
| `published_at` | Thời điểm xuất bản |

### 3.1. Xác định tài liệu đa chủ đề

Quy tắc được thống nhất:

`is_multi_topic = second_topic_probability >= 0.20`

Kết quả:

- Tổng số bài báo: **9.274**.
- Số bài được đánh dấu đa chủ đề: **4.574**.
- Tỷ lệ bài đa chủ đề: **49,32%**.

`top_two_share_gap` và `topic_entropy` giúp giao diện phân tích mức độ không chắc chắn và hiện tượng chồng lấn chủ đề.

LDA không sử dụng cơ chế phát hiện nhiễu như HDBSCAN. Trường `is_outlier` của run này được đặt là `False`.

## 4. Kết quả đánh giá LDA

Các chỉ số được lưu trong:

`artifacts/common/lda/lda_k15_seed42/model_summary.json`

### 4.1. Đánh giá Development Validation

Mô hình K = 15, seed = 42, được huấn luyện trên Train và đánh giá trên Validation.

| Chỉ số | Giá trị |
|---|---:|
| ARI | 0,415226 |
| NMI | 0,559439 |
| NPMI | 0,134351 |
| Perplexity | 1.783,212573 |
| Topic Diversity | 0,640000 |

Nguồn:

`results/lda/validation_supplement/validation_metrics.json`

### 4.2. Đánh giá Final Test

Mô hình Final được huấn luyện trên Train + Validation và đánh giá trên Test.

| Chỉ số | Giá trị |
|---|---:|
| ARI | 0,406466 |
| NMI | 0,572713 |
| NPMI | 0,147094 |
| Topic Diversity | 0,653333 |

Nguồn kết quả gốc:

`results/lda/final/test_metrics.json`

**Lưu ý về tính khoa học:**

- NPMI của hai giai đoạn sử dụng cùng định nghĩa `document_binary_common`.
- Development và Final là hai mô hình được huấn luyện trên các tập dữ liệu khác nhau.
- Perplexity hiện có được tính trên Validation của mô hình Development, không phải Perplexity của Final Test.
- Không sử dụng Test để lựa chọn lại K hoặc điều chỉnh siêu tham số.
- Các kết quả Final Test đã được chốt và giữ nguyên.

## 5. Runtime LDA

Kết quả benchmark Final Train+Validation → Test, gồm 5 lần lặp.

| Trường | Giá trị (giây) |
|---|---:|
| `representation_seconds` | 2,3262 |
| `fit_seconds` | 19,7644 |
| `test_inference_seconds` | 0,2286 |
| `evaluation_seconds` | 0,0127 |
| `runtime_seconds` | 22,3437 |
| `runtime_mean` | 22,3437 |
| `runtime_std` | 0,3521 |

Thời gian suy luận riêng của benchmark Development:

`validation_inference_seconds = 0.2198`

Nguồn:

`results/lda/runtime/runtime_summary.csv`

**Phạm vi đo:** Runtime Final bao gồm biểu diễn Bag-of-Words, huấn luyện LDA, suy luận trên Test và tính các chỉ số đánh giá. Không bao gồm thời gian tải dữ liệu và xuất artifact.

Thời gian suy luận trên Validation lấy từ benchmark Development riêng, không thuộc tổng Runtime Final.

## 6. Đánh giá độ ổn định — M09

Thực nghiệm M09 sử dụng:

- Số chủ đề cố định: K = 15.
- Seed tham chiếu: 42.
- Các seed so sánh: 52, 62, 72, 82.
- Tập huấn luyện: Train.
- Tập đánh giá: Validation.

### 6.1. Kết quả Stability

| Chỉ số | Mean | Std |
|---|---:|---:|
| Matched Jaccard | 0,443351 | 0,034057 |
| Document Cosine | 0,678444 | 0,045904 |

Độ lệch chuẩn được tính theo công thức độ lệch chuẩn mẫu (`ddof=1`).

Trường hợp seed 42 tự so sánh với chính nó không được đưa vào thống kê tổng hợp.

`stability_mean` và `stability_std` trong Common Artifact sử dụng **Matched Jaccard**, không phải ARI hoặc NMI.

### 6.2. Kết quả theo từng seed

File:

`results/lda/m09/stability_per_seed.csv`

| Seed | ARI | NMI | NPMI | Diversity | Runtime (giây) |
|---|---:|---:|---:|---:|---:|
| 42 | 0,415226 | 0,559439 | 0,134351 | 0,640000 | 31,90 |
| 52 | 0,405376 | 0,590689 | 0,128216 | 0,586667 | 24,48 |
| 62 | 0,439829 | 0,600622 | 0,142414 | 0,613333 | 23,58 |
| 72 | 0,460879 | 0,597811 | 0,120071 | 0,626667 | 23,57 |
| 82 | 0,405311 | 0,580351 | 0,128628 | 0,626667 | 26,29 |

ARI, NMI, NPMI và Topic Diversity được tính trong lần đánh giá Development bổ sung, theo cùng cấu hình M09.

Runtime lấy từ kết quả thực nghiệm M09 gốc tại `runtime_by_seed.csv`, không phải thời gian chạy của script đánh giá bổ sung.

## 7. Dữ liệu phục vụ giao diện và biểu đồ

TV3 cung cấp các nguồn dữ liệu sau cho thành viên phụ trách Streamlit.

| Chức năng hiển thị | File dữ liệu |
|---|---|
| Biểu đồ chọn K — Coherence CV/NPMI | `results/lda/m08/topic_number_results.csv` |
| Topic Diversity và Runtime theo K | `results/lda/m08/topic_number_results.csv` |
| Biểu đồ Stability | `results/lda/m09/stability_metrics.csv` |
| ARI/NMI/NPMI theo seed | `results/lda/m09/stability_per_seed.csv` |
| Tổng quan metrics và runtime | `model_summary.json` |
| Phân bố chủ đề | `document_topics.csv`, `topic_summary.csv` |
| Từ khóa đại diện | `topic_keywords.csv` |
| Bài báo đại diện | `representative_docs.csv` |
| Bản đồ chủ đề 2D | `projection_2d.csv` |
| Phân tích Multi-topic | `document_topics.csv` |
| Xu hướng chủ đề theo thời gian | `document_topics.csv`, sử dụng `published_at` |

Các tên file không ghi đầy đủ đường dẫn trong bảng thuộc thư mục Common Artifact LDA tại mục 2.

**Lưu ý khi trực quan hóa kết quả chọn K:**

Trong tập giá trị K đã thử của M08, K = 20 có Coherence CV và NPMI cao nhất. K = 15 là cấu hình đại diện đã được nhóm phê duyệt, không phải cấu hình có Coherence cao nhất trong toàn bộ tập thử nghiệm.

TV3 cung cấp dữ liệu thực nghiệm. Việc thiết kế và lập trình các biểu đồ trên Streamlit do thành viên phụ trách giao diện thực hiện.

## 8. Human Evaluation LDA/NMF — ĐANG CHỜ R1/R2

**Trạng thái: PENDING — Chưa hoàn thành đánh giá thủ công.**

Tài liệu quy trình:

`docs/lda_nmf_human_eval_protocol.md`

Dữ liệu mẫu:

`results/human_eval/lda_nmf/evaluation_samples.csv`

Bộ mẫu đánh giá gồm:

- 15 topic LDA.
- 10 topic NMF.
- Tổng cộng 25 topic.
- 125 cặp bài báo–topic cho mỗi reviewer.
- Hai reviewer độc lập: R1 và R2.

Các phiếu đánh giá đã được chuẩn bị, nhưng hiện chưa có điểm chấm thực tế.

Vì vậy:

- Chưa tổng hợp điểm Human Evaluation.
- Chưa tính mức đồng thuận giữa hai reviewer.
- Chưa đưa ra kết luận định tính dựa trên điểm chấm.
- Không tự động tạo hoặc giả lập điểm đánh giá bằng AI.

Sau khi R1 và R2 hoàn thành, nhóm sẽ kiểm tra phiếu, tổng hợp kết quả thực tế và cập nhật báo cáo.

**Việc chờ R1/R2 không ảnh hưởng đến việc bàn giao các kết quả thực nghiệm LDA đã hoàn thành.**

## 9. Kiểm thử và tái lập kết quả

### 9.1. Kiểm thử Common Artifact

Chạy tại thư mục gốc dự án:

```powershell
python -m pytest tests/test_lda_artifact_export.py tests/test_lda_schema_extension.py -q
```

Kết quả kiểm tra local:

**24/24 test PASS.**

Các nội dung đã kiểm tra gồm:

- Đủ cấu trúc Common Artifact.
- Đúng 9.274 tài liệu và 15 topic.
- Đúng phân chia Train/Validation/Test.
- Các trường Multi-topic và Top-2.
- Metrics và Runtime.
- Evaluation Protocol.
- Per-seed Metrics CSV.
- Tính hợp lệ của Projection 2D và Representative Documents.

### 9.2. Tái tạo kết quả đánh giá bổ sung

Script Validation:

```powershell
python -m experiments.lda.evaluate_lda_validation
```

Script Per-seed Metrics:

```powershell
python -m experiments.lda.evaluate_m09_per_seed
```

Hai script từ chối ghi đè file kết quả đã tồn tại. Chỉ chạy lại khi chuẩn bị môi trường tái lập phù hợp và đã bảo toàn kết quả cũ.

### 9.3. Xuất Common Artifact

Ví dụ xuất vào một thư mục kiểm chứng mới chưa tồn tại:

```powershell
python -m experiments.lda.export_common_artifacts `
  --source-dir artifacts/lda/final `
  --output-dir artifacts/common/lda/lda_verification_export
```

Không ghi đè mô hình LDA Final hoặc dữ liệu Test đã được chốt.

### 9.4. Kiểm tra Data Service

Đã kiểm tra thành công việc tải run LDA bằng `app/data_service.py`.

Kết quả:

- Model: LDA K15.
- Số bài báo: 9.274.
- Số topic: 15.
- Runtime: 22,3437 giây.
- Common Artifact Loader: PASS.
- Data Service: PASS.

## 10. Trạng thái bàn giao

| Hạng mục | Trạng thái |
|---|---|
| LDA Final K15 seed42 | Hoàn thành |
| Common Artifact Schema 1.0 | Hoàn thành tại local |
| Document Topics mở rộng | Hoàn thành |
| Validation/Test Metrics | Hoàn thành |
| Runtime chi tiết | Hoàn thành |
| Stability Mean/Std | Hoàn thành |
| Per-seed Metrics CSV | Hoàn thành |
| Automated Tests | 24/24 PASS |
| Data Service kiểm tra LDA | PASS |
| Human Evaluation R1/R2 | **PENDING — Chờ reviewer** |
| Tích hợp giao diện Streamlit | Thành viên phụ trách giao diện thực hiện |
| GitHub Pull Request | Chờ tạo PR bàn giao |

### Kết luận

TV3 đã hoàn thiện phần dữ liệu, kết quả thực nghiệm và Common Artifact mở rộng của mô hình LDA để cung cấp cho giao diện Streamlit.

Các kết quả đã được kiểm tra tại local. Bước tiếp theo là commit, push và tạo Pull Request để bàn giao cho nhóm.

**Human Evaluation LDA/NMF vẫn ở trạng thái chờ R1/R2 đánh giá và sẽ được cập nhật sau khi có kết quả thực tế.**
