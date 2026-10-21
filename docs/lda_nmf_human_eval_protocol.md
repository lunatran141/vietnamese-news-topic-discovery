
# Human Evaluation Protocol — LDA Final vs NMF Final

## 1. Mục tiêu

Đánh giá định tính khả năng diễn giải và mức độ phù hợp giữa bài báo và chủ đề của hai pipeline Topic Modeling:

- LDA Final: K=15, P2 Segmented, Bag-of-Words, seed=42.
- NMF Final: K=10, P3 No Stopwords, TF-IDF, seed=42.

Human Evaluation bổ sung cho các kết quả NPMI, Topic Diversity, ARI và NMI đã báo cáo trong `docs/lda_nmf_comparison.md`.

Đây là so sánh hai pipeline hoàn chỉnh, không phải thí nghiệm tách riêng ảnh hưởng của thuật toán.

## 2. Dữ liệu và quy trình lấy mẫu

- Dataset: v1.0, gồm 9.274 bài báo.
- Nguồn lấy mẫu: Train (6.493 bài) và Validation (1.391 bài).
- Không lấy mẫu từ Test (1.390 bài).
- Random seed: 42.
- Mô hình và tham số Final được giữ nguyên.

Đánh giá toàn bộ 25 topic:

| Model | Topics | Top-ranked/topic | Random/topic | Tổng mẫu |
|---|---:|---:|---:|---:|
| LDA | 15 | 3 | 2 | 75 |
| NMF | 10 | 3 | 2 | 50 |
| Tổng | 25 | 75 | 50 | 125 |

### 2.1. Top-ranked samples

- LDA: sử dụng ba representative documents/topic từ frozen Final artifact, thuộc Train+Validation.
- NMF: chọn ba bài có topic share cao nhất trong nhóm bài được gán dominant topic tương ứng trên Train+Validation.
- Khi điểm bằng nhau, sắp xếp theo `article_id` để bảo đảm tái lập.

### 2.2. Random samples

- Hai bài/topic được chọn ngẫu nhiên, không hoàn lại.
- Chỉ chọn trong nhóm bài có dominant topic tương ứng.
- Loại trừ ba bài top-ranked của topic đó.
- Sử dụng `numpy.random.default_rng(seed=42)`.
- Sắp xếp ứng viên theo `article_id` trước khi lấy mẫu.

Một bài báo có thể xuất hiện ở các topic hoặc mô hình khác nhau. Đơn vị đánh giá là cặp bài–topic.

### 2.3. Nội dung hiển thị

Người đánh giá đọc:

- Top-10 keywords của topic.
- Tiêu đề và nội dung bài báo lấy từ cùng Dataset v1.0.

Không hiển thị `category_gold`, nhãn topic sơ bộ, tên mô hình hoặc topic ID gốc.

## 3. Reviewer và đánh giá mù

Có ít nhất hai reviewer độc lập, ký hiệu R1 và R2.

- Mỗi reviewer đánh giá toàn bộ 25 topic và 125 cặp bài–topic.
- Topic được mã hóa thành `T01` đến `T25`.
- Thứ tự năm bài trong mỗi topic được xáo trộn.
- Reviewer không được xem `topic_mapping.csv` hoặc `evaluation_samples.csv`.
- Reviewer không trao đổi kết quả trước khi khóa phiếu.
- Không sử dụng AI để tự động tạo điểm thay reviewer.

Sau khi cả hai reviewer nộp phiếu, người tổng hợp mới giải mã topic.

## 4. Tiêu chí đánh giá

### 4.1. Keyword Coherence (1–5)

Đánh giá sự liên kết ngữ nghĩa của 10 từ khóa trong cùng topic.

| Điểm | Diễn giải |
|---|---|
| 1 | Từ khóa rời rạc, không hình thành chủ đề rõ ràng |
| 2 | Một số từ liên quan nhưng phần lớn thiếu nhất quán |
| 3 | Nhận diện được chủ đề rộng, còn nhiều từ không phù hợp |
| 4 | Phần lớn từ khóa liên quan rõ ràng |
| 5 | Từ khóa nhất quán, thể hiện chủ đề cụ thể |

Chấm một lần cho mỗi topic.

### 4.2. Topic Interpretability (1–5)

Đánh giá khả năng nhận diện và đặt tên topic sau khi xem keywords và năm bài mẫu.

| Điểm | Diễn giải |
|---|---|
| 1 | Không thể xác định chủ đề |
| 2 | Khó diễn giải, chỉ đặt được nhãn rất mơ hồ |
| 3 | Đặt được nhãn tổng quát nhưng nội dung còn pha trộn |
| 4 | Chủ đề rõ ràng, có thể đặt tên tương đối chính xác |
| 5 | Chủ đề rất rõ, nhãn ngắn gọn và đặc trưng |

Chấm một lần cho mỗi topic.

### 4.3. Document–Topic Relevance (1–5)

Đánh giá mức độ phù hợp của từng bài báo với topic được đề xuất.

| Điểm | Diễn giải |
|---|---|
| 1 | Bài không liên quan đến topic |
| 2 | Chỉ liên quan rất ít hoặc gián tiếp |
| 3 | Liên quan một phần, trọng tâm bài thuộc chủ đề khác |
| 4 | Phần lớn nội dung bài phù hợp với topic |
| 5 | Nội dung bài rất phù hợp với topic |

Chấm riêng từng cặp bài–topic.

### 4.4. Mixed Topic (yes/no)

- `yes`: Topic pha trộn rõ rệt nhiều nhóm nội dung không liên quan.
- `no`: Các nội dung thuộc một chủ đề rộng vẫn có thể diễn giải hợp lý.

Reviewer ghi lý do trong `topic_notes` khi chọn `yes`.

### 4.5. Topic quá rộng và khó đặt tên

- `too_broad_yes_no`: chọn `yes` nếu topic gộp phạm vi quá rộng khiến nhãn chung khó phân biệt các bài; `no` nếu phạm vi vẫn hợp lý. Đây là tiêu chí khác với pha trộn các nội dung không liên quan.
- `hard_to_name_yes_no`: chọn `yes` nếu sau khi đọc keywords và bài mẫu vẫn khó đặt nhãn ngắn gọn, rõ nghĩa; `no` nếu đặt tên được. Không tự quy đổi điểm interpretability thành cờ này.
- Reviewer chấm độc lập và giải thích các trường hợp `yes` trong `topic_notes`.

## 5. Phiếu đánh giá

Mỗi reviewer được cung cấp hai file:

- `reviewer_R1_topics.csv` / `reviewer_R2_topics.csv`: 25 dòng chấm cấp topic.
- `reviewer_R1_documents.csv` / `reviewer_R2_documents.csv`: 125 dòng chấm cấp bài.

Các cột điểm phải sử dụng số nguyên 1, 2, 3, 4 hoặc 5.

`mixed_topic_yes_no` chỉ nhận `yes` hoặc `no`.

`proposed_topic_label` do reviewer tự đặt, không sao chép nhãn sơ bộ.

Không sửa các mã `sample_id`, `blind_topic_id`, keywords, tiêu đề hoặc nội dung bài.

## 6. Quy trình thực hiện

1. Người quản lý kiểm tra artifact và tạo mẫu với seed cố định.
2. Khóa bộ mẫu và lưu bảng mapping riêng.
3. R1 và R2 đánh giá độc lập theo cùng rubric.
4. Mỗi reviewer nộp hai phiếu đã hoàn thành.
5. Kiểm tra dữ liệu chấm: đầy đủ, đúng thang điểm, đúng ID.
6. Khóa kết quả trước khi mở bảng mapping.
7. Tổng hợp theo topic và theo mô hình.
8. Đưa kết quả thực tế vào báo cáo NCKH.

## 7. Tổng hợp và báo cáo

Báo cáo riêng:

- Keyword Coherence trung bình theo topic và mô hình.
- Topic Interpretability trung bình theo topic và mô hình.
- Document Relevance trung bình theo topic, phân biệt mẫu top-ranked và random khi phân tích.
- Tỷ lệ Mixed Topic theo mô hình.
- Chênh lệch điểm giữa hai reviewer.
- Mức độ đồng thuận giữa reviewer.

Có thể dùng weighted Cohen's kappa cho điểm thứ bậc khi dữ liệu phù hợp; báo cáo thêm tỷ lệ đồng thuận tuyệt đối. Trường hợp chỉ số không xác định phải ghi rõ, không tự gán bằng 0.

Điểm Document Relevance được tổng hợp theo topic trước, không xem tất cả lượt chấm là quan sát độc lập.

## 8. Giới hạn nghiên cứu

- LDA và NMF khác preprocessing, representation và số topic.
- Mỗi topic chỉ được kiểm tra năm bài từ Development.
- Mẫu random giới hạn trong các bài có dominant topic tương ứng.
- Hai reviewer có thể có nhận định chủ quan.
- Reviewer có thể nhận biết lĩnh vực chủ đề, dù không biết tên mô hình.
- Kết quả Human Evaluation là bằng chứng bổ sung, không thay thế Final Test.
- Không dùng điểm Human Evaluation để chọn lại K, thay đổi cấu hình hoặc điều chỉnh kết quả Final Test.

## 9. Reproducibility

Script tạo mẫu:

`experiments/human_eval/build_lda_nmf_samples.py`

Nguồn dữ liệu:

- `artifacts/common/lda/lda_k15_seed42/`
- `artifacts/nmf/nmf_p3_k10_seed42_dashboard_v2/` (bộ phiếu bàn giao giao diện; run nguồn được ghi trong metadata)
- `data/processed/p2_segmented.csv`
- `data/splits/split_manifest.csv`

Bộ mẫu phải được tạo lại với cùng seed và cùng artifact trước khi đánh giá chính thức.

Lệnh cho bộ phiếu NMF dashboard, không ghi đè bộ đã có:

```powershell
.venv/Scripts/python.exe -m experiments.human_eval.build_lda_nmf_samples --nmf-dir artifacts/nmf/nmf_p3_k10_seed42_dashboard_v2 --output-dir results/human_eval/lda_nmf_new
```

Hai cờ `too_broad_yes_no` và `hard_to_name_yes_no` cũng chỉ nhận `yes`/`no`. Script tổng hợp là `experiments/human_eval/summarize_lda_nmf.py`; chỉ công bố điểm khi cả hai reviewer chấm đủ. Xem `docs/nmf_dashboard.md` để biết trạng thái bộ phiếu hiện tại và phạm vi so sánh.
