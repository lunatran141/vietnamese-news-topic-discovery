# BERTopic E1: đánh giá cuối và tích hợp giao diện

## Phạm vi

E1 (`e1_vietnamese_sbert`) được chọn trên validation vì có tỷ lệ outlier
validation thấp nhất trong E0–E3. Đây không phải khẳng định E1 tốt nhất trên
mọi metric. E0–E3 tiếp tục được trình bày tại trang **Thí nghiệm**; trang
**So sánh mô hình** dùng E1 làm run BERTopic đại diện.

## Runtime

Run mới xuất riêng:

- `model_load_seconds`
- `representation_seconds`
- `model_init_seconds`
- `fit_seconds`
- `validation_predict_seconds`
- `evaluation_seconds`
- `runtime_seconds`
- `runtime_scope`

`runtime_seconds` là tổng thời gian pipeline được nêu trong `runtime_scope`,
không được gọi là thời gian huấn luyện. `fit_seconds` mới là thời gian fit.
Để benchmark, chạy warm-up rồi đo 3–5 lần trên cùng máy, cùng cache và ghi
CPU/GPU, RAM, batch size, seed cùng phiên bản thư viện.

## Validation metrics

Run thí nghiệm mới xuất thêm ARI, NMI và NPMI coherence trên validation.
ARI/NMI dùng `category_gold`, vốn là chuyên mục báo chí rộng chứ không phải
ground truth topic. Outlier vẫn được giữ như một cluster riêng trong ARI/NMI
chính; nếu báo cáo metric trên phần được bao phủ phải ghi rõ phạm vi.

## Test cuối

Test chỉ được mở sau khi khóa E1 và tiêu chí đánh giá. Script cuối tái fit E1
trên `train + validation`, sau đó dự đoán test đúng một lần:

```powershell
.\.venv\Scripts\python.exe -m experiments.bertopic.evaluate_e1_test `
    --confirm-test-evaluation
```

Script xuất `test_document_topics.csv`, bổ sung metric test vào `metrics.json`
và lưu full model để inference. Không dùng kết quả test để đổi E1 hoặc tuning.

Sau khi chạy xong, xuất lại common artifact:

```powershell
.\.venv\Scripts\python.exe -m experiments.export_common_artifacts
```

## Phân tích lỗi

Khi `calculate_probabilities=true`, artifact mới có:

- `second_topic_probability`
- `top_two_share_gap`
- `topic_entropy`
- `is_multi_topic`

Các giá trị này là membership strength của BERTopic/HDBSCAN, không nên mô tả
như xác suất đã hiệu chỉnh. Artifact cũ không có ma trận membership sẽ để trống
các trường chẩn đoán thay vì tạo dữ liệu giả.

## Việc cần toàn nhóm cung cấp

Mỗi phương pháp phải có một run đại diện, test artifact, ARI/NMI nếu áp dụng,
coherence/diversity theo định nghĩa đã ghi và runtime có `runtime_scope`.
Giá trị chưa đo để trống; không thay bằng 0. Chỉ so sánh trực tiếp runtime khi
các run được benchmark cùng điều kiện và cùng phạm vi.
