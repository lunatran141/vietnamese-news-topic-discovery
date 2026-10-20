# M06 — Phân tích bài đa chủ đề bằng NMF

## Mục tiêu

Kiểm tra các bài test có trọng số đáng kể ở nhiều topic và phân tích ví dụ
bằng nội dung bài. Ngưỡng tự động chỉ dùng để lọc ứng viên; không xem đây là
nhãn ground truth.

## Protocol

Sau khi K được chốt từ M05, stage `finalize` fit vectorizer/NMF trên
train+validation và đánh giá một lần trên test:

```powershell
python -m experiments.nmf.run_experiment --stage finalize --n-topics 10
```

Mỗi hàng `W` được chuẩn hóa để tổng topic share bằng 1. Một bài được gắn cờ
nếu topic phụ đạt share ít nhất `0.20`; tối đa ba topic được giữ. Danh sách
case sắp theo gap nhỏ giữa hai share cao nhất, sau đó theo entropy cao. Các
ngưỡng nằm trong `configs/nmf.yaml`.

## Đầu ra hiện có

- `artifacts/nmf/nmf_p3_k10_seed42_v1/document_topics.parquet` — topic chính,
  topic phụ, shares và cờ đa chủ đề cho test.
- `results/nmf/nmf_p3_k10_seed42_v1/multi_topic_cases.csv` — tối đa 50 bài
  được chọn để review.
- `results/nmf/nmf_p3_k10_seed42_v1/multi_topic_case_studies.md` — năm case
  đã được TV2 đối chiếu với nội dung đầy đủ.
- `results/nmf/nmf_p3_k10_seed42_v1/test_metrics.json` — metric cuối.

Run hiện tại gắn cờ 523/1.390 bài test do có topic phụ đạt ngưỡng. Con số này
đo lượng ứng viên theo quy tắc share, không phải số bài đa chủ đề đã được
người xác nhận. Năm case trong ghi chú là phân tích sơ bộ và vẫn cần reviewer
độc lập xác nhận.

## Hạn chế và kiểm tra

- Topic share NMF là trọng số tương đối, không phải xác suất nhãn đúng.
- `category_gold` là chuyên mục báo dùng để đối chiếu; nó không phải nhãn topic
  chuẩn.
- Không dùng nhận xét từ test để điều chỉnh K, ngưỡng hoặc cách đặt tên rồi
  báo cáo lại trên cùng test như đánh giá độc lập.
- Mỗi case cần được kiểm tra bằng `article_id` và nội dung nguồn; ghi rõ bằng
  chứng cũng như bất đồng của reviewer.
