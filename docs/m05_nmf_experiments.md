# M05 — Chọn số topic NMF

## Mục tiêu

So sánh các giá trị K trên validation để đề xuất cấu hình cuối, đồng thời xuất
keywords, trọng số topic và bài đại diện để review thủ công. Không dùng test
để chọn K.

## Protocol

Runner fit TF-IDF và NMF trên train, rồi biến đổi validation bằng các mô hình
đã fit. Các giá trị K mặc định là 4, 6, 8, 10 và 12. Tham số còn lại dùng
`configs/nmf.yaml`.

Runner xếp hạng theo NPMI coherence giảm dần, topic diversity giảm dần, lỗi
tái tạo chuẩn hóa tăng dần và K tăng dần. Metric này đề xuất K; trước khi chốt
cần xem topic keywords cùng bài đại diện. Các tiêu chí và bộ dữ liệu cần ghi
theo từng run.

## Chạy và đầu ra

```powershell
python -m experiments.nmf.run_experiment --stage select
```

Run review hiện có là `nmf_p3_select_review_seed42_v1`. Tại thời điểm ghi tài
liệu, K=10 đạt NPMI validation cao nhất trong các giá trị đã thử. K=12 có lỗi
tái tạo chuẩn hóa thấp hơn. Đây là kết quả metric, chưa phải quyết định cuối
được xác nhận bằng đánh giá con người.

Thư mục `results/nmf/<run_id>/` chứa:

- `topic_count_metrics.csv` và `topic_count_selection.png`.
- `candidate_topic_summary.csv` — keywords và metric theo topic/K.
- `candidate_representative_docs.csv` — 5 bài validation mỗi topic/K.
- `candidate_topic_review.csv` — phiếu đánh giá thủ công.

`artifacts/nmf/<run_id>/` lưu các phân phối validation theo K cùng `config.json`
và `run_metadata.json`. Phiếu review hiện có 40 dòng vì tổng topic của năm
giá trị K là 40.

## Review thủ công và trạng thái

Ít nhất hai người chấm độc lập cần đọc keywords và bài đại diện, ghi nhãn, điểm
diễn giải 1–5 và ghi chú topic chồng lấn/nhiễu. Quy ước điểm nằm trong
`experiments/nmf/README.md`. Không tự điền điểm từ metric.

Phiếu `results/nmf/nmf_p3_select_review_seed42_v1/candidate_topic_review.csv`
hiện chưa có nhãn, điểm, reviewer hoặc ghi chú. Do đó K=10 là đề xuất NPMI,
chưa đủ evidence để tuyên bố review M05 đã hoàn tất.

## Kiểm tra

- Xác nhận stage `select` fit trên train và tính metric/chọn K trên validation.
- Xác nhận metadata ghi `test_accessed: false`.
- Đọc topic words và toàn bộ bài đại diện cần thiết trước khi ghi điểm.
- Chuyển K đã review sang M04/M06 để chạy final; không chạy nhiều K trên test.
