# NMF experiments (TV2 — M04, M05, M06)

Module này triển khai NMF trên Dataset V1 đã duyệt, chọn số topic bằng
validation và xuất case study bài đa chủ đề trên test.

Chạy chọn K:

```powershell
python -m experiments.nmf.run_experiment --stage select
```

Review `topic_count_metrics.csv`, `topic_count_selection.png`,
`candidate_topic_summary.csv` và `candidate_representative_docs.csv` trên
validation. Điền `candidate_topic_review.csv` sau khi đọc bài đại diện rồi
chạy K đã chốt:

```powershell
python -m experiments.nmf.run_experiment --stage finalize --n-topics 10
```

Không dùng stage `finalize` để thử nhiều K vì stage này đọc test set. Cấu hình,
seed và dữ liệu được mô tả tại:

- `docs/m04_nmf_guide.md` — pipeline và model artifacts.
- `docs/m05_nmf_experiments.md` — chọn K trên validation.
- `docs/m06_nmf_multi_topic_analysis.md` — case đa chủ đề trên test.

Xuất run cuối sang common artifact schema:

```powershell
python -m experiments.export_common_artifacts --model nmf
```

Exporter đọc Parquet gốc, tạo bundle tại
`artifacts/common/nmf/<run_id>/` và kiểm tra bundle bằng
`load_artifact_bundle()`.

## Đánh giá stability với K cố định

```powershell
.\.venv\Scripts\python.exe -B -u -m experiments.nmf.run_stability
.\.venv\Scripts\python.exe -B -m experiments.summarize_e04_stability
```

Cấu hình `configs/nmf_stability.yaml` giữ K=10 và dùng seed 42, 52, 62, 72,
82. TF-IDF được fit một lần trên Train; NMF được fit lại theo seed và đánh giá
trên Validation. Thí nghiệm không biến đổi hoặc tính metric trên Test.

Kết quả và bằng chứng nằm ở `results/stability/submissions/nmf/`. Runner từ
chối ghi đè thư mục đã tồn tại. Để chạy lại, dùng thư mục mới, ví dụ:

```powershell
.\.venv\Scripts\python.exe -B -u -m experiments.nmf.run_stability --output results/nmf/stability_rerun_01
```

Xem `docs/nmf_stability.md` để đọc metric, cách ghép topic, các giới hạn và
cách trỏ E04 sang kết quả chạy lại. Mô hình Final vẫn dùng seed 42.
