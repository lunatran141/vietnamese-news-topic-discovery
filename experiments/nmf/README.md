# NMF experiments (TV2 — M04, M05, M06)

Module này triển khai NMF trên Dataset V1 đã duyệt, chọn số topic bằng
validation và xuất case study bài đa chủ đề trên test.

Chạy chọn K:

```powershell
python -m experiments.nmf.run_experiment --stage select
```

Review `topic_count_metrics.csv` và `topic_count_selection.png`, sau đó chạy K
đã chốt:

```powershell
python -m experiments.nmf.run_experiment --stage finalize --n-topics 8
```

Không dùng stage `finalize` để thử nhiều K vì stage này đọc test set. Cấu hình,
seed, dữ liệu và output được mô tả tại `docs/m04_m06_nmf_guide.md`.
