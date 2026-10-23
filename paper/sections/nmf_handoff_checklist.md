# TV2 — Gói bàn giao phần NMF

## Đã chuẩn bị trong repo

- Bản Methodology/Results: [`nmf_method_results.md`](nmf_method_results.md).
- Hình chọn K: `results/nmf/nmf_p3_select_review_seed42_v1/topic_count_selection.png`, được nhúng tương đối trong bản paper.
- Bảng số liệu chọn K trên Validation: `results/nmf/nmf_p3_select_review_seed42_v1/topic_count_metrics.csv`.
- Kết quả Test NMF: `results/nmf/nmf_p3_k10_seed42_v1/test_metrics.json`.
- Case study ứng viên: `results/nmf/nmf_p3_k10_seed42_v1/multi_topic_case_studies.md` và `multi_topic_cases.csv`.
- Ghi chú nguồn cho Related Work: `paper/notes/tv2_lee_seung_1999_nmf.md` và `paper/notes/tv2_ocallaghan_2015_topic_coherence.md`.

## Cần người trong nhóm hoàn tất

- [ ] Mở bản paper trên GitHub và xác nhận Hình 1 tải lên, dễ đọc, đúng đường dẫn.
- [ ] Hoàn thành `candidate_topic_review.csv`: reviewer điền nhãn, điểm diễn giải, ghi chú trùng lặp/nhiễu và tên reviewer. Hiện các cột đánh giá còn trống; không xem phần này là đã review thủ công.
- [ ] Reviewer độc lập đọc toàn văn năm case trong `multi_topic_case_studies.md`, xác nhận hoặc sửa nhận xét và ghi tên người review. Bản hiện tại ghi nhận nhận xét sơ bộ; chưa phải đánh giá độc lập đã nghiệm thu.
- [ ] Gửi đoạn so sánh NMF–LDA trong phần Results cho Tuấn để thống nhất câu chữ với bảng so sánh chính của paper.
- [ ] Sau khi các mục trên được xử lý, kiểm tra lại citations và số liệu với bản paper tổng hợp trước khi báo hoàn tất.

## Diễn giải cần giữ nhất quán

- K=10 được chọn theo NPMI Validation cao nhất trong các giá trị đã thử; K=12 có sai số tái tạo thấp nhất.
- 523 bài thỏa ngưỡng topic phụ là danh sách ứng viên để kiểm tra, không phải 523 bài đa chủ đề đã được xác nhận.
- NMF và LDA dùng khác biểu diễn đầu vào và số topic; kết quả hiện tại so sánh hai pipeline đã chọn, không cô lập tác động của thuật toán.
- Topic share là trọng số chuẩn hóa theo bài, không phải xác suất đã hiệu chuẩn.

## Xác nhận trước khi push phần cập nhật

```powershell
git status --short
git diff -- paper/sections/nmf_method_results.md paper/sections/nmf_handoff_checklist.md
```

Chỉ stage các file đã rà soát và các artifact NMF cần bàn giao; tránh thêm dữ liệu đầu vào hoặc file không thuộc phần này.
