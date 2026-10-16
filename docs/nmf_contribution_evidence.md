# TV2 — Bằng chứng đóng góp và demo NMF (task 2.11, 2.12)

| Task | Code/commit hiện có | Experiment/artifact | Figure/table | Demo | Báo cáo |
|---|---|---|---|---|---|
| M04 / 2.2 | `5ab373d` — `src/models/nmf.py`, `experiments/nmf/run_experiment.py` | `artifacts/nmf/nmf_p3_k10_seed42_v1/` — model, W/H, keywords, representatives | W/H và 10 topic trong artifact | Mở một topic, từ khóa và 5 bài đại diện | `paper/sections/nmf_method_results.md` — Phương pháp |
| M05 / 2.3–2.4 | `5ab373d`; phần review từng K bổ sung, **chưa có commit/PR** | `results/nmf/nmf_p3_select_review_seed42_v1/`: 40 topic, 200 bài đại diện validation, phiếu review | `topic_count_selection.png`, bảng K–NPMI–diversity–error | Giải thích vì sao đề xuất K=10 và vì sao vẫn cần review bằng người | Cùng file — Kết quả chọn K |
| M06 / 2.5 | `5ab373d`; case study đang bổ sung, **chưa có commit/PR** | `multi_topic_cases.csv`, `multi_topic_case_studies.md` | Bảng share của bài `VNN_93c346ff1800` | Bài “Khám xét nhà nữ diễn viên 37 tuổi”: NMF_02 0,4141 và NMF_05 0,4096; nêu giới hạn của ngưỡng 0,20 | Cùng file — Đánh giá test và giới hạn |
| 2.1 / 2.7 / 2.RW | Tài liệu bổ sung, **chưa có commit/PR** | `docs/nmf_data_review.md`, `topic_display_mapping.json`, `paper/notes/tv2_*.md` | Bảng mapping nhãn tạm | Nêu cách giữ raw topic ID khi đổi tên hiển thị | Notes gửi TV3 tổng hợp R06 |

## Kịch bản 2–3 phút

1. **30 giây:** Dataset V1 và `article_id` chung; NMF tạo W/H và 10 topic.
2. **40 giây:** Mở bảng K và biểu đồ; NPMI chọn K=10, giải thích thêm nhu cầu
   đọc từ khóa/bài đại diện, không dựa một metric.
3. **50 giây:** Mở ví dụ `VNN_93c346ff1800`; giải thích điều tra và giải trí
   cùng xuất hiện, nhưng trọng số chuẩn hóa không là xác suất đúng.
4. **30 giây:** Chỉ vào commit/PR sau khi người dùng tạo PR, artifact và phần
   báo cáo tương ứng trong bảng trên. Không điền PR ID trước khi PR tồn tại.

Trước buổi demo cần TV2 đọc toàn văn ví dụ, reviewer xác nhận nhãn hiển thị,
chụp lại PR/commit cuối và kiểm tra đường dẫn artifact trên máy demo.
