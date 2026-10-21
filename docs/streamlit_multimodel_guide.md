# Streamlit đa mô hình: cách chạy và kiểm tra

Nhánh triển khai: `feature/P02-P09-multimodel-streamlit`. Giao diện đọc các
`manifest.json` trong `artifacts/common/<method>/<run>/` để phát hiện run.
Không cần sửa danh sách mô hình bằng tay khi thêm run mới đúng schema.

## Bước 1 — Vào repo và bật môi trường Python

Trong PowerShell:

```powershell
cd C:\Users\Ngan\Documents\NLP\vietnamese-news-topic-discovery
.\.venv\Scripts\Activate.ps1
```

Nếu chưa có `.venv`, làm theo README của repo để tạo môi trường và cài
`requirements.txt`.

## Bước 2 — Kiểm tra artifact và test dữ liệu

```powershell
python -m pytest tests\test_streamlit_data_service.py -q
```

Mong đợi: 8 test pass. Năm run BERTopic, HDBSCAN, KMeans, LDA và NMF phải
xuất hiện trong bộ chọn. Nếu thiếu một run, kiểm tra file
`artifacts/common/<method>/<run>/manifest.json` và các đường dẫn trong đó.

## Bước 3 — Chạy giao diện

```powershell
python -m streamlit run app\streamlit_app.py
```

Mở URL `http://localhost:8501`. Ở sidebar, chọn trang, mô hình/run và tập
`train`, `val`, `test` hoặc “Tất cả có sẵn”. Run nào không có split thì sẽ
không hiện split đó.

## Bước 4 — Xem và demo từng trang

1. **Tổng quan:** xem số bài, topic, outlier và bảng topic của split đang chọn.
   Các số đếm trên thẻ được tính từ `document_topics.csv`; metric run là giá
   trị tác giả báo cáo, có thể thuộc phạm vi đánh giá khác.
2. **Khám phá topic:** chọn topic để xem từ khóa, bài đại diện nếu run xuất
   `representative_docs.csv`, và danh sách bài. Nếu file đại diện chưa có,
   trang chỉ gợi ý bài theo điểm gán topic/khoảng cách tới tâm.
3. **Tìm kiếm & lọc:** tìm tiêu đề hoặc article ID, lọc topic, nguồn và ngày.
   Có thể tải toàn bộ kết quả CSV.
4. **Bản đồ 2D:** BERTopic dùng `embedding_projection.csv` sẵn có;
   HDBSCAN/LDA/NMF dùng `projection_2d.csv`. Chọn topic và rê chuột để xem
   bài. HDBSCAN có tọa độ train/val chung; KMeans chưa có bản đồ.
5. **Xu hướng:** chọn topic và ngày/tuần. Đây là số bài trong corpus, không
   phải phép đo mức độ phổ biến của cả thị trường.
6. **So sánh mô hình:** ưu tiên chọn cùng một split. HDBSCAN, BERTopic và
   KMeans có train và val; LDA/NMF có thêm test. Không dùng
   “Tất cả có sẵn” để kết luận định lượng giữa các mô hình.
7. **Thí nghiệm:** xem hình/bảng kết quả đã xuất của từng người.
8. **Phân tích lỗi:** xem outlier, topic lớn bất thường, bài đa chủ đề
   hoặc hai topic có tỷ trọng gần nhau.
   Các nhóm này là tín hiệu để review, chưa tự động là lỗi thật.
9. **Đổi tên & gộp:** tạo `display_name` hoặc `display_group_id` rồi tải
   mapping JSON. Có thể nạp lại JSON cùng run. Nhóm chỉ thay cách trình bày;
   không sửa assignment/model gốc.
10. **Bài báo mới:** UI đã sẵn. Trang cần API nội bộ `/predict` và model
    inference chạy trên bài chưa thấy; hiện API repo mới có `/health`.
11. **Phương pháp & đóng góp:** tóm tắt artifact và vai trò của TV1–TV5.

## Bước 5 — Những việc còn phải bàn giao

- TV4 + TV5: hoàn thiện inference dùng model/artifact đã lưu và API
  `GET /topics`, `POST /predict`, validate input; không huấn luyện lại khi
  người dùng gửi bài.
- Các thành viên: chuẩn hóa `projection_2d.csv` và `representative_docs.csv`
  cho những run còn thiếu nếu muốn bản đồ/bài đại diện đầy đủ 5 phương pháp.
  HDBSCAN đã xuất cả hai file này ở PR #51; bài đại diện trên UI được lọc
  theo split đang chọn.
- Cả nhóm: chốt phạm vi split và định nghĩa metric cho bảng so sánh E03;
  hoàn thành human evaluation/OOD/runtime nếu đưa lên giao diện như kết quả.
- TV5: sau khi API hoạt động, kiểm thử `docker compose up --build` trên máy
  sạch và demo bài báo mới từ đầu đến cuối.

## Bước 6 — Lưu nhánh lên GitHub sau khi tự xem giao diện

```powershell
git status
git add app\data_service.py app\streamlit_app.py .streamlit\config.toml requirements.txt tests\test_streamlit_data_service.py docs\streamlit_multimodel_guide.md
git commit -m "feat(P02-P09): add multi-model Streamlit exploration"
git push -u origin feature/P02-P09-multimodel-streamlit
```

Trước khi commit nên chạy lại `python -m pytest` và chụp màn hình các trang
chính cho PR/demo.
