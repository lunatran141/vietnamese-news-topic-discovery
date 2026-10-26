# E07 — Kế hoạch dữ liệu mới cho HDBSCAN

## Phạm vi

- Người phụ trách thu thập và đánh giá HDBSCAN: TV4.
- Thành viên phối hợp rà soát kết quả: TV3.
- Loại đánh giá: dữ liệu khác thời gian so với bộ dữ liệu gốc.
- Khoảng ngày xuất bản: từ 2026-10-01 đến hết 2026-10-15, theo giờ Việt Nam.
- Nguồn: VnExpress, Tuổi Trẻ, VietnamNet.
- Chỉ đánh giá các bài nằm trong khoảng ngày trên.
- Không dùng dữ liệu E07 để chọn lại embedding, seed hoặc tham số HDBSCAN.

## Cấu hình thu thập đã chốt trước khi xem kết quả

- VnExpress: tối đa 25 trang mỗi chuyên mục.
- Tuổi Trẻ: tối đa 40 trang mỗi chuyên mục.
- VietnamNet: tối đa 18 trang mỗi chuyên mục.
- Dùng các crawler hiện có và lưu đầu ra riêng tại `data/ood/oct2026/`.
- Nếu một nguồn thu thập thất bại hoặc không có bài, dừng đánh giá để kiểm tra.

## Kiểm tra trước khi khóa dữ liệu

1. Có bài từ đủ ba nguồn.
2. `published_at` nằm trong khoảng 2026-10-01 đến 2026-10-15 theo giờ Việt Nam.
3. Không có `article_id` hoặc URL trùng trong bộ mới.
4. Không có bài trùng ID hoặc URL với bộ dữ liệu tháng 9.
5. Lưu số bài theo nguồn và theo ngày, đường dẫn file, thời điểm thu thập và SHA-256 vào manifest.

Sau khi khóa bộ dữ liệu và tạo manifest, không thay đổi khoảng ngày,
nguồn hoặc giới hạn trang dựa trên metric HDBSCAN thu được. Nếu cần
thu thập lại vì lỗi crawler, ghi lý do và tạo phiên bản dữ liệu mới.