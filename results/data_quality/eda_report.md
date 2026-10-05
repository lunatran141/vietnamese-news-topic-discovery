# Báo cáo Chất lượng Dữ liệu & Khám phá (EDA) - Task D01

- **Dataset:** `data/raw/news_articles.csv`
- **Tổng số bản ghi:** 9,304 bài viết
- **Kích thước file:** 40.54 MB | **SHA256:** `548b35953368857373e8f26639531b96ab83b742ed6432bff0b6462661b5077a`
- **Commit gốc:** `5c215c0130e00c7f80ffad021a7838fdd2ddc373`
- **Kết luận nghiệm thu:** **PASS_WITH_WARNINGS**

## 1. Kết quả kiểm tra Schema & Hợp lệ (Tầng 1)
- Đủ đúng 7 cột bắt buộc: `article_id, title, content, published_at, source, url, category_gold`.
- Không có ô rỗng hoặc trùng lặp ở `article_id`.
- 100% ngày đăng hợp lệ thuộc tháng 09/2026.
- Đầy đủ 3 nguồn báo (`vnexpress`, `tuoitre`, `vietnamnet`) và 7 nhãn chuyên mục chuẩn.

## 2. Phân bố Dữ liệu (Tầng 2)
- **Nguồn báo:** Tuổi Trẻ (40.61%), VnExpress (31.91%), VietnamNet (27.48%).
- **Chuyên mục:** Nhiều nhất là Kinh doanh (1,995 bài), ít nhất là Khoa học công nghệ (702 bài). Tỷ lệ mất cân bằng: 2.84x.
- **Độ dài bài viết:** Trung vị độ dài nội dung là 658 từ (min: 50, max: 5,885 từ).

## 3. Nhiễu Văn bản & Boilerplate (Tầng 3)
- Phát hiện 1,940 bài chứa cụm từ 'chia sẻ', 1,263 bài chứa 'theo dõi'.
- 941 bài chứa chú thích media (ảnh, video), 276 bài nhắc đến quảng cáo.
- 180 bài chứa URL thô và 62 bài chứa email trong nội dung bài viết.

## 4. Trùng lặp & Nguy cơ Rò rỉ (Tầng 4)
- 0 bài trùng URL hoặc trùng nội dung hoàn toàn.
- Phát hiện **12 bài viết trùng tiêu đề chính xác** và **13 bài viết gần trùng tiêu đề**. Cần loại bỏ ở Task D02 trước khi chia tập Train/Test để tránh rò rỉ thông tin.

## 5. Đề xuất cho Task D02 & D03
1. **D02:** Khử trùng lặp dựa trên nhóm tiêu đề chuẩn hóa (25 bài); ưu tiên giữ bản ghi có nội dung dài và chi tiết hơn.
2. **D03 (P0):** Bóc tách thẻ HTML tag còn sót lại (1 bài).
3. **D03 (P1):** Loại bỏ boilerplate kêu gọi chia sẻ/theo dõi ở cuối bài và thay thế các URL thô bằng token `<URL>`.
