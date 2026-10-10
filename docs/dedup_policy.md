# Chính sách khử trùng lặp (Deduplication Policy)

> Tài liệu tổng kết chính sách và kết quả kiểm toán trùng lặp dữ liệu — Task D02.

## 1. Tổng quan

Pipeline khử trùng lặp 2 tầng được áp dụng trên tập dữ liệu `data/interim/news_articles_validated.csv` (9,304 bài viết).

| Chỉ số | Giá trị |
|---|---|
| Tổng bài trước dedup | 9,304 |
| Tổng bài sau dedup | 9,258 |
| Exact duplicate loại bỏ | 13 |
| Near duplicate loại bỏ | 33 |
| **Tổng loại bỏ** | **46** |
| **Tỷ lệ loại bỏ** | **0.49%** |
| Candidate pairs (near-dup) | 2,098 |

## 2. Tầng 1: Exact Duplicate

### 2.1 Chuẩn hóa văn bản
- Chuyển bảng mã Unicode NFC
- Chuyển chữ thường (lowercase)
- Loại bỏ toàn bộ dấu câu
- Gộp khoảng trắng thừa và cắt khoảng trắng hai đầu

### 2.2 URL normalization
- Bỏ scheme (http/https), tiền tố `www.`, các tham số theo dõi (query params) và dấu gạch chéo cuối.

### 2.3 Ba khóa so sánh tuần tự
1. `exact_url`: Trùng URL sau chuẩn hóa.
2. `exact_content`: Trùng SHA-256 hash của nội dung chuẩn hóa.
3. `exact_title`: Trùng tiêu đề chuẩn hóa.

Sau mỗi bước, các bài đã bị loại không tham gia vào các bước kế tiếp để tránh rò rỉ cụm trùng lặp (transitive leakage).

## 3. Tầng 2: Near Duplicate

### 3.1 Sinh cặp ứng viên (Candidate Generation)
- Phương pháp: MinHash LSH (`datasketch`)
- Đặc trưng: Tập từ vựng (Word tokens) của nội dung chuẩn hóa
- Số hàm băm: 128 permutations
- Ngưỡng LSH: 0.70

### 3.2 Đánh giá kép (Dual-Scoring)
- `title_similarity`: Jaccard trên tập từ vựng của tiêu đề.
- `content_similarity`: Jaccard ước lượng từ MinHash signature.

### 3.3 Ma trận quyết định (Decision Matrix)
- `remove`: `content_sim >= 0.85` HOẶC (`title_sim >= 0.60` VÀ `content_sim >= 0.70`).
- `manual_review`: `0.80 <= content_sim < 0.85` (vùng biên nhạy cảm, mặc định giữ cả hai).
- `keep_both`: `content_sim < 0.80` (giữ cả hai bài).

### 3.4 Transitive Closure với Union-Find
Các cặp bài có quyết định `remove` được gom cụm bằng cấu trúc Union-Find, mỗi cụm chỉ chọn 1 bài đại diện duy nhất theo Tie-breaking rule.

## 4. Thứ tự ưu tiên giữ bài (Tie-breaking)
1. Ưu tiên 1: Bài có nội dung dài hơn (`len(content)`).
2. Ưu tiên 2: Bài có ngày đăng hợp lệ (`published_at`).
3. Ưu tiên 3: Bài có `article_id` nhỏ hơn (ổn định, tái lập được).
4. Tuyệt đối KHÔNG ưu tiên dựa trên nhãn `category_gold`.

## 5. Giải thích quyết định `keep_both`
Các cặp bài có độ tương đồng nội dung thấp hơn 0.85 được giữ lại cả hai vì:
- Cùng đưa tin về một sự kiện thời sự nhưng cách hành văn và góc nhìn độc lập.
- Bảo toàn tính đa dạng ngôn ngữ cho mô hình chủ đề (topic model).

## 6. Hạn chế đã biết
- Bài đa chuyên mục: Khi bài bị loại, thông tin `category_gold` của nó không được bảo toàn. Chấp nhận vì `category_gold` chỉ đóng vai trò ground truth khi đánh giá phân loại.
- Sai số ước lượng MinHash: Sai số tiêu chuẩn xấp xỉ 1 / sqrt(128) ≈ 0.0884.

## 7. Biểu đồ minh họa

### 7.1 Phân bố Similarity
![Similarity Distribution](../results/dedup_data/figures/similarity_distribution.png)

### 7.2 Trước và Sau Dedup
![Before After](../results/dedup_data/figures/dedup_before_after.png)

### 7.3 Số lượng và Tỷ lệ Trùng lặp theo Chuyên mục
Chuyên mục Kinh doanh ghi nhận số lượng bài trùng lặp cao nhất (23 bài, tỷ lệ 1.15%), theo sau là Pháp luật (9 bài, 0.76%) và Giáo dục (6 bài, 0.58%). Ngược lại, chuyên mục Giải trí có 0 bài trùng lặp.
![Duplicates by Category](../results/dedup_data/figures/duplicates_by_category.png)

### 7.4 Ma trận Trùng lặp giữa các Nguồn Báo (Cross-Source)
Trùng lặp chủ yếu diễn ra giữa các nguồn báo khác nhau (đặc biệt giữa Tuổi Trẻ và VietNamNet với các thông cáo báo chí tài chính / doanh nghiệp) hơn là trùng lặp nội bộ cùng một báo.
![Cross-Source Heatmap](../results/dedup_data/figures/cross_source_heatmap.png)
