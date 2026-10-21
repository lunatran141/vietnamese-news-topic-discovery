# 1. MỞ ĐẦU (INTRODUCTION)

## 1.1. Bối cảnh và động lực nghiên cứu

Sự phát triển của báo chí điện tử khiến số lượng bài viết tiếng Việt ngày càng gia tăng, tạo ra nhu cầu tổ chức và khai thác thông tin hiệu quả. Việc phân loại bài báo theo chuyên mục có sẵn chưa phản ánh đầy đủ các chủ đề hoặc sự kiện cụ thể trong nội dung. Chẳng hạn, các bài thuộc chuyên mục thể thao có thể đề cập đến nhiều giải đấu, đội tuyển và sự kiện khác nhau. Vì vậy, khám phá chủ đề tự động (*topic discovery*) là hướng tiếp cận hữu ích nhằm nhóm các bài viết tương đồng, hỗ trợ tìm kiếm thông tin và nhận diện các nhóm nội dung nổi bật mà không cần nhãn chủ đề chi tiết.

## 1.2. Phát biểu bài toán

Nghiên cứu giải quyết bài toán khám phá chủ đề không giám sát trên tập tin tức tiếng Việt. Đầu vào là tiêu đề và nội dung bài báo; đầu ra gồm các cụm hoặc chủ đề được phát hiện, từ khóa đặc trưng, bài báo đại diện và thông tin về các bài bất thường nếu phương pháp hỗ trợ phát hiện outlier.

Nhóm khảo sát năm phương pháp: TF-IDF kết hợp K-Means, Non-negative Matrix Factorization (NMF), Latent Dirichlet Allocation (LDA), sentence embeddings kết hợp HDBSCAN và BERTopic. Các phương pháp được thực nghiệm trên cùng phiên bản dữ liệu và giao thức chia Train/Validation/Test cố định. Nhãn `category_gold` chỉ được sử dụng làm chuyên mục tham chiếu để đánh giá mức độ tương ứng với kết quả phân cụm, không được xem là nhãn chủ đề chuẩn.

## 1.3. Câu hỏi nghiên cứu

Nghiên cứu tập trung trả lời bốn câu hỏi:

**RQ1:** Các phương pháp K-Means, NMF, LDA, HDBSCAN và BERTopic khác nhau như thế nào về chất lượng khám phá chủ đề, khả năng diễn giải, độ ổn định và chi phí tính toán?

**RQ2:** Các mức tiền xử lý văn bản P0–P3 ảnh hưởng như thế nào đến chất lượng phân cụm của TF-IDF kết hợp K-Means?

**RQ3:** Việc sử dụng riêng tiêu đề (*Title-only*) và kết hợp tiêu đề với nội dung (*Title+Content*) tạo ra sự khác biệt như thế nào về chất lượng phân cụm?

**RQ4:** Biểu diễn ngữ nghĩa bằng sentence embeddings mang lại những lợi ích và hạn chế gì so với biểu diễn văn bản dựa trên từ vựng trong bài toán khám phá chủ đề?

Các câu hỏi được liên kết với thực nghiệm như sau:

| Câu hỏi | Task thực nghiệm | Kết quả dùng để trả lời |
|---|---|---|
| RQ1 | E03 | Bảng so sánh năm phương pháp, kết hợp các chỉ số và phân tích khả năng diễn giải |
| RQ2 | M03 | Kết quả preprocessing ablation P0–P3 trên K-Means |
| RQ3 | M03A | Kết quả so sánh Title-only và Title+Content |
| RQ4 | E02, E03 | Kết quả HDBSCAN, BERTopic đối chiếu với các phương pháp từ vựng |

Các kết quả được phân tích trong phạm vi những cấu hình và thước đo đã thực nghiệm. Khi điều kiện tiền xử lý, biểu diễn văn bản hoặc số lượng chủ đề khác nhau, nghiên cứu không quy toàn bộ chênh lệch cho riêng thuật toán. Các tiêu chí chưa có kết quả đánh giá đầy đủ sẽ được nêu rõ trong phần hạn chế.

## 1.4. Mục tiêu và đóng góp

Mục tiêu của nghiên cứu là xây dựng quy trình khám phá chủ đề tin tức tiếng Việt có khả năng tái lập và khảo sát sự khác biệt giữa các phương pháp biểu diễn từ vựng, mô hình hóa chủ đề và phân cụm ngữ nghĩa.

Các đóng góp chính bao gồm: (1) xây dựng tập dữ liệu thống nhất gồm 9.274 bài báo tiếng Việt sau khử trùng lặp và giao thức Train/Validation/Test cố định; (2) triển khai năm pipeline gồm K-Means, NMF, LDA, HDBSCAN và BERTopic; (3) thực hiện các thí nghiệm về ảnh hưởng của tiền xử lý P0–P3 và đầu vào Title-only so với Title+Content; (4) đánh giá kết quả bằng các thước đo phù hợp với từng phương pháp, kết hợp phân tích từ khóa và bài báo đại diện; và (5) xây dựng các artifacts phục vụ tổng hợp kết quả và trực quan hóa trên ứng dụng minh họa.

Các hướng mở rộng như phát hiện sự kiện, theo dõi xu hướng và đánh giá trên dữ liệu ngoài phân phối được xem là định hướng phát triển khi chưa có thực nghiệm hoàn chỉnh.

## 1.5. Cấu trúc bài báo

Phần còn lại của bài báo được tổ chức như sau. Mục 2 tổng hợp các nghiên cứu liên quan. Mục 3 trình bày tập dữ liệu và quy trình tiền xử lý. Mục 4 giới thiệu các phương pháp khám phá chủ đề. Mục 5 mô tả thiết lập thực nghiệm và các chỉ số đánh giá. Mục 6 trình bày kết quả và thảo luận. Các mục tiếp theo phân tích lỗi, hạn chế nghiên cứu và hệ thống minh họa. Cuối cùng, bài báo tổng kết những phát hiện chính và đề xuất hướng phát triển trong tương lai.