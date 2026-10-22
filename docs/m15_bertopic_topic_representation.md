# M15 — BERTopic Topic Representation and Naming

## 1. Mục tiêu

Task M15 chuyển kết quả thí nghiệm BERTopic của M14 thành các artifact
có thể dùng cho phân tích, review thủ công và giao diện Streamlit.

Các đầu ra chính gồm:

- từ khóa và trọng số c-TF-IDF của từng topic;
- các bài báo đại diện;
- nhãn hiển thị được đề xuất;
- kết quả gán topic cho từng tài liệu;
- thông tin riêng cho các bài outlier.

## 2. Run được lựa chọn

Run được sử dụng:

`e1_vietnamese_sbert`

Embedding model:

`keepitreal/vietnamese-sbert`

Run này được chọn vì có đầy đủ artifact cần thiết và đạt tỷ lệ outlier
trên validation thấp nhất trong các thí nghiệm M14 đã hoàn thành.

Kết quả chính:

| Chỉ số | Giá trị |
|---|---:|
| Số topic | 96 |
| Train outlier rate | 0.2461 |
| Validation outlier rate | 0.3587 |
| Topic diversity | 0.6813 |
| Runtime pipeline mới nhất | 1086.07 giây |

Việc lựa chọn không sử dụng test set. Test set vẫn được khóa theo protocol
D05.

## 3. Topic keywords

Mỗi topic lưu tối đa 10 từ khóa có trọng số c-TF-IDF cao nhất.

Topic `-1` không được đưa vào danh sách keyword thông thường vì đây là
nhóm outlier của BERTopic.

Artifact:

`artifacts/bertopic/best_run/topic_keywords.json`

## 4. Representative documents

Mỗi topic lưu tối đa 5 bài có `topic_probability` cao nhất.

Mỗi bài đại diện gồm:

- `article_id`;
- tiêu đề;
- split;
- xác suất thuộc topic;
- chuyên mục gốc nếu có.

Các bài thuộc topic `-1` không được chọn làm representative document.

Artifact:

`artifacts/bertopic/best_run/representative_docs.json`

## 5. Topic labels

Nhãn gợi ý ban đầu được tạo từ ba keyword đứng đầu của topic. Sau đó,
96 nhãn dự thảo có ý nghĩa hơn được bổ sung tại:

`configs/bertopic_topic_labels.yaml`

Nhãn dự thảo được đọc từ keyword và năm bài đại diện của từng topic.

Các nhãn dự thảo có trạng thái `pending_human_review`. Trạng thái này
không thay thế human evaluation: người phụ trách vẫn phải kiểm tra và
E05 cần ít nhất hai người đánh giá độc lập trước khi đổi thành
`approved`.

Nhãn chỉ phục vụ hiển thị. Raw topic ID do BERTopic sinh ra luôn được giữ
nguyên.

Topic `-1` sử dụng nhãn:

`Outlier / Chưa xác định`

Artifact:

`artifacts/bertopic/best_run/topic_display_mapping.json`

## 6. Human review

Bảng hỗ trợ review được xuất tại:

`results/bertopic_m15/topic_review.csv`

Hiện bảng có đủ 96 nhãn dự thảo. Các trường review gồm:

- nhãn đề xuất và nhãn cuối;
- keyword;
- tiêu đề đại diện;
- chất lượng keyword;
- chất lượng representative documents;
- topic rộng;
- topic trùng;
- ghi chú và trạng thái review.

Để hoàn thành E05, cần ít nhất hai người đánh giá độc lập. Không ghi nhận
Codex như một human rater.

Không merge topic trực tiếp trong M15. Các đề xuất rename hoặc merge được
lưu riêng để sử dụng ở task P07.

## 7. Artifacts

Thư mục `artifacts/bertopic/best_run/` gồm:

- `config.json`;
- `document_topics.parquet`;
- `representative_docs.json`;
- `run_metadata.json`;
- `topic_display_mapping.json`;
- `topic_keywords.json`.

Common artifact dùng cho Streamlit còn nhận thêm:

- `representative_docs.csv`;
- `topic_display_mapping.json`;
- `topic_summary.csv` với cả `topic_name` đã review và `raw_topic_name`.

Model BERTopic gốc vẫn được giữ tại:

`artifacts/bertopic/experiments/e1_vietnamese_sbert/model`

M15 không sao chép model nhằm tránh lưu trùng artifact dung lượng lớn.

## 8. Cách chạy lại

Từ thư mục gốc của repository:

```powershell
.\.venv\Scripts\python.exe -m experiments.bertopic.build_m15_artifacts
