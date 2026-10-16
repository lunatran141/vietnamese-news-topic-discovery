# TV2 — Quy tắc đặt tên và ánh xạ topic (task 2.7)

Artifact `artifacts/nmf/nmf_p3_k10_seed42_v1/topic_display_mapping.json`
chỉ dành cho nhãn **hiển thị**. `topic_id` trong ma trận W/H và trong
`document_topics.parquet` luôn giữ nguyên. Mỗi nhãn dựa trên top words và 5 bài
đại diện; trạng thái hiện là `provisional`, chờ TV2 và ít nhất một người khác
đọc bài và chấm độc lập.

Quy trình review:

1. Đọc từ khóa và trọng số trong `topic_keywords.json` cùng 5 bài trong
   `representative_docs.json`. Đọc thêm bài gốc khi trích đoạn không đủ rõ.
2. Ghi nhãn ngắn, mô tả ranh giới và ví dụ bài không thuộc chủ đề. Chấm độ dễ
   hiểu từ 1–5 trên phiếu `candidate_topic_review.csv` của run validation.
3. Kiểm tra các cặp có thể chồng lấn: NMF_01/NMF_06/NMF_07 là các khía cạnh
   thể thao; NMF_00/NMF_09 là kinh doanh và thị trường tài chính.
4. Chỉ đề xuất gộp qua `display_group_id` sau khi review. Gộp để hiển thị không
   làm đổi K=10, topic share, metric hoặc artifact mô hình gốc. Nếu muốn thay
   K/mô hình vì phát hiện topic dư thừa, phải quay lại validation và tạo run
   mới; không dùng test để chọn lại.

TV5 có thể đọc `topic_id -> display_label` để hiển thị trong Streamlit/API.
Nếu chưa duyệt, giao diện nên ghi rõ “nhãn tạm” hoặc hiển thị ID và từ khóa.
Không suy diễn `display_group_id` là kết quả đánh giá mô hình.
