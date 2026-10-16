# TV2 — Xác nhận dữ liệu đầu vào NMF (task 2.1)

## Snapshot và cấu hình

Nguồn chuẩn là `data/processed/dataset_manifest.json` của Dataset V1. Manifest
ghi `dataset_version=v1.0`, `review_status=approved`, `quality_gate=PASSED`.
NMF dùng `data/processed/p3_no_stopwords.csv` với 9.274 bài và SHA-256
`663a34f309ef1f6d8d2dc61a68fd041d5f7aac71aa930dccb74841847a7d6710`.
Khóa nối là `article_id`; đầu vào mô hình là cột `text`. Manifest định nghĩa
`text = title + ' ' + processed_content`, nên đây **không phải** hai cột
`processed_title` và `processed_content` nối lại. P3 kế thừa chuẩn hóa P1,
tách từ PyVi ở P2 và loại 391 stopword. Cấu hình dùng trong
`configs/nmf.yaml`.

Split khóa ở `data/splits/split_manifest.csv`: 6.493 train, 1.391 validation,
1.390 test. Stage `select` fit trên train và so K trên validation. Stage
`finalize` fit lại trên train+validation và đọc test đúng một lần. Không dùng
test để đổi K, ngưỡng đa chủ đề hoặc luật đặt tên.

## Ranh giới khi bàn giao

- `load_dataset_v1()` trong `experiments/nmf/run_experiment.py` so version,
  SHA-256, schema, ID duy nhất và tập ID của split trước khi chạy.
- NMF dùng TF-IDF unigram/bigram. LDA được manifest khuyến nghị P2; BERTopic
  được khuyến nghị P1. Khi so phương pháp, giữ cùng `article_id` và split,
  đồng thời ghi rõ mỗi biến thể tiền xử lý.
- Mô hình đã lưu chỉ nhận văn bản ở đúng dạng P3. TV5 cần tái sử dụng pipeline
  P0→P3 của TV1 khi dự đoán một bài mới; đưa thẳng văn bản thô vào
  `vectorizer.transform()` sẽ không cùng phân phối đầu vào lúc huấn luyện.
- `category_gold` là nhãn chuyên mục hỗ trợ đối chiếu ARI/NMI, không phải nhãn
  chủ đề chuẩn để huấn luyện NMF.

Đây là ghi chú đối chiếu từ manifest và config, không thay thế bước xác nhận
dataset của TV1 nếu snapshot chung thay đổi.
