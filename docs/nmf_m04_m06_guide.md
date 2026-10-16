# TV2 — Hướng dẫn M04, M05 và M06

## Xác nhận dữ liệu đầu vào (task 2.1)

- Dataset: `v1.0`, trạng thái `approved`, quality gate `PASSED`.
- Input NMF chính: `data/processed/p3_no_stopwords.csv`.
- Cột mô hình: `text`; khóa liên kết xuyên pipeline: `article_id`.
- Snapshot P3: 9.274 bài, SHA-256
  `663a34f309ef1f6d8d2dc61a68fd041d5f7aac71aa930dccb74841847a7d6710`.
- Tokenizer: PyVi 0.1.1. Danh sách stopword có 391 từ.
- Split chính thức: `data/splits/split_manifest.csv`, gồm `train`, `val`, `test`.
- Train dùng để fit candidate; validation dùng để chọn K; test bị khóa đến stage
  `finalize`.

Runner tự kiểm tra dataset version, trạng thái duyệt, hash P3, schema, ID duy
nhất và việc Dataset V1 có cùng tập `article_id` với split manifest. Nếu một
điều kiện sai, chương trình dừng với lỗi có ngữ cảnh.

## M04 — NMF pipeline

NMF phân rã ma trận TF-IDF không âm `X` thành `W @ H`:

- `W[d, k]`: trọng số topic `k` trong bài `d`;
- `H[k, v]`: trọng số từ `v` trong topic `k`.

Mô hình dùng unigram và bigram, seed 42 và khởi tạo `nndsvda`. Tất cả tham số
nằm trong `configs/nmf.yaml`. Identifier dùng tiếng Anh; docstring và comment
trong module NMF dùng tiếng Việt theo quy ước nhóm.

## M05 — Chọn số topic

Chạy candidate trên train và đánh giá trên validation:

```powershell
python -m experiments.nmf.run_experiment --stage select
```

Các K mặc định: 4, 6, 8, 10, 12. Bảng kết quả báo:

- NPMI coherence ↑;
- topic diversity ↑;
- normalized reconstruction error ↓;
- mean primary topic share;
- số vòng lặp và trạng thái hội tụ.

Runner đề xuất K bằng NPMI, rồi dùng diversity, reconstruction error và K nhỏ
hơn để phá hòa. Stage `select` còn xuất từ khóa, trọng số topic và bài đại diện
trên validation cho **từng K** để nhóm chấm độ dễ hiểu trước khi chốt. File
`candidate_topic_review.csv` là phiếu review trống; không điền điểm tự động.
Stage này không đọc test set. Run `nmf_p3_select_seed42_v1` trước đây chỉ có
metric; muốn có bộ review cho từng K cần chạy lại với run ID mới.

Run review đã tạo là `nmf_p3_select_review_seed42_v1`. Phiếu review có 40 dòng,
mỗi dòng ứng với một topic trong một K. Người chấm điền tên, nhãn gợi ý, điểm
1–5 và ghi chú chồng lấn/nhiễu sau khi xem file bài đại diện. Quy ước điểm:
1 = không đặt tên được; 2 = nhiều từ/bài không cùng nghĩa; 3 = có chủ đề nhận
ra nhưng lẫn bài; 4 = phần lớn từ và bài cùng nghĩa; 5 = từ khóa và bài đại
diện nhất quán, ranh giới với topic khác rõ. Cần ít nhất hai người chấm độc
lập và ghi khác biệt trước khi nhóm chốt K. Không nhập điểm từ thuật toán.

## M06 — Phân tích bài đa chủ đề

Sau khi review bảng/figure của M05, chạy stage cuối với K đã chốt:

```powershell
python -m experiments.nmf.run_experiment --stage finalize --n-topics 10
```

Stage cuối refit TF-IDF và NMF trên `train+val`, sau đó đánh giá đúng một lần
trên `test`. Bài đa chủ đề được nhận diện khi ngoài topic chính còn topic phụ có
share từ 0,20 trở lên. Bảng case study ưu tiên bài có gap giữa hai topic cao
nhất nhỏ và entropy cao. Đây là tiêu chí sàng lọc để con người giải thích, không
phải nhãn ground truth.

## Output contract

Stage `select` tạo:

- `results/nmf/<run_id>/topic_count_metrics.csv`;
- `results/nmf/<run_id>/topic_count_selection.png`;
- `results/nmf/<run_id>/candidate_topic_summary.csv` (top words, NPMI từng
  topic, mean share, số bài có topic chính);
- `results/nmf/<run_id>/candidate_representative_docs.csv` (5 bài validation
  tiêu biểu cho mỗi topic của mỗi K);
- `results/nmf/<run_id>/candidate_topic_review.csv` (phiếu chấm thủ công);
- `artifacts/nmf/<run_id>/validation_document_topics_kXX.parquet` (phân phối
  topic của validation theo từng K);
- `artifacts/nmf/<run_id>/config.json` và `run_metadata.json`.

Stage `finalize` tạo:

- `model/nmf_pipeline.joblib`;
- `development_document_topics.parquet` (W của train+val);
- `document_topics.parquet` (phân phối topic của test) và
  `topic_word_matrix.parquet` (H);
- `topic_keywords.json` và `representative_docs.json`;
- `config.json`, `run_metadata.json`;
- `test_metrics.json` và `multi_topic_cases.csv` trong `results/nmf/<run_id>`.

Mỗi topic có 10 từ khóa và 5 bài tiêu biểu. `manual_label`, điểm
interpretability và review note được để trống để nhóm đánh giá thủ công dựa trên
cả từ khóa và bài đại diện. Bảng multi-topic có trích đoạn thật và cột
`manual_analysis_note` để TV2 ghi giải thích case study.

## Bàn giao cho TV3/TV5

- Dùng `article_id` để nối với Dataset V1 và output các phương pháp khác.
- `document_topics.parquet` của run cuối chỉ chứa test; `development_document_topics.parquet`
  chứa train+val. Không ghép hai file mà quên cột `split`.
- `topic_display_mapping.json` là nhãn hiển thị có thể đổi sau review; ID `NMF_XX`
  và ma trận W/H gốc không đổi. Không dùng nhãn hoặc gộp topic hậu kiểm để tính
  lại metric trên test.
- Văn bản mới đưa vào NMF phải dùng đúng biến thể P3 như Dataset V1, rồi mới
  gọi vectorizer và mô hình trong `model/nmf_pipeline.joblib`.
