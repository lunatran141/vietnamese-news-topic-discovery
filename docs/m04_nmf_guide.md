# M04 — NMF Topic Modeling

## Mục tiêu

Huấn luyện Non-negative Matrix Factorization (NMF) trên TF-IDF không âm của
Dataset V1, lưu mô hình cùng document-topic matrix `W` và topic-word matrix
`H`. Dataset, input và split được mô tả trong [ghi chú dữ liệu NMF](nmf_data_review.md).

## Phương pháp và cấu hình

Input là cột `text` của `data/processed/p3_no_stopwords.csv`, nối với split
chuẩn bằng `article_id`. TF-IDF dùng unigram/bigram, `min_df=3`, `max_df=0.95`,
tối đa 50.000 feature và sublinear TF. Cấu hình NMF dùng `nndsvda`, coordinate
descent, Frobenius loss, seed 42, tối đa 500 vòng lặp và tolerance `0.0001`.
Tham số nằm trong `configs/nmf.yaml`.

Với ma trận đầu vào `X`, mô hình tìm xấp xỉ `X ≈ W @ H`. `W[d, k]` lưu trọng số
bài–topic; `H[k, v]` lưu trọng số topic–từ. Hàng `W` chuẩn hóa thành tổng 1 để
diễn giải nhiều topic, nhưng giá trị đó là **topic share**, không phải xác suất
nhãn đúng.

## Chạy và đầu ra

M04 dùng stage `finalize` sau khi K được chọn ở M05. Stage fit TF-IDF/NMF trên
train+validation và lưu mô hình cuối:

```powershell
python -m experiments.nmf.run_experiment --stage finalize --n-topics 10
```

Trong `artifacts/nmf/<run_id>/`, stage này tạo:

- `model/nmf_pipeline.joblib` — TF-IDF vectorizer và NMF model.
- `development_document_topics.parquet` — W chuẩn hóa của train+validation.
- `document_topics.parquet` — kết quả test, article metadata và topic shares.
- `topic_word_matrix.parquet` — H theo vocabulary.
- `topic_keywords.json`, `representative_docs.json` và `config.json`.
- `run_metadata.json` — version, split, seed và lineage của run.

Metric cuối và case-study của M06 nằm tại `results/nmf/<run_id>/`.

## Kiểm tra

- Xác nhận input khớp version, hash, schema và ID của split manifest trước khi
  chạy.
- Dùng `article_id` để nối dữ liệu; không ghép theo vị trí dòng.
- Mô hình cuối fit trên train+validation; test chỉ được đánh giá sau khi đã
  chốt K.
- Dùng lại cùng preprocessing P3 trước khi gọi `vectorizer.transform()` cho
  văn bản mới.

Các kiểm thử mô hình NMF nằm trong `tests/test_nmf.py`.
