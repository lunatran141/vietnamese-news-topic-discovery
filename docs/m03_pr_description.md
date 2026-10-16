## Task ID

M03

## Changes

- Xây dựng file cấu hình thực nghiệm `configs/m03_preprocessing_ablation.yaml`: thiết lập kiểm soát biến chặt chẽ (*Ceteris Paribus*) trên cùng tập Train (6,493 bài) và Validation (1,391 bài) từ `data/splits/split_manifest.csv`. Cố định bộ siêu tham số TF-IDF và K-Means ($K=9$ làm mốc chính, $K=7$ làm mốc đối chiếu độ nhạy).
- Thiết lập cơ chế **cô lập biến thể tuyệt đối** (*Technical Isolation Guard*): cấu hình `lowercase: false` và `stopwords_file: null` trong TF-IDF vectorizer để bảo toàn trọn vẹn đặc trưng chữ hoa/chữ thường phân biệt giữa P0 và P1, cũng như bảo toàn ranh giới giữa P2 (chưa lọc từ dừng) và P3 (đã lọc từ dừng).
- Cập nhật module `src/features/tfidf.py`: bổ sung tham số `lowercase` linh hoạt trong hàm `create_tfidf_vectorizer(config)` (mặc định `True` để đảm bảo 100% tương thích ngược với M01/M02).
- Xây dựng script thực nghiệm `experiments/kmeans/run_m03_preprocessing_ablation.py`:
  - Thực thi kiểm thử bóc tách 4 biến thể tiền xử lý tuần tự: $P0 \rightarrow P1 \rightarrow P2 \rightarrow P3$.
  - Đo lường toàn diện các chỉ số: Kích thước từ vựng, độ thưa (Sparsity), tỷ lệ OOV, chỉ số nội tại (Silhouette, Davies-Bouldin, Calinski-Harabasz), chỉ số ngoại vi (ARI, NMI, Purity so với `category_gold` - tuyệt đối Zero Label Leakage), độ ổn định đa seed (5 seeds).
  - Bổ sung các chỉ số chất lượng diễn giải từ vựng (Lexical Interpretability): Tỷ lệ từ ghép ngữ nghĩa trong top từ khóa (`compound_keyword_ratio_pct`), tỷ lệ ô nhiễm từ dừng (`stopword_keyword_ratio_pct`), độ đa dạng chủ đề (`topic_diversity_top10`).
  - Trích xuất dữ liệu định tính chi tiết cho từng cụm ở $K=9$ (`results/kmeans/m03_qualitative_cluster_examples.csv` và `results/kmeans/m03_cluster_keywords_p0_p3.json`).
  - Xuất bản bộ 4 biểu đồ trực quan hóa nghiên cứu đạt chuẩn 300 DPI tại `results/kmeans/figures/`.
- Xây dựng notebook phân tích tương tác `notebooks/06_m03_preprocessing_ablation.ipynb` trình bày trực quan bảng đối chiếu, biểu đồ và dẫn chứng định tính phục vụ báo cáo.
- Xây dựng bộ kiểm thử tự động tại `tests/test_m03_preprocessing_ablation.py`.
- Viết tài liệu nghiên cứu chuyên sâu `docs/m03_preprocessing_ablation_guide.md` giải thích thấu đáo hiện tượng "Nghịch lý chỉ số nội tại" (Geometric Paradox) và trả lời toàn diện câu hỏi nghiên cứu RQ2.

## How to Run and Test

```bash
# 1. Chạy thực nghiệm bóc tách tiền xử lý P0 -> P3 (sinh bảng metrics, dữ liệu định tính và 4 biểu đồ)
python -m experiments.kmeans.run_m03_preprocessing_ablation

# 2. Chạy bộ unit tests cho Task M03
pytest tests/test_m03_preprocessing_ablation.py -v

# 3. Chạy toàn bộ các bài test liên quan của dự án
pytest tests/test_m03_preprocessing_ablation.py tests/test_kmeans.py tests/test_tfidf.py tests/test_splits.py tests/test_freeze_manifest.py tests/test_preprocessing.py tests/test_deduplication.py tests/test_data_quality.py tests/test_dataset_loader.py tests/test_embeddings.py -v
```

## Results

- Tests performed: 100% tests passed.
- Kết quả đối chiếu định lượng trên tập Validation ($K=9$):
  - **P0 (Tối giản - Minimal):** Vocab 17,983 | Purity 83.61% | NMI 0.6652 | ARI 0.5457 | DBI 5.7112 | Từ ghép: 0.0% | Ô nhiễm từ dừng: 42.2% | Từ vựng bị vỡ vụn âm tiết đơn (`phiếu`, `lãi`), lẫn lộn chữ hoa/thường (`USD`, `Việt`).
  - **P1 (Chuẩn hóa cơ bản - Normalized):** Vocab 15,285 | Purity 81.52% | NMI 0.6503 | ARI 0.5161 | DBI 5.5708 | Từ ghép: 0.0% | Ô nhiễm từ dừng: 41.5% | Hợp nhất chữ thường nhưng vẫn vỡ vụn âm tiết đơn và ngập tràn từ dừng.
  - **P2 (Tách từ - Segmented):** Vocab 20,000 | Purity 81.60% | NMI 0.6494 | ARI 0.4958 | DBI 6.1151 | Từ ghép: 57.8% | Ô nhiễm từ dừng: 42.2% | Xuất hiện từ ghép có nghĩa (`bác_sĩ`, `chứng_khoán`, `điều_trị`) nhưng từ dừng tần số cao vẫn làm loãng độ phân cụm.
  - **P3 (Lọc từ dừng - No Stopwords):** Vocab 20,000 | Purity **87.28%** (+5.68%) | NMI **0.7004** (+0.051) | ARI **0.5925** (+0.097) | DBI 6.1566 | Từ ghép: **74.1%** | Ô nhiễm từ dừng: **0.0%** | Độ ổn định đa seed cao (ARI = 0.8224 ± 0.0444).
- Giải quyết thành công yêu cầu giảng viên về *"Trường hợp điểm số tự động cao nhưng topic khó diễn giải"*:
  - P0 và P1 có DBI thấp hơn và CH cao hơn do các âm tiết đơn tần số cao kéo các vector lại gần nhau trong không gian hình học, tạo ảo giác cụm co cụm chặt. Tuy nhiên, các topic này không thể diễn giải được (đa nghĩa, từ dừng, vỡ cụm).
  - P3 loại bỏ từ dừng, giúp phân tách rõ rệt các chủ đề thực sự, đạt Purity cao nhất (87.28%) và 100% từ khóa tâm cụm là danh từ/thuật ngữ chuyên ngành sạch.
- Kiểm tra độ nhạy trên $K=7$ khẳng định xu hướng $P3 > P0, P1, P2$ là hoàn toàn nhất quán và độc lập với số cụm K.
- Artifacts & kết quả:
  - Bảng metrics chính ($K=9$): `results/kmeans/m03_preprocessing_ablation_metrics.csv`
  - Bảng độ nhạy ($K=7$ & $K=9$): `results/kmeans/m03_preprocessing_ablation_k7_sensitivity.csv`
  - Minh chứng định tính từng cụm: `results/kmeans/m03_qualitative_cluster_examples.csv`
  - Từ khóa chi tiết: `results/kmeans/m03_cluster_keywords_p0_p3.json`
  - 4 biểu đồ 300 DPI: `results/kmeans/figures/`
  - Báo cáo kỹ thuật chi tiết: `docs/m03_preprocessing_ablation_guide.md`
  - Notebook tương tác: `notebooks/06_m03_preprocessing_ablation.ipynb`

## Checklist

- [x] I worked on a feature branch (`feature/M03-preprocessing-ablation`).
- [x] I did not push directly to main or develop.
- [x] The code runs with valid input.
- [x] Required tests or checks have passed.
- [x] No password, token, API key, or .env file is committed.
- [x] No personal absolute path is hard-coded.
- [x] Data in data/raw was not modified.
- [x] File `docs/tv1_kmeans_tasks.csv` is excluded via .gitignore and not committed.

