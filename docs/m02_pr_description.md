## Task ID

M02

## Changes

- Xây dựng module thuật toán `src/models/kmeans.py`: huấn luyện mô hình K-Means, dự đoán và tính khoảng cách Euclid chính xác đến tâm cụm, đo lường các chỉ số nội tại (Silhouette, Davies-Bouldin, Calinski-Harabasz, Inertia) và ngoại vi (ARI, NMI, Purity), trích xuất từ khóa tâm cụm và bài báo tiêu biểu gần tâm nhất.
- Xây dựng script thực nghiệm `experiments/kmeans/run_m02_kmeans.py`: thực hiện khảo sát số cụm K (K-search) trên tập Validation với dải $K \in [3, 5, 7, 9, 11, 14, 18]$ (trong đó $K = 7$ là mốc đối chiếu trực tiếp tương ứng 7 nhãn chuyên mục `category_gold` gốc của corpus); tự động chọn cấu hình tối ưu $K = 9$ dựa trên đánh giá đa chiều (chỉ số nội tại, tính ổn định đa seed, kích thước cụm và đánh giá định tính thủ công).
- Xây dựng công cụ đánh giá định tính thủ công `experiments/kmeans/build_manual_evaluation.py`: trích xuất song song bài báo gần tâm (Train) và bài báo ngẫu nhiên (Validation) cho các mốc $K \in [7, 9, 11]$, xuất báo cáo `results/kmeans/kmeans_manual_evaluation.csv` và `results/kmeans/kmeans_manual_evaluation_drafts.json` phục vụ chuyên gia rà soát.
- Đảm bảo nguyên tắc **Zero Label Leakage**: `category_gold` chỉ dùng để đo ARI, NMI, Purity sau khi phân cụm trên tập Validation; tuyệt đối không đưa vào huấn luyện.
- Xuất đầy đủ 6 artifacts chuẩn Handbook Mục E1 tại `artifacts/kmeans/` (`kmeans_model.joblib`, `document_topics.csv`, `topic_keywords.json`, `representative_docs.json`, `config.json`, `run_metadata.json`).
- Tạo 5 biểu đồ trực quan hóa nghiên cứu tại `results/kmeans/figures/` (Elbow, Silhouette vs DBI, Silhouette Analysis từng mẫu, Cluster size distribution, 2D TruncatedSVD).
- Tạo notebook trực quan tương tác `notebooks/05_kmeans_analysis.ipynb` phục vụ demo, phân tích ma trận chéo và đánh giá đóng góp cá nhân (Task 1.16).
- Bổ sung bộ kiểm thử 13 bài test tại `tests/test_kmeans.py`.

## How to Run and Test

```bash
# 1. Chạy pipeline K-Means (K-search + huấn luyện K=9 + xuất artifacts + vẽ biểu đồ)
python -m experiments.kmeans.run_m02_kmeans

# 2. Sinh bảng dữ liệu đánh giá định tính thủ công (K=7, 9, 11)
python -m experiments.kmeans.build_manual_evaluation

# 3. Chạy bộ unit tests cho K-Means
pytest tests/test_kmeans.py -v

# 4. Chạy toàn bộ các bài test liên quan của dự án
pytest tests/test_kmeans.py tests/test_freeze_manifest.py tests/test_splits.py tests/test_tfidf.py tests/test_preprocessing.py tests/test_deduplication.py tests/test_data_quality.py tests/test_dataset_loader.py tests/test_embeddings.py -v
```

## Results

- Tests performed: 13/13 tests K-Means passed; 78/78 bài test toàn dự án passed 100%.
- Cấu hình tối ưu được chọn: **K = 9** (vượt trội so với mốc đối chiếu $K = 7$ về độ thuần và phân tách chủ đề, đồng thời ổn định và chặt chẽ hơn $K = 11$):
  - Silhouette Score trên Val: +0.0190
  - Davies-Bouldin Index: 6.1566
  - Calinski-Harabasz Index: 11.37
  - Purity so với category_gold: **87.28%**
  - NMI: **0.7004**
  - ARI: **0.5925**
  - Multi-seed Stability (5 seeds): ARI trung bình = **0.8224 ± 0.0444** (độ ổn định cao)
  - Phân bố cụm: Kích thước cụm nhỏ nhất trên tập Val là 76 bài báo (không bị phân mảnh cụm rác).
- Artifact or result location:
  - Artifacts: `artifacts/kmeans/` (6 files)
  - Figures: `results/kmeans/figures/` (5 biểu đồ)
  - K-search metrics: `results/kmeans/kmeans_k_search_metrics.csv`
  - Đánh giá định tính: `results/kmeans/kmeans_manual_evaluation.csv`
  - Notebook: `notebooks/05_kmeans_analysis.ipynb`
  - Docs: `docs/m02_kmeans_guide.md`

## Checklist

- [x] I worked on a feature branch.
- [x] I did not push directly to main or develop.
- [x] The code runs with valid input.
- [x] Required tests or checks have passed.
- [x] No password, token, API key, or .env file is committed.
- [x] No personal absolute path is hard-coded.
- [x] Data in data/raw was not modified.
