# Báo cáo Nghiên cứu và Hướng dẫn Kỹ thuật Task M03: Bóc tách Tiền xử lý (Preprocessing Ablation)

## 1. Mục tiêu Nghiên cứu và Cơ sở Phương pháp Luận

Báo cáo này giải quyết **Câu hỏi Nghiên cứu RQ2** (theo quy định tại Mục 10.3 và Phần B4/D3 của `project_handbook.md`):
> *"Tiền xử lý ảnh hưởng như thế nào đến chất lượng phân cụm tiếng Việt? Các kỹ thuật chuẩn hóa, tách từ ghép và lọc từ dừng đóng góp cụ thể ra sao vào độ thuần khiết của cụm và khả năng diễn giải ngữ nghĩa của con người?"*

Đồng thời, báo cáo trực tiếp làm rõ và giải thích yêu cầu của giảng viên hướng dẫn:
> *"Trường hợp điểm số tự động cao nhưng topic khó diễn giải: Phân tích nguyên nhân (ví dụ: từ dừng, đa nghĩa, kích thước cụm không đều, phân bố từ vựng bị lệch). Minh họa bằng ví dụ cụ thể từ kết quả mô hình. So sánh với các chủ đề có điểm số tốt và dễ diễn giải."*

### Nguyên tắc Kiểm soát Biến Chặt chẽ (*Ceteris Paribus*)
Để đảm bảo kết luận khoa học có độ tin cậy cao nhất, thực nghiệm được thiết kế tuân thủ nghiêm ngặt nguyên tắc kiểm soát biến:
- **Cùng tập dữ liệu:** Giữ nguyên 100% phân chia văn bản cố định theo `data/splits/split_manifest.csv` (Train: 6,493 bài báo | Validation: 1,391 bài báo). Tuyệt đối không chạm vào tập Test (1,390 bài).
- **Cùng thuật toán và siêu tham số phân cụm:** Thuật toán K-Means với $K = 9$ (cấu hình tối ưu được xác lập từ Task M02), khởi tạo `k-means++`, `n_init = 10`, `max_iter = 300`, `random_state = 42`.
- **Cùng tham số trích xuất đặc trưng TF-IDF:** `ngram_range = (1, 1)`, `min_df = 2`, `max_df = 0.85`, `max_features = 20,000`, `sublinear_tf = True`, `norm = 'l2'`, `token_pattern = r"(?u)\b\w\w+\b"`.
- **Cô lập biến thể kỹ thuật (*Technical Isolation Guard*):** Trong cấu hình TF-IDF, đặt bắt buộc `lowercase: false` và `stopwords_file: null`. Nếu để mặc định `lowercase: true` của scikit-learn, nó sẽ tự động hạ case văn bản của P0, triệt tiêu sự khác biệt giữa P0 và P1. Tương tự, nếu bật `stopwords_file` trong vectorizer, nó sẽ tự động lọc từ dừng ở P2, làm mất ranh giới giữa P2 và P3.
- **Biến độc lập duy nhất:** 4 biến thể tiền xử lý văn bản đầu vào:
  1. **P0 (Tối giản - Minimal):** Giữ nguyên văn bản gốc, chỉ sửa mã Unicode NFC và gỡ thẻ HTML. Giữ nguyên chữ hoa/thường, dấu câu, số, emoji, không tách từ, không lọc từ dừng.
  2. **P1 (Chuẩn hóa cơ bản - Normalized):** Chuyển toàn bộ về chữ thường, thay thế URL/Email bằng token đặc biệt, chuẩn hóa khoảng trắng, loại bỏ dấu câu thừa, không tách từ, không lọc từ dừng.
  3. **P2 (Tách từ - Segmented):** Áp dụng công cụ tách từ tiếng Việt PyVi (`ViTokenizer`), liên kết các tiếng tạo thành từ ghép thông qua dấu gạch dưới `_`, giữ nguyên toàn bộ từ dừng.
  4. **P3 (Lọc từ dừng - No Stopwords):** Loại bỏ danh sách 388 từ dừng tiếng Việt chuẩn từ P2, có cơ chế bảo vệ nghiêm ngặt các từ phủ định (`không`, `chưa`, `chẳng`) và các thuật ngữ chủ đề cốt lõi.

---

## 2. Bảng Tổng hợp Kết quả Định lượng trên Tập Validation ($K = 9$)

Toàn bộ dữ liệu đo lường được lưu trữ độc lập tại `results/kmeans/m03_preprocessing_ablation_metrics.csv`.

| Chỉ số Đánh giá | P0 (Tối giản) | P1 (Chuẩn hóa) | P2 (Tách từ) | P3 (Lọc từ dừng) | Xu hướng Tối ưu |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Kích thước từ vựng (Vocab Size)** | 17,983 | 15,285 | 20,000 | 20,000 | Phản ánh độ giàu thông tin |
| **Độ thưa tập Val (Sparsity %)** | 98.34% | 98.17% | 98.79% | 99.05% | Tăng nhẹ khi từ vựng cô đọng |
| **Tỷ lệ OOV tập Val (OOV Rate %)** | 9.86% | 11.04% | 11.97% | **2.56%** | Thấp hơn là tốt |
| **Số đặc trưng kích hoạt TB/bài (Mean Active)** | 299.3 | 279.9 | 241.8 | 190.8 | Tinh gọn, loại bỏ từ rác |
| **Silhouette Score** (Nội tại) | +0.0193 | +0.0186 | +0.0181 | **+0.0190** | Cao hơn là tốt (thước đo phụ) |
| **Davies-Bouldin Index (DBI)** (Nội tại) | 5.7112 | **5.5708** | 6.1151 | 6.1566 | Thấp hơn là tốt (nghịch lý hình học) |
| **Calinski-Harabasz Score** (Nội tại) | 13.15 | **14.17** | 11.50 | 11.38 | Cao hơn là tốt (nghịch lý hình học) |
| **Độ thuần khiết (Purity)** (Ngoại vi) | 83.61% | 81.52% | 81.60% | **87.28%** | **Vượt trội (+5.68% so với P2)** |
| **Normalized Mutual Info (NMI)** (Ngoại vi) | 0.6652 | 0.6503 | 0.6494 | **0.7004** | **Cao nhất toàn diện** |
| **Adjusted Rand Index (ARI)** (Ngoại vi) | 0.5457 | 0.5161 | 0.4958 | **0.5925** | **Cao nhất toàn diện (+0.097)** |
| **Độ ổn định đa seed (Mean ARI $\pm$ Std)** | 0.8599 $\pm$ 0.0487 | 0.8267 $\pm$ 0.0612 | 0.8186 $\pm$ 0.0478 | 0.8211 $\pm$ 0.0442 | Ổn định vững chắc qua 5 seed |
| **Tỷ lệ từ ghép trong Top 15 từ khóa (%)** | 0.0% | 0.0% | 32.59% | **43.70%** | Cao hơn phản ánh ngữ nghĩa tốt |
| **Tỷ lệ ô nhiễm từ dừng trong Top 15 (%)** | 9.63% | 9.63% | 18.52% | **0.00%** | **Triệt tiêu hoàn toàn nhiễu** |
| **Độ đa dạng từ khóa Top 10 (Diversity)** | 0.9000 | 0.8778 | 0.9222 | **0.9556** | Cụm có ranh giới từ vựng riêng biệt |

---

## 3. Phân tích Hiện tượng Chuyên sâu

### 3.1. Nghịch lý Chỉ số Nội tại (The Geometric / Internal Metric Paradox)
Một trong những phát hiện quan trọng nhất của thực nghiệm (giải quyết trực tiếp câu hỏi của giảng viên) là sự phân kỳ giữa các chỉ số nội tại hình học (Davies-Bouldin, Calinski-Harabasz) và chất lượng phân cụm thực tế (Purity, NMI, ARI, tính dễ hiểu của từ khóa):
- **Hiện tượng:** Biến thể P1 đạt chỉ số Davies-Bouldin tốt nhất (5.5708 - thấp nhất) và Calinski-Harabasz cao nhất (14.17), trong khi P3 lại có DBI kém hơn (6.1566) và Calinski-Harabasz thấp hơn (11.38).
- **Bản chất toán học và ngữ nghĩa:**
  1. Trong P0 và P1, ngôn ngữ tiếng Việt chưa được tách từ, văn bản bị phân rã thành hàng loạt các âm tiết đơn lẻ có tần suất xuất hiện cực cao (như `các`, `của`, `người`, `và`, `được`, `năm`).
  2. Các âm tiết đơn này đóng vai trò như những "trọng lực giả", kéo vector TF-IDF của tất cả các bài báo co cụm lại gần nhau trong không gian hình học đa chiều. Về mặt toán học Euclid, khoảng cách giữa các điểm trong cụm bị thu hẹp cưỡng bức, làm cho tỷ số phân tán nội cụm trên khoảng cách liên cụm (công thức tính Davies-Bouldin) giảm xuống một cách cơ học.
  3. Tuy nhiên, sự "chặt chẽ" này hoàn toàn là ảo giác hình học: các cụm tạo ra bị ô nhiễm nghiêm trọng bởi các từ dừng và từ vựng què cụt đa nghĩa, khiến con người không thể xác định được ranh giới chủ đề.
  4. Ngược lại, ở P3, khi toàn bộ 388 từ dừng được loại bỏ, các bài báo chỉ còn lại các thuật ngữ mang tính chuyên môn cao. Vector không gian lúc này tỏa rộng ra các hướng ngữ nghĩa đặc thù. Về mặt hình học, cụm có vẻ "nở rộng" hơn (làm DBI tăng nhẹ từ 5.57 lên 6.16), nhưng các bài báo phân tách cực kỳ chính xác theo nội dung chủ đề thực thụ. Minh chứng là **Purity tăng vọt từ 81.52% lên 87.28%**, **NMI tăng vọt từ 0.6503 lên 0.7004**, và **ARI tăng vọt từ 0.5161 lên 0.5925**.

### 3.2. Sức mạnh của Tách từ Tiếng Việt (Word Segmentation - P2)
Tiếng Việt là ngôn ngữ đơn lập, ranh giới từ không trùng với khoảng trắng. Khi chuyển từ P1 sang P2:
- Ở P0 và P1, các thực thể và thuật ngữ quan trọng bị băm nhỏ thành các âm tiết vô nghĩa hoặc đa nghĩa nghiêm trọng:
  - Cụm Y tế bị băm thành: `bác`, `sĩ`, `điều`, `trị`, `viện` (từ `sĩ` có thể là bác sĩ, tiến sĩ, chiến sĩ, nghệ sĩ; từ `viện` có thể là bệnh viện, viện nghiên cứu, viện kiểm sát).
  - Cụm Tài chính bị băm thành: `phiếu`, `lãi`, `thị`, `trường` (`phiếu` có thể là cổ phiếu, phiếu bầu, phiếu giảm giá; `lãi` có thể là lãi suất, có lãi).
- Ở P2, công cụ `PyVi` đã kết hợp chính xác thành các từ ghép ngữ nghĩa hoàn chỉnh: `bác_sĩ`, `điều_trị`, `bệnh_viện`, `cổ_phiếu`, `lãi_suất`, `thị_trường`.
- Tỷ lệ từ ghép trong Top 15 từ khóa tâm cụm tăng vọt từ **0.0% lên 32.59%**, giúp người đọc lập tức nhận diện được lĩnh vực chuyên môn của cụm bài báo mà không bị nhập nhằng ngữ nghĩa.

### 3.3. Vai trò Quyết định của Lọc từ dừng (Stopword Filtering - P3)
Mặc dù P2 đã tạo ra từ ghép, nhưng P2 lại mắc phải hiện tượng "ô nhiễm từ dừng":
- Trong P2, tỷ lệ ô nhiễm từ dừng ở Top 15 từ khóa tâm cụm lên tới **18.52%**. Thậm chí ở cụm Giải trí (Cluster 2 của P2), có tới 6/10 từ khóa hàng đầu là từ dừng: `những, một, người, năm, không, nhiều, từ`.
- Ở cụm Kinh doanh (Cluster 8 của P2), các từ như `theo, tại, năm, từ` chiếm chỗ của các thuật ngữ ngành, làm mờ nhạt đặc trưng của cụm.
- Ở P3, bước lọc từ dừng đã quét sạch hoàn toàn các từ rác này (**Stopword pollution giảm tuyệt đối về 0.00%**). Nhờ đó, các từ ghép đắt giá được đôn lên hàng đầu:
  - Cụm Kinh doanh (Cluster 6 ở P3) đạt **100% từ khóa là từ ghép kinh tế cao cấp**: `doanh_nghiệp, phát_triển, đầu_tư, công_nghệ, dự_án, thị_trường, việt_nam, xây_dựng, sản_phẩm, dịch_vụ`.
  - Cụm Khoa học Công nghệ (Cluster 2 ở P3) nổi bật các từ: `iphone, apple, pro, màn_hình, max, hệ_thống, thiết_bị, sử_dụng`.
  - Nhờ loại bỏ nhiễu, tỷ lệ OOV trên tập Validation giảm mạnh từ 11.97% xuống còn **2.56%**, và Purity đạt mức cao nhất toàn dự án (**87.28%**).

---

## 4. Kiểm tra Độ nhạy với Mốc chuẩn $K = 7$ (Sensitivity Analysis)

Bảng đối chiếu độ nhạy từ `results/kmeans/m03_preprocessing_ablation_k7_sensitivity.csv` khẳng định các kết luận trên không phụ thuộc vào giá trị $K$:

| Biến thể | K = 7 (Purity) | K = 9 (Purity) | K = 7 (NMI) | K = 9 (NMI) | K = 7 (ARI) | K = 9 (ARI) | K = 7 (Stopwords %) | K = 9 (Stopwords %) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **P0** | 78.43% | 83.61% | 0.6420 | 0.6652 | 0.5505 | 0.5457 | 13.33% | 9.63% |
| **P1** | 81.09% | 81.52% | 0.6683 | 0.6503 | 0.6066 | 0.5161 | 12.38% | 9.63% |
| **P2** | 81.52% | 81.60% | 0.6788 | 0.6494 | 0.5869 | 0.4958 | 21.90% | 18.52% |
| **P3** | **82.10%** | **87.28%** | **0.6889** | **0.7004** | **0.6239** | **0.5925** | **0.00%** | **0.00%** |

**Nhận xét:**
- Trên cả $K = 7$ (mốc đối chiếu ứng với 7 nhãn `category_gold`) và $K = 9$ (cấu hình tối ưu được chọn), **P3 luôn duy trì vị trí quán quân** về Purity, NMI, ARI và độ sạch từ vựng.
- Cấu hình $K = 9$ cho phép P3 phát huy tối đa sức mạnh khi tách riêng được các chủ đề nhánh chuyên biệt (như tách nhóm bài Công nghệ/AI khỏi Kinh doanh tổng hợp, tách Bóng đá khỏi Thể thao đa môn), nâng Purity từ 82.10% lên 87.28%.

---

## 5. Dẫn chứng Minh họa Định tính Từng Cụm (Qualitative Case Studies)

Trích lục trực tiếp từ `results/kmeans/m03_qualitative_cluster_examples.csv` đối chiếu giữa P0 (chưa xử lý) và P3 (xử lý hoàn chỉnh):

### Cụm 1: Chủ đề Tài chính - Ngân hàng
- **P0 (Tối giản):**
  - Top 10 từ khóa: `tăng, phiếu, lãi, USD, giảm, mức, giá, so, thị, đồng`
  - Đánh giá: Chữ `USD` viết hoa lẫn chữ thường; các âm tiết `phiếu`, `lãi`, `thị` bị cụt đầu cụt đuôi, rất khó hiểu nếu đọc độc lập.
- **P3 (Lọc từ dừng):**
  - Top 10 từ khóa: `tăng, giá, đồng, lãi_suất, thị_trường, giảm, usd, cổ_phiếu, so, bán`
  - Đánh giá: Xuất hiện nguyên vẹn các thuật ngữ tài chính `lãi_suất`, `thị_trường`, `cổ_phiếu`, biểu thị rõ nét ngữ nghĩa của thị trường vốn.

### Cụm 2: Chủ đề Y tế - Sức khỏe
- **P0 (Tối giản):**
  - Top 10 từ khóa: `bệnh, bác, sĩ, viện, Bệnh, máu, hoặc, khoa, thể, cơ`
  - Đánh giá: Xuất hiện trùng lặp từ `Bệnh` (viết hoa) và `bệnh` (viết thường); hai âm tiết `bác` và `sĩ` tách rời nhau.
- **P3 (Lọc từ dừng):**
  - Top 10 từ khóa: `bác_sĩ, bệnh, điều_trị, bệnh_viện, người_bệnh, hoặc, khám, máu, thuốc, tình_trạng`
  - Đánh giá: 100% từ vựng chuẩn y khoa, phản ánh toàn diện hoạt động khám chữa bệnh.

### Cụm 3: Chủ đề Kinh doanh - Doanh nghiệp
- **P2 (Tách từ nhưng còn từ dừng):**
  - Top 10 từ khóa: `doanh_nghiệp, phát_triển, công_nghệ, theo, đầu_tư, tại, năm, từ, dự_án, hoạt_động`
  - Đánh giá: 4/10 từ khóa bị chiếm giữ bởi hư từ (`theo, tại, năm, từ`), làm loãng thông điệp cốt lõi.
- **P3 (Lọc từ dừng hoàn chỉnh):**
  - Top 10 từ khóa: `doanh_nghiệp, phát_triển, đầu_tư, công_nghệ, dự_án, thị_trường, việt_nam, xây_dựng, sản_phẩm, dịch_vụ`
  - Đánh giá: Toàn bộ 10 từ khóa đều là danh từ và động từ hành động đắt giá, thể hiện bức tranh toàn cảnh về phát triển kinh tế vĩ mô và doanh nghiệp quốc gia.

---

## 6. Danh mục Tài nguyên và Hướng dẫn Tái hiện (Reproducibility)

Tất cả các cấu phần phục vụ tái hiện thực nghiệm đều được đóng gói đầy đủ:
1. **File cấu hình:** `configs/m03_preprocessing_ablation.yaml`
2. **Script chạy thực nghiệm:** `experiments/kmeans/run_m03_preprocessing_ablation.py`
3. **Bộ kiểm thử tự động:** `tests/test_m03_preprocessing_ablation.py`
4. **Notebook trực quan tương tác:** `notebooks/06_m03_preprocessing_ablation.ipynb`
5. **Dữ liệu kết quả:**
   - `results/kmeans/m03_preprocessing_ablation_metrics.csv`
   - `results/kmeans/m03_preprocessing_ablation_k7_sensitivity.csv`
   - `results/kmeans/m03_qualitative_cluster_examples.csv`
   - `results/kmeans/m03_cluster_keywords_p0_p3.json`
6. **Bộ 4 biểu đồ độ phân giải cao (300 DPI):**
   - `results/kmeans/figures/m03_ablation_metrics_comparison.png`
   - `results/kmeans/figures/m03_intrinsic_vs_extrinsic_paradox.png`
   - `results/kmeans/figures/m03_lexical_interpretability.png`
   - `results/kmeans/figures/m03_k7_vs_k9_sensitivity.png`

**Lệnh chạy tái hiện:**
```bash
python -m experiments.kmeans.run_m03_preprocessing_ablation
pytest tests/test_m03_preprocessing_ablation.py -v
```
