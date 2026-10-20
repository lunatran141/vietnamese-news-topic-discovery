# E04 — Đánh giá độ ổn định của HDBSCAN

## 1. Mục tiêu và phạm vi

Thí nghiệm E04 kiểm tra mức độ thay đổi của kết quả HDBSCAN khi thay `random_state` của UMAP. Cấu hình được đánh giá là **`minilm_title_content + eom_15_5`**, cấu hình HDBSCAN đại diện đã được chọn cho phần so sánh E03.

Chúng tôi thực hiện hai giai đoạn:

1. Chạy thử với seed **42, 43, 44** để kiểm tra pipeline, phân tích các bài chuyển giữa nhãn nhiễu `-1` và cụm, và đọc ví dụ cụ thể.
2. Mở rộng cùng cấu hình lên **10 seed định trước, từ 42 đến 51**, để mô tả mức dao động trên nhiều lần chạy hơn.

Kết quả 10 seed **bao gồm lại ba seed 42–44**; hai giai đoạn không phải hai bộ dữ liệu độc lập. Thí nghiệm chỉ thay seed UMAP, không dùng validation để chọn seed có kết quả đẹp nhất. Seed **42** vẫn là seed của kết quả M11 gốc.

## 2. Dữ liệu và cấu hình cố định

| Thành phần | Giá trị |
| --- | --- |
| Embedding | `minilm_title_content` — tiêu đề và 60 từ đầu nội dung |
| Train | 6.493 bài báo |
| Validation | 1.391 bài báo |
| UMAP | `n_neighbors=15`, `n_components=5`, `min_dist=0`, khoảng cách cosine |
| Seed UMAP | 42–51 |
| HDBSCAN | `min_cluster_size=15`, `min_samples=5`, khoảng cách Euclidean, `cluster_selection_method=eom` |

Với mỗi seed, UMAP và HDBSCAN được **fit trên cùng tập train**. Các bài validation được biến đổi bằng UMAP đã fit trên train và được gán nhãn bằng HDBSCAN đã fit trên train; **không fit lại trên validation**. Embedding của validation được tạo một lần trong mỗi lần chạy script validation rồi dùng chung cho các seed.

Script kiểm tra seed 42 tạo lại đúng nhãn train M11 và nhãn validation M11 đã lưu. Các kiểm tra này đã đạt khi xuất kết quả E04.

## 3. Chỉ số và cách đối chiếu

- **Số cụm:** số nhãn train khác `-1`.
- **Tỷ lệ nhiễu:** tỷ lệ bài có nhãn `-1` trên từng split.
- **Độ bao phủ validation:** tỷ lệ bài validation được gán vào cụm, bằng `100% − tỷ lệ nhiễu`.
- **Pairwise ARI toàn bộ:** Adjusted Rand Index giữa hai bộ nhãn của cùng các `article_id` ở hai seed, **bao gồm nhãn `-1`**.
- **Tỷ lệ đổi trạng thái nhiễu:** tỷ lệ bài chuyển giữa `-1` và một cụm trong một cặp seed.
- **Jaccard tập nhiễu:** mức trùng nhau giữa hai tập `article_id` mang nhãn `-1`.
- **ARI trên bài cùng được gán cụm:** ARI chỉ trên những bài không mang nhãn `-1` ở **cả hai seed**; luôn đọc cùng số bài còn lại sau khi lọc.

`cluster_id` được tạo riêng trong từng lần fit. Không kết luận hai cụm giống hoặc khác nhau chỉ vì số `cluster_id` bằng hoặc khác nhau; việc so sánh cụm phải dựa trên tập `article_id`. Pairwise ARI ở đây đo độ nhất quán **giữa các seed**, khác ARI so với `category_gold` trong đánh giá M11.

Các giá trị `±` là trung bình và độ lệch chuẩn mẫu. Mười seed tạo **45 cặp** để tính pairwise ARI. Vì các cặp dùng chung seed và dữ liệu, 45 giá trị này **không phải 45 phép thử độc lập**; trung bình và độ lệch chuẩn chỉ mang tính mô tả.

## 4. Giai đoạn đầu: ba seed 42–44

### 4.1. Train

| Seed | Số cụm | Số bài `-1` | Tỷ lệ nhiễu |
| ---: | ---: | ---: | ---: |
| 42 | 101 | 1.608 | 24,77% |
| 43 | 112 | 1.824 | 28,09% |
| 44 | 113 | 1.750 | 26,95% |

Pairwise ARI trên 6.493 bài train:

| Cặp seed | ARI |
| --- | ---: |
| 42–43 | 0,4344 |
| 42–44 | 0,4669 |
| 43–44 | 0,5710 |

### 4.2. Validation

| Seed | Số bài `-1` | Tỷ lệ nhiễu | Độ bao phủ |
| ---: | ---: | ---: | ---: |
| 42 | 484 | 34,80% | 65,20% |
| 43 | 575 | 41,34% | 58,66% |
| 44 | 550 | 39,54% | 60,46% |

Pairwise ARI trên 1.391 bài validation:

| Cặp seed | ARI |
| --- | ---: |
| 42–43 | 0,4193 |
| 42–44 | 0,4325 |
| 43–44 | 0,5380 |

Ở ba seed đầu, tỷ lệ bài chuyển giữa `-1` và cụm nằm trong khoảng **12,91–16,02% trên train** và **17,76–21,93% trên validation**. Jaccard giữa các tập bài nhiễu lần lượt là **0,535–0,620** và **0,553–0,640**.

Riêng cặp seed 43–44, ARI trên những bài được gán cụm ở cả hai seed là **0,9123 trên train (4.287 bài)** và **0,9660 trên validation (705 bài)**, cao hơn ARI toàn bộ. Điều này cho thấy cần xem riêng việc bài chuyển giữa nhiễu và cụm. ARI sau khi lọc bài **không thay thế** ARI trên toàn bộ dữ liệu.

## 5. Kết quả chính: mười seed 42–51

### 5.1. Kết quả từng seed

| Seed | Số cụm train | Nhiễu train | Bài `-1` validation | Nhiễu validation | Độ bao phủ validation |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 101 | 24,77% | 484 | 34,80% | 65,20% |
| 43 | 112 | 28,09% | 575 | 41,34% | 58,66% |
| 44 | 113 | 26,95% | 550 | 39,54% | 60,46% |
| 45 | 116 | 27,94% | 551 | 39,61% | 60,39% |
| 46 | 99 | 23,86% | 491 | 35,30% | 64,70% |
| 47 | 117 | 28,08% | 571 | 41,05% | 58,95% |
| 48 | 110 | 29,83% | 565 | 40,62% | 59,38% |
| 49 | 101 | 25,27% | 487 | 35,01% | 64,99% |
| 50 | 110 | 27,52% | 568 | 40,83% | 59,17% |
| 51 | 117 | 30,77% | 603 | 43,35% | 56,65% |

### 5.2. Tóm tắt 10 seed

| Chỉ số | Trung bình ± độ lệch chuẩn mẫu | Min–max |
| --- | ---: | ---: |
| Số cụm train | 109,60 ± 6,90 | 99–117 |
| Tỷ lệ nhiễu train | 27,31 ± 2,18% | 23,86–30,77% |
| Tỷ lệ nhiễu validation | 39,15 ± 3,02% | 34,80–43,35% |
| Độ bao phủ validation | 60,86 ± 3,02% | 56,65–65,20% |
| Pairwise ARI train, 45 cặp | 0,5165 ± 0,0636 | 0,4223–0,6693 |
| Pairwise ARI validation, 45 cặp | 0,4774 ± 0,0619 | 0,3796–0,6266 |

Tỷ lệ nhiễu validation trung bình cao hơn train khoảng **11,84 điểm phần trăm**. Đây là so sánh mô tả giữa hai split; không tự nó xác định nguyên nhân khiến bài validation khó gán cụm hơn.

### 5.3. Chuyển trạng thái `-1` qua 45 cặp seed

| Chỉ số theo cặp seed | Train | Validation |
| --- | ---: | ---: |
| Tỷ lệ bài chuyển giữa `-1` và cụm, trung bình | 14,16% | 19,96% |
| Khoảng tỷ lệ chuyển trạng thái | 11,01–16,54% | 14,45–24,16% |
| Jaccard tập nhiễu, trung bình | 0,5884 | 0,5940 |
| Khoảng Jaccard tập nhiễu | 0,5267–0,6348 | 0,5179–0,6581 |
| ARI trên bài cùng được gán cụm, trung bình | 0,7386 | 0,7952 |
| Khoảng ARI trên bài cùng được gán cụm | 0,5081–0,9839 | 0,5722–0,9936 |

ARI trên bài cùng được gán cụm được tính trên **một tập bài khác nhau cho mỗi cặp seed**. Vì vậy, không diễn giải việc chỉ số này cao hơn ARI toàn bộ là mô hình đã ổn định trên toàn bộ dữ liệu. Trên validation, số bài cùng được gán cụm trong từng cặp dao động từ **661 đến 805 bài**.

File chi tiết có **41.363 dòng chuyển trạng thái train** và **12.493 dòng chuyển trạng thái validation** trên tổng 45 cặp. Một `article_id` có thể xuất hiện ở nhiều cặp seed; đây là **số lượt chuyển theo cặp**, không phải số bài duy nhất.

## 6. Ví dụ đọc thủ công từ ba seed đầu

Các ví dụ sau dùng nhãn theo thứ tự **seed 42 → 43 → 44**. Chúng minh họa những kiểu thay đổi khác nhau; không được dùng riêng để kết luận cấu hình nào đúng hơn.

- **`TTR_056b1c42a223` — “HLV Kim Sang Sik xin lỗi người hâm mộ sau trận thua Thái Lan”:** `-1 → 32 → 31`. Hai cụm nhận bài ở seed 43 và 44 đều có 218 bài train, trong đó **212 bài chung**; phần lớn là tin thể thao, gồm nhiều tin về tuyển Việt Nam và FIFA ASEAN Cup. Ở seed 42, các bài train của nhóm này bị phân tán sang nhiều cụm hoặc `-1`. Bài validation chỉ được gán vào nhóm bóng đá này ở seed 43 và 44.

- **`TTR_384e9e8f3c70` — “Thu hồi toàn quốc lô thuốc Colchicina Seid trị gout do vi phạm”:** `-1 → 19 → 44`. Cả ba seed đều có một cụm train gồm **đúng cùng 15 bài**, trong đó có nhiều tin về sản phẩm hoặc thuốc bị cảnh báo, thu hồi. Thành viên cụm train giữ nguyên, nhưng nhãn của bài validation thay đổi. Ca này cho thấy chỉ kiểm tra độ ổn định thành viên cụm train là chưa đủ để đánh giá việc gán bài mới.

- **`TTR_3ab70a8a794a` — “Học mầm non ở TP.HCM: từ 25 triệu đến hơn 350 triệu đồng/năm”:** `-1 → -1 → 92`. Cụm ở seed 44 có 111 bài train, gồm 110 bài thuộc chuyên mục Giáo dục. Nhiều bài của nhóm này được chia sang các cụm khác ở seed 42 và 43. Đây là bài có nội dung giáo dục hợp lệ nhưng chỉ được gán cụm ở một trong ba seed.

- **`TTR_8082fcb986b2` — “Bác sĩ hướng dẫn 7 dấu hiệu mắc đái tháo đường dễ bị bỏ qua”:** `1 → -1 → -1`. Cụm seed 42 có 854 bài train, trong đó 769 bài thuộc chuyên mục Sức khỏe; ở các seed sau, nhiều bài của cụm lớn này chuyển sang cụm khác hoặc `-1`. Bài đái tháo đường có nội dung sức khỏe rõ ràng, nên nhãn `-1` không tự nó có nghĩa là dữ liệu sai hoặc bài không có chủ đề.

- **`TTR_facebe2bccf3` — “Bắc Kinh cấm drone toàn diện, người dân không được giữ thiết bị và linh kiện”:** `26 → 23 → -1`. Hai cụm train ở seed 42 và 43 có **71 bài chung** trên các cụm 71 và 72 bài; nhiều bài nổi bật nói về UAV và công nghệ quân sự. Bài validation tập trung vào quy định cấm sở hữu drone, với membership khi được gán cụm chỉ khoảng **0,094** và **0,109**. Đây là ca gán cụm yếu; khác biệt trọng tâm nội dung có thể góp phần vào việc đổi nhãn, nhưng CSV hiện có chưa xác định được nguyên nhân trực tiếp.

Các ví dụ cho thấy có ít nhất hai dạng thay đổi cần phân biệt: cấu trúc cụm train thay đổi giữa các seed, và cụm train giữ nguyên thành viên nhưng dự đoán cho bài validation vẫn thay đổi. Membership mô tả mức gắn với cụm theo mô hình; nó không chứng minh nhãn đúng về mặt ngữ nghĩa.

## 7. Kết luận và giới hạn

Trên cấu hình `minilm_title_content + eom_15_5`, việc thay seed UMAP làm thay đổi cả số cụm, tỷ lệ nhiễu và nhãn từng bài. Với 10 seed, pairwise ARI trung bình đạt **0,5165 trên train** và **0,4774 trên validation**. Trung bình gần **một phần năm** bài validation chuyển trạng thái giữa `-1` và cụm khi xét một cặp seed. Vì vậy, không nên trình bày số cụm hoặc tỷ lệ nhiễu của seed 42 như giá trị cố định của cấu hình mà không nêu seed và kết quả kiểm tra độ ổn định.

Phạm vi kết luận được giới hạn như sau:

1. Mười seed dùng cùng embedding và cùng các bài train/validation. Thí nghiệm đo độ nhạy với **seed UMAP**, chưa đo độ nhạy với việc thay mẫu train.
2. Validation thuộc cùng bộ dữ liệu đã chia trước. Thí nghiệm này không thay thế đánh giá dữ liệu khác thời gian hoặc nguồn của nhiệm vụ OOD.
3. ARI toàn bộ bao gồm nhãn `-1`; ARI trên bài cùng được gán cụm chỉ áp dụng cho một tập con thay đổi theo cặp seed. Cần xem hai chỉ số cùng tỷ lệ chuyển trạng thái, Jaccard tập nhiễu và các bài ví dụ.
4. Đây là kết quả mô tả của 10 seed và 45 cặp phụ thuộc nhau, không phải khoảng tin cậy từ 45 thí nghiệm độc lập.
5. Không dùng tỷ lệ nhiễu hoặc ARI validation để chọn seed tốt nhất. **Seed 42** vẫn là kết quả M11 gốc được báo cáo trong so sánh E03.

## 8. Script và file kết quả

Các script:

- `experiments/hdbscan/run_e04_train_stability.py`: fit trên train theo từng seed, xuất assignments và chỉ số train.
- `experiments/hdbscan/run_e04_validation_stability.py`: fit trên train rồi dự đoán cùng validation theo từng seed.
- `experiments/hdbscan/analyze_e04_seed_changes.py`: phân tích các assignments đã lưu; bước này không fit lại mô hình.

Kết quả **ba seed ban đầu** nằm trực tiếp trong `results/hdbscan/stability/`. Kết quả **10 seed chính** nằm trong:

`results/hdbscan/stability/minilm_title_content_eom_15_5_seeds42_51/`

Mỗi thư mục gồm:

- `train_seed_metrics.csv`, `train_pairwise_ari.csv`, `train_assignments.csv`, `train_summary.csv`.
- `validation_seed_metrics.csv`, `validation_pairwise_ari.csv`, `validation_assignments.csv`, `validation_summary.csv`.
- `train_pairwise_state_analysis.csv`, `train_transition_cases.csv`.
- `val_pairwise_state_analysis.csv`, `val_transition_cases.csv`.

Trong thư mục 10 seed, `train_assignments.csv` có **64.930 dòng** (6.493 × 10) và `validation_assignments.csv` có **13.910 dòng** (1.391 × 10). Mỗi file `*_pairwise_ari.csv` và `*_pairwise_state_analysis.csv` có **45 dòng**, ứng với 45 cặp seed.

## 9. Cách chạy lại kết quả 10 seed

Chạy từ thư mục gốc của dự án:

```powershell
python -B -m experiments.hdbscan.run_e04_train_stability
python -B -m experiments.hdbscan.run_e04_validation_stability
python -B -m experiments.hdbscan.analyze_e04_seed_changes
```

Trước khi chạy, kiểm tra trong script train: `EMBEDDING_RUN="minilm_title_content"`, `EXPERIMENT="eom_15_5"` và `SEEDS=tuple(range(42, 52))`. Kiểm tra `RESULT_DIR` của script phân tích trỏ đến thư mục kết quả 10 seed nêu trên.

Các lệnh trên tạo lại **kết quả 10 seed**. CSV ba seed ở thư mục gốc là kết quả của giai đoạn chạy thử đã được giữ lại; không ghi đè thư mục đó khi chạy 10 seed.