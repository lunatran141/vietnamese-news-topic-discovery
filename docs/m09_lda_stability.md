
# M09 – Đánh giá độ ổn định của mô hình LDA

**Người phụ trách:** TV3 – LDA

**Nhánh Git:** `feature/M09-lda-stability`

**Phiên bản dữ liệu:** Dataset v1.0

**Biến thể tiền xử lý:** `p2_segmented`

## 1. Mục tiêu

Task M09 nhằm đánh giá độ ổn định của mô hình Latent Dirichlet Allocation (LDA) khi thay đổi giá trị khởi tạo ngẫu nhiên (`random_state`).

Trong quá trình thực nghiệm, các yếu tố sau được giữ cố định:

- Tập dữ liệu huấn luyện và tập Validation.
- Phương pháp tiền xử lý văn bản.
- Cấu hình Bag-of-Words (BoW).
- Số lượng topic K = 15.
- Các siêu tham số của mô hình LDA, ngoại trừ `random_state`.

Thực nghiệm tập trung trả lời ba câu hỏi:

1. Khi thay đổi random seed, các từ khóa đại diện cho topic có thay đổi nhiều không?
2. Phân phối xác suất topic của cùng một bài báo có ổn định giữa các lần huấn luyện không?
3. Kết quả từ M08 với K = 15 có nhạy cảm với việc khởi tạo ngẫu nhiên hay không?

## 2. Dữ liệu và cấu hình thực nghiệm

| Tham số | Giá trị |
|---|---|
| Phiên bản dữ liệu | Dataset v1.0 |
| Biến thể tiền xử lý | `p2_segmented` |
| Số bài báo Train | 6.493 |
| Số bài báo Validation | 1.391 |
| Tập dùng để đánh giá | Validation |
| Kích thước từ vựng BoW | 15.000 |
| Số lượng topic K | 15 |
| Random seeds | 42, 52, 62, 72, 82 |
| Seed tham chiếu | 42 |
| Số từ khóa đại diện mỗi topic | 10 |
| Thuật toán ghép topic | Hungarian Matching |
| Độ tương đồng từ khóa | Jaccard Similarity |
| Độ tương đồng phân phối topic | Cosine Similarity |

BoW được fit một lần duy nhất trên tập Train và dùng để transform tập Validation.

Năm mô hình LDA sử dụng chung ma trận BoW và cùng các siêu tham số, chỉ thay đổi `random_state`.

Tập Test được giữ riêng và không tham gia huấn luyện, lựa chọn tham số hoặc đánh giá stability.

## 3. Phương pháp thực nghiệm

### 3.1. Huấn luyện LDA với nhiều random seeds

Thực nghiệm huấn luyện năm mô hình LDA với cùng số topic K = 15.

Các random seed được sử dụng:

`[42, 52, 62, 72, 82]`

Với mỗi seed, hệ thống thực hiện:

1. Khởi tạo mô hình LDA với cấu hình tương ứng.
2. Huấn luyện mô hình trên ma trận BoW của Train.
3. Trích xuất 10 từ khóa có trọng số cao nhất của từng topic.
4. Suy luận phân phối xác suất topic cho 1.391 bài báo Validation.
5. Ghi nhận thời gian thực hiện.

Các kết quả này được dùng để so sánh giữa những lần huấn luyện.

### 3.2. Ghép topic bằng thuật toán Hungarian

Topic ID không có ý nghĩa cố định giữa các mô hình LDA.

Ví dụ, Topic 0 của seed 42 có thể tương ứng với Topic 7 của seed 52.

Vì vậy, không thể trực tiếp so sánh các topic chỉ dựa trên số thứ tự.

M09 sử dụng thuật toán Hungarian để tìm cách ghép các topic sao cho tổng độ tương đồng từ khóa là lớn nhất.

Độ tương đồng giữa hai tập từ khóa A và B được tính bằng chỉ số Jaccard:

J(A, B) = |A ∩ B| / |A ∪ B|

Trong đó:

- A và B là tập hợp từ khóa của hai topic.
- |A ∩ B| là số từ khóa xuất hiện trong cả hai topic.
- |A ∪ B| là tổng số từ khóa phân biệt của hai topic.

Jaccard có giá trị từ 0 đến 1:

- Jaccard = 1: Hai topic có tập từ khóa giống nhau.
- Jaccard = 0: Hai topic không có từ khóa chung.
- Giá trị nằm giữa 0 và 1: Hai topic có một phần từ khóa trùng nhau.

Sau khi Hungarian matching, độ tương đồng Jaccard trung bình được tính trên 15 cặp topic đã ghép.

### 3.3. Đánh giá độ ổn định phân phối topic

Ngoài từ khóa đại diện, M09 đánh giá sự ổn định của phân phối xác suất topic trên các bài báo Validation.

Đối với cùng một bài báo, mỗi mô hình LDA tạo ra một vector xác suất có 15 phần tử.

Trước khi so sánh, thứ tự các cột topic của mô hình thứ hai được căn chỉnh theo kết quả Hungarian matching.

Sau đó, Cosine Similarity được tính giữa hai vector phân phối topic.

Cosine Similarity càng gần 1 cho thấy hai phân phối càng tương đồng về hướng.

Giá trị cuối cùng là trung bình Cosine Similarity trên toàn bộ bài báo Validation.

Lưu ý: Việc căn chỉnh bằng Hungarian dựa trên từ khóa, vì vậy điểm Cosine cũng phụ thuộc vào chất lượng ghép topic.

### 3.4. So sánh theo seed tham chiếu

Seed 42 được chọn làm mô hình tham chiếu.

Các mô hình còn lại được so sánh với mô hình này.

Phép so sánh seed 42 với chính nó được giữ lại để kiểm tra tính đúng đắn của thuật toán ghép và căn chỉnh topic.

### 3.5. So sánh toàn bộ các cặp seed

Để tránh phụ thuộc quá nhiều vào seed tham chiếu, M09 thực hiện thêm đánh giá giữa tất cả các cặp seed khác nhau.

Với 5 random seeds, số cặp duy nhất là:

C(5,2) = 5 × 4 / 2 = 10 cặp.

Các cặp gồm:

- 42–52, 42–62, 42–72, 42–82.
- 52–62, 52–72, 52–82.
- 62–72, 62–82.
- 72–82.

Không tính các phép so sánh một seed với chính nó khi tổng hợp kết quả toàn cặp.

Kết quả trung bình trên 10 cặp là chỉ số tổng hợp chính của thực nghiệm M09.

## 4. Kết quả thực nghiệm

### 4.1. Kết quả so với seed tham chiếu

| Seed tham chiếu | Seed so sánh | Jaccard trung bình | Cosine trung bình |
|---:|---:|---:|---:|
| 42 | 42 | 1.000000 | 1.000000 |
| 42 | 52 | 0.417206 | 0.614288 |
| 42 | 62 | 0.480944 | 0.700272 |
| 42 | 72 | 0.411963 | 0.720019 |
| 42 | 82 | 0.463291 | 0.679198 |

Kết quả seed 42 so với chính nó có Jaccard và Cosine đều bằng 1, phù hợp với kỳ vọng của phép kiểm tra tự so sánh.

Các seed còn lại cho thấy sự thay đổi về từ khóa đại diện và phân phối topic.

### 4.2. Kết quả so sánh toàn bộ 10 cặp seed

| Seed A | Seed B | Jaccard trung bình | Cosine trung bình |
|---:|---:|---:|---:|
| 42 | 52 | 0.417206 | 0.614288 |
| 42 | 62 | 0.480944 | 0.700272 |
| 42 | 72 | 0.411963 | 0.720019 |
| 42 | 82 | 0.463291 | 0.679198 |
| 52 | 62 | 0.419412 | 0.635157 |
| 52 | 72 | 0.454728 | 0.637271 |
| 52 | 82 | 0.440728 | 0.643854 |
| 62 | 72 | 0.461152 | 0.692398 |
| 62 | 82 | 0.530217 | 0.788937 |
| 72 | 82 | 0.477215 | 0.701336 |

**Kết quả trung bình trên 10 cặp:**

- Mean Keyword Jaccard: **0.455685**.
- Mean Document Cosine: **0.681273**.

Đây là hai kết quả chính được sử dụng để nhận xét mức độ ổn định của mô hình LDA tại K = 15.

## 5. Phân tích và thảo luận

### 5.1. Độ ổn định của từ khóa topic

Jaccard trung bình đạt 0.455685.

Kết quả này cho thấy các topic sau khi được ghép có một phần từ khóa trùng nhau, nhưng sự trùng khớp chưa cao.

Điều đó phản ánh sự nhạy cảm của LDA đối với random seed trong việc hình thành các tập từ khóa đại diện.

Tuy nhiên, không nên tự động xem 0.455685 là kết quả không đạt nếu nhóm chưa có ngưỡng đánh giá stability được thống nhất.

### 5.2. Độ ổn định của phân phối topic

Cosine trung bình đạt 0.681273.

Kết quả cho thấy các mô hình vẫn tạo ra những phân phối topic có mức độ tương đồng nhất định trên cùng tập Validation.

Tuy nhiên, phân phối topic vẫn có sự biến động giữa các lần huấn luyện.

Việc điểm Cosine cao hơn điểm Jaccard không có nghĩa hai chỉ số đo cùng một đại lượng, vì một chỉ số so sánh tập từ khóa và chỉ số còn lại so sánh vector xác suất.

### 5.3. Các cặp seed đáng chú ý

Cặp seed 62–82 đạt giá trị cao nhất ở cả hai chỉ số:

- Jaccard = 0.530217.
- Cosine = 0.788937.

Cặp seed 42–72 có Jaccard thấp nhất:

- Jaccard = 0.411963.

Cặp seed 42–52 có Cosine thấp nhất:

- Cosine = 0.614288.

Sự khác biệt giữa các cặp seed cho thấy kết quả huấn luyện LDA không hoàn toàn giống nhau khi thay đổi khởi tạo ngẫu nhiên.

## 6. Liên hệ với kết quả M08

Trong M08, nhóm thực nghiệm các giá trị:

`K = [4, 6, 8, 10, 12, 15, 20]`

Kết quả cho thấy:

- K = 20 đạt Coherence cao nhất trong các giá trị thử nghiệm.
- K = 8 và K = 10 đạt Topic Diversity cao nhất.
- K = 15 được đề xuất là ứng viên cân bằng giữa Coherence, Topic Diversity và khả năng diễn giải sơ bộ.

M09 tiếp tục khảo sát K = 15 để đánh giá độ nhạy với random seed.

Kết quả M09 cho thấy các topic tại K = 15 vẫn thay đổi đáng kể giữa những lần huấn luyện.

Như vậy, kết quả M09 bổ sung thêm một tiêu chí đánh giá cho M08, nhưng không phủ nhận hoàn toàn đề xuất K = 15.

Để khẳng định K = 15 ổn định hơn hoặc kém hơn K = 20, cần thực hiện thí nghiệm stability tương đương với K = 20.

## 7. Hướng dẫn tái lập thực nghiệm

### 7.1. Cài đặt thư viện

Tại thư mục gốc của repository:

```powershell
python -m pip install -r requirements.txt
```

### 7.2. Chạy thực nghiệm M09

```powershell
python -m experiments.lda.run_m09_lda_stability
```

Có thể chỉ định file cấu hình:

```powershell
python -m experiments.lda.run_m09_lda_stability --config configs/lda_stability.yaml
```

### 7.3. Chạy kiểm thử M09

```powershell
python -m pytest tests/test_lda_stability.py -v
```

Kết quả đã ghi nhận:

`12 passed in 0.98s`

### 7.4. Kiểm thử kết hợp M07, M08 và M09

```powershell
python -m pytest tests/test_lda_pipeline.py tests/test_lda_metrics.py tests/test_lda_stability.py -q
```

Kết quả đã ghi nhận:

`25 passed in 2.77s`

## 8. Cấu trúc file và kết quả đầu ra

### File cấu hình

`configs/lda_stability.yaml`

### File xử lý và đánh giá

- `src/evaluation/lda_stability.py`
- `experiments/lda/run_m09_lda_stability.py`

### File kiểm thử

`tests/test_lda_stability.py`

### Các file CSV kết quả

Thư mục: `results/lda/m09/`

| File | Nội dung | Số dòng dữ liệu |
|---|---|---:|
| stability_metrics.csv | Chỉ số so với seed tham chiếu | 5 |
| topic_matches.csv | Kết quả ghép 15 topic theo từng seed | 75 |
| runtime_by_seed.csv | Thời gian thực hiện của từng seed | 5 |
| pairwise_stability.csv | Đánh giá toàn bộ cặp seed | 10 |

Cả bốn file CSV đã được xuất thành công trong lần thực nghiệm.

## 9. Hạn chế

Thực nghiệm M09 hiện có một số hạn chế:

1. Chỉ đánh giá stability tại K = 15.
2. Chỉ sử dụng 5 random seeds.
3. Hungarian matching dựa trên Jaccard của Top-10 keywords, chưa xem xét đầy đủ độ tương đồng ngữ nghĩa.
4. Cosine Similarity được tính sau khi căn chỉnh topic dựa trên từ khóa.
5. Chưa xây dựng khoảng tin cậy thống kê hoặc đối chiếu với một chuẩn stability độc lập.
6. Script hiện tại lưu bảng kết quả CSV nhưng chưa lưu riêng tất cả mô hình đã huấn luyện.
7. Các bài kiểm thử hiện tại xác minh chức năng của các thành phần chính, chưa thay thế việc tái lập toàn bộ thực nghiệm trên môi trường độc lập.

## 10. Kết luận

Task M09 đã triển khai thành công quy trình đánh giá độ ổn định của mô hình LDA với nhiều random seeds trên Dataset v1.0.

Năm mô hình LDA được huấn luyện với cùng dữ liệu, số topic K = 15 và các siêu tham số cố định, chỉ thay đổi random seed.

Kết quả đánh giá 10 cặp seed cho thấy:

- Mean Keyword Jaccard = **0.455685**.
- Mean Document Cosine = **0.681273**.

Các kết quả này cho thấy LDA có sự nhạy cảm đáng kể với random seed, đặc biệt đối với tập từ khóa đại diện cho topic.

M09 cung cấp bằng chứng thực nghiệm bổ sung cho quá trình lựa chọn mô hình từ M08.

Trong các bước nghiên cứu tiếp theo, nhóm có thể so sánh stability giữa K = 15 và K = 20, kết hợp Coherence, Diversity và đánh giá thủ công để lựa chọn cấu hình LDA cuối cùng.
