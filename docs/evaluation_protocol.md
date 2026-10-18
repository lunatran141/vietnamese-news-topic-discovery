# E01 — Evaluation protocol chung cho năm phương pháp

## 1. Mục tiêu và phạm vi

Tài liệu này đề xuất protocol đánh giá chung cho năm phương pháp của đồ án:

1. KMeans với TF-IDF.
2. NMF.
3. LDA.
4. HDBSCAN với sentence embeddings.
5. BERTopic.

E01 tương ứng với phần việc 4.6 của TV4: xây **metric protocol** phù hợp từng họ mô hình, quy tắc ghi runtime và định dạng kết quả chung. Protocol này giúp các thành viên tạo kết quả có thể kiểm tra và so sánh mà không ép những metric khác bản chất vào cùng một thang điểm.

**Trạng thái:** bản đề xuất. TV4 và các thành viên phụ trách bốn phương pháp còn lại cần xác nhận những quy tắc áp dụng cho phương pháp mình trước khi coi E01 là protocol chính thức.

E01 quy định **cách đánh giá**. Bảng kết quả so sánh đủ năm phương pháp thuộc E03; benchmark thời gian và tài nguyên đầy đủ thuộc E08. Không cần hoàn thành E03 hoặc E08 mới có thể chốt protocol E01.

Các kết quả đã tạo trước khi thống nhất protocol phải được mô tả theo đúng cách tính ban đầu. Không đổi tên metric cũ hoặc diễn giải lại chúng như thể tất cả đã được tính theo cùng một quy tắc.

## 2. Dữ liệu và vai trò của từng split

Sử dụng split do D05 cung cấp:

- Train: `data/splits/train_ids.csv` — 6.493 bài.
- Validation: `data/splits/validation_ids.csv` — 1.391 bài.
- Test: `data/splits/test_ids.csv` — 1.390 bài.
- Manifest tổng: `data/splits/split_manifest.csv` — 9.274 bài.

Dữ liệu văn bản, nhãn tham chiếu và kết quả mô hình được ghép bằng `article_id`. Mỗi phương pháp được dùng biến thể tiền xử lý phù hợp, nhưng phải ghi rõ biến thể, cột đầu vào và cấu hình.

Trước khi tính metric, kiểm tra:

- Số bài đúng với split được công bố.
- Không có `article_id` rỗng hoặc trùng.
- Không có ID nằm ngoài split.
- Mỗi bài có đúng một nhãn dự đoán nếu đang tính metric dựa trên nhãn đơn.
- Nhãn tham chiếu được ghép bằng ID, không ghép theo vị trí dòng CSV.

### Vai trò của train, validation và test

1. **Train:** fit vectorizer, mô hình chủ đề, phép giảm chiều và các thành phần cần học từ dữ liệu.
2. **Validation:** dùng các thành phần đã fit trên train để biến đổi hoặc dự đoán bài mới; dùng để so sánh và chọn cấu hình.
3. **Test:** đánh giá cuối sau khi đã chốt phương pháp, cấu hình, metric và cách xử lý các bài không được gán chủ đề.

Không fit lại vectorizer, UMAP hoặc mô hình phân cụm trên validation rồi gọi các nhãn mới là dự đoán của mô hình train. Nếu một phương pháp chạy thí nghiệm phân cụm độc lập trên validation, phải đặt tên riêng và mô tả rõ sự khác biệt đó.

### Chính sách đối với test

Không dùng test để dò siêu tham số, chọn số chủ đề, chọn embedding, chọn cách tiền xử lý hoặc điều chỉnh ngưỡng. Sau khi xem kết quả test, không thay đổi cấu hình dựa trên kết quả đó rồi báo cáo như một phép đánh giá độc lập.

Nếu repo đã có kết quả test từ một thí nghiệm trước, ghi rõ mục đích và thời điểm tạo kết quả. Kết quả test đã xem không được coi lại là validation.

## 3. Điều kiện để hai kết quả được so sánh

Một phép so sánh phải ghi rõ:

- Split và tập `article_id` thực sự được đánh giá.
- Đầu vào văn bản, biến thể tiền xử lý và biểu diễn vector.
- Thành phần nào được fit trên train.
- Cách tạo nhãn cho bài validation/test.
- Cách xử lý bài `-1`, nếu phương pháp có cơ chế nhiễu.
- Định nghĩa metric và mẫu số của các tỷ lệ.
- Seed, cấu hình, phiên bản dữ liệu và đường dẫn kết quả.

Silhouette hoặc Davies–Bouldin tính trong những không gian vector khác nhau chỉ dùng để xem xét **trong cùng họ thí nghiệm**. Không dùng điểm hình học của KMeans trên TF-IDF để xếp hạng trực tiếp với HDBSCAN trên UMAP.

Không so một metric tính trên toàn bộ 1.391 bài validation với metric chỉ tính trên các bài được gán chủ đề mà không ghi rõ hai tập bài khác nhau.

## 4. Nhãn tham chiếu và metric dùng chung

`category_gold` là chuyên mục báo dùng làm nhãn tham chiếu. Chuyên mục báo có thể rộng hơn một chủ đề nhỏ; một sự kiện cũng có thể liên quan nhiều chuyên mục. Vì vậy, độ khớp với `category_gold` không đại diện cho toàn bộ chất lượng khám phá chủ đề.

### ARI và NMI

Khi một phương pháp xuất được **một nhãn dự đoán cho mỗi bài** và có `category_gold` trên cùng split, báo cáo:

- **ARI (Adjusted Rand Index):** mức tương đồng giữa hai phép phân nhóm sau khi điều chỉnh sự trùng khớp ngẫu nhiên.
- **NMI (Normalized Mutual Information):** lượng thông tin chung giữa nhãn dự đoán và chuyên mục tham chiếu.

ARI và NMI là **metric chung bắt buộc trong điều kiện trên**. Nếu phương pháp chưa xuất được nhãn dự đoán cho cùng tập bài, ghi `chưa tính` và lý do; không tự điền `0`. Nếu metric thật sự không áp dụng, ghi `không áp dụng` và giải thích.

Quy tắc tính:

1. Ghép `category_gold` và nhãn dự đoán bằng `article_id`.
2. Xác nhận mỗi bài có đúng một nhãn dự đoán.
3. Tính ARI/NMI trên **toàn bộ bài của split**.
4. Với HDBSCAN hoặc BERTopic, xem `-1` là một nhãn dự đoán khi tính metric trên toàn split.
5. Báo riêng số bài `-1` và tỷ lệ được gán chủ đề để người đọc biết độ bao phủ.
6. Nếu tính thêm ARI/NMI chỉ trên các bài khác `-1`, ghi tên là **assigned-only ARI/NMI**, kèm số bài được giữ lại. Không dùng kết quả này thay thế ARI/NMI toàn split.

Với NMF và LDA, khi cần nhãn đơn, dùng chủ đề có trọng số hoặc xác suất cao nhất của mỗi bài. Phải ghi rõ quy tắc chuyển từ phân bố chủ đề sang nhãn đơn.

ARI/NMI chỉ phản ánh mức khớp với `category_gold`. Cần đọc từ khóa, bài đại diện và trường hợp khó giải thích trước khi kết luận phương pháp nào tạo chủ đề tốt hơn.

## 5. Metric riêng cho từng họ phương pháp

### 5.1. KMeans với TF-IDF

Báo cáo số cụm `K`, phân bố kích thước cụm, Silhouette, Davies–Bouldin, Calinski–Harabasz và độ ổn định giữa các seed nếu thí nghiệm có nhiều seed. Inertia có thể dùng để xem xu hướng khi thay đổi `K`.

Báo ARI/NMI theo Mục 4 khi có nhãn dự đoán trên cùng split. Đọc từ khóa và bài đại diện để kiểm tra cụm có nhất quán về nội dung không.

Metric hình học được diễn giải trong không gian TF-IDF của KMeans; không so trực tiếp giá trị của chúng với metric hình học trong embedding hoặc UMAP.

Kết quả hiện có: `results/kmeans/kmeans_k_search_metrics.csv` và `results/kmeans/kmeans_manual_evaluation.csv`.

### 5.2. NMF

Báo cáo số chủ đề, coherence theo đúng công thức và tập văn bản sử dụng, topic diversity, reconstruction error nếu có, trạng thái hội tụ, từ khóa và bài tiêu biểu.

Khi tính ARI/NMI, lấy chủ đề có trọng số lớn nhất làm nhãn đơn cho bài. Đây là phép rút gọn phân bố chủ đề; phân tích định tính vẫn cần xem những bài có trọng số đáng kể ở nhiều chủ đề.

Không so trực tiếp reconstruction error của NMF với inertia của KMeans hoặc tỷ lệ nhiễu của HDBSCAN.

Kết quả NMF hiện có trong `results/nmf/`.

### 5.3. LDA

Báo cáo số chủ đề, coherence `c_v`, coherence `c_npmi`, topic diversity, từ khóa và bài tiêu biểu. Ghi rõ cách tách từ, số từ khóa mỗi chủ đề và tập văn bản dùng để tính coherence.

Khi tính ARI/NMI, lấy chủ đề có xác suất cao nhất làm nhãn đơn cho bài. Dùng cùng quy tắc giữa các cấu hình LDA.

Không chọn mô hình chỉ vì coherence cao nhất. Cần kiểm tra chủ đề trùng lặp, quá rộng hoặc khó đặt tên.

Kết quả thử số chủ đề hiện có: `results/lda/m08/topic_number_results.csv`.

### 5.4. HDBSCAN với sentence embeddings

Báo cáo embedding run, cấu hình UMAP, `min_cluster_size`, `min_samples`, cách chọn cụm (`eom` hoặc `leaf`), số cụm, kích thước cụm, số bài `-1`, tỷ lệ `-1` và tỷ lệ bài được gán cụm.

Với validation/test:

1. Tạo embedding bằng cùng mô hình và cách xử lý văn bản như train.
2. Dùng UMAP đã fit trên train để biến đổi bài mới.
3. Dùng HDBSCAN đã fit trên train để gán bài mới vào các cụm train.

Phép gán gần đúng cho bài mới không tạo cụm mới. Không fit lại HDBSCAN trên validation rồi coi nhãn mới là dự đoán của mô hình train.

Tính ARI/NMI theo Mục 4 và đọc thêm bài đại diện, bài `-1`. Tỷ lệ nhiễu thấp hơn không tự chứng minh cụm tốt hơn.

Kết quả M11 trên train: `results/hdbscan/m11_summary.csv`. Validation hiện đã chạy `minilm_title_content` và `mpnet_title_content`, mỗi embedding với `eom_15_5` và `leaf_15_5`, trên cùng 1.391 bài. Xem `docs/m11_validation_guide.md`.

### 5.5. BERTopic

Báo cáo số chủ đề, kích thước chủ đề, số bài `-1`, tỷ lệ `-1`, tỷ lệ bài được gán chủ đề, topic diversity, từ khóa và bài đại diện. Ghi rõ embedding, UMAP, HDBSCAN và cách gán bài validation/test.

Tính ARI/NMI theo Mục 4 nếu đã có nhãn dự đoán cho toàn bộ split. Xử lý `-1` nhất quán với HDBSCAN trong metric chung. Đọc mẫu để phát hiện chủ đề trùng nhau, quá rộng hoặc khó giải thích.

Kết quả hiện có trong `results/bertopic_experiments/summary.csv` và thư mục của từng run.

## 6. Quy tắc chọn cấu hình trên validation

Trước khi xem test, người phụ trách mỗi phương pháp cần ghi danh sách cấu hình ứng viên và tiêu chí chọn. Quy trình đề xuất:

1. Loại cấu hình chạy lỗi, thiếu bài, ghép sai ID hoặc không thể tái tạo.
2. Xem metric riêng trong cùng họ phương pháp; ghi cả ưu điểm và điểm yếu.
3. Xem ARI/NMI như chỉ báo về mức khớp với chuyên mục. Nếu đã dùng hai điểm này để chọn cấu hình, ghi rõ.
4. Đọc bài đại diện và một mẫu bài khó theo cách lấy mẫu được ghi lại.
5. Chốt cấu hình và ghi lý do, giới hạn, seed cùng đường dẫn artifact.
6. Chỉ sau khi chốt lựa chọn mới đánh giá trên test.

Không chọn HDBSCAN hoặc BERTopic chỉ vì có ít bài `-1`; không chọn LDA hoặc NMF chỉ vì coherence cao nhất; không chọn KMeans chỉ vì một metric hình học cao nhất.

Nếu nhóm dùng điểm tổng hợp để xếp hạng, phải thống nhất trước metric nào được đưa vào, chiều tốt/xấu, cách chuẩn hóa và trọng số. Không cộng trực tiếp các metric có thang đo khác nhau.

### Nhận xét về MiniLM và MPNet hiện tại

Trên validation, `minilm_title_content + eom_15_5` có tỷ lệ nhiễu thấp hơn, ARI và NMI cao hơn `mpnet_title_content + eom_15_5`. Điều này cho thấy MiniLM EOM khớp `category_gold` hơn trong phép đo hiện tại; chưa đủ để kết luận chất lượng chủ đề tổng thể tốt hơn.

Với `leaf_15_5`, hai embedding có tỷ lệ nhiễu và ARI/NMI gần nhau. Cần xem cụm và bài đại diện trước khi chọn cấu hình cuối.

## 7. Protocol đo runtime

E01 quy định **cách đo và ghi runtime**. E08 sẽ thực hiện benchmark thời gian và tài nguyên đầy đủ cho năm phương pháp. Không ước lượng runtime của những run cũ nếu lúc chạy không đo đúng các giai đoạn dưới đây.

### Giai đoạn cần tách

- `representation_seconds`: thời gian tạo hoặc biến đổi biểu diễn văn bản cho split được xét, ví dụ TF-IDF, BoW hoặc sentence embeddings. Ghi rõ thao tác `fit_transform`, `transform` hay `encode`.
- `fit_seconds`: thời gian fit mô hình chủ đề và các thành phần của pipeline cần học từ train, ví dụ UMAP và HDBSCAN. Nếu giai đoạn này chứa nhiều bước, có thể ghi thêm thời gian từng bước.
- `predict_seconds`: thời gian biến đổi và gán chủ đề cho validation/test bằng mô hình đã fit, **không gồm** thời gian tạo biểu diễn văn bản nếu đã ghi ở `representation_seconds`.
- `evaluation_seconds`: thời gian tính metric từ nhãn hoặc kết quả đã có; ghi riêng, không gộp vào thời gian dự đoán.

Dùng đồng hồ đo thời gian thực như `time.perf_counter()`; đơn vị lưu là **giây**. Ghi điểm bắt đầu và kết thúc quanh đúng công đoạn được đo. Không lấy thời gian từ log của những run khác nhau rồi cộng nếu chúng không được đo theo cùng ranh giới.

### Điều kiện đo cần ghi kèm

- CPU, RAM và GPU nếu sử dụng.
- Phiên bản Python và thư viện chính.
- Seed và cấu hình.
- Số bài của split.
- Batch size đối với bước mã hóa.
- Trạng thái model đã có trong cache hay phải tải lần đầu.

Thời gian tải dependency hoặc tải model từ mạng không đưa vào `fit_seconds` hay `predict_seconds`. Nếu muốn báo thời gian khởi tạo/tải lần đầu, ghi thành mục riêng `setup_seconds`.

Với các cấu hình dùng chung một bộ embedding hoặc vectorizer, không cộng lặp lại chi phí tạo biểu diễn vào từng cấu hình mà không ghi chú. Báo riêng chi phí dùng chung và chi phí của từng run. Khi so runtime giữa phương pháp, phải xác nhận các giai đoạn, máy chạy và điều kiện cache tương đương.

Khi có thể, chạy lặp lại theo cùng số lần và báo trung bình cùng độ lệch chuẩn; nếu chỉ chạy một lần, ghi rõ `n_repeats = 1`. Việc đo và so sánh đầy đủ này thuộc E08.

## 8. Định dạng đầu ra chung

Mỗi phương pháp vẫn được giữ file kết quả riêng, nhưng khi đưa vào bảng tổng hợp E03 phải xuất dữ liệu theo cùng cấu trúc. **Một hàng tương ứng với một `method_id + run_id + split`.**

### 8.1. Bảng assignments

Nếu phương pháp tạo nhãn cho từng bài, file assignments tối thiểu gồm:

- `article_id`: khóa duy nhất trong một run và split.
- `split`: `train`, `val` hoặc `test`.
- `method_id`: `kmeans`, `nmf`, `lda`, `hdbscan` hoặc `bertopic`.
- `run_id`: tên cấu hình đủ phân biệt với các run khác.
- `predicted_label`: nhãn cụm/chủ đề dùng cho metric chung.
- `config_id` hoặc `config_hash`: mã nhận diện cấu hình nếu có.

Có thể thêm xác suất chủ đề, membership strength, nhãn `-1` hoặc các cột riêng của phương pháp. Khi ghép nhiều file, khóa là `method_id + run_id + split + article_id`.

### 8.2. Bảng metric tổng hợp

Một hàng ứng với `method_id + run_id + split`. Các cột chung đề xuất:

- `method_id`
- `run_id`
- `split`
- `n_articles`
- `n_unique_article_ids`
- `n_groups_train`: số cụm hoặc chủ đề của mô hình train
- `ari`
- `nmi`
- `n_noise`
- `noise_percent`
- `coverage_percent`
- `runtime_seconds`
- `runtime_scope`: những giai đoạn đã được tính vào `runtime_seconds`
- `seed`
- `config_path`
- `assignments_path`
- `method_metrics_path`
- `artifact_path`
- `notes`

Các metric riêng như coherence, Silhouette hoặc reconstruction error nằm trong `method_metrics_path` hoặc các cột có tên và định nghĩa rõ. Không đổi tên nhiều metric khác nhau thành một cột chung mơ hồ như `score`.

`n_noise`, `noise_percent` và `coverage_percent` chỉ dùng để so sánh các phương pháp có cơ chế bài không được gán chủ đề. Với phương pháp không có khái niệm nhiễu, để trống các cột không áp dụng và ghi lý do trong `notes`; không điền `0%` rồi diễn giải là phương pháp xử lý nhiễu tốt hơn.

Nếu chưa đo runtime theo Mục 7, để trống `runtime_seconds` và ghi `chưa đo` trong `notes`. Không dùng `0` thay cho giá trị thiếu.

### 8.3. Giá trị thiếu và kiểm tra schema

- `0` chỉ dùng khi metric thực sự bằng không.
- `chưa tính`: có thể tính nhưng chưa có kết quả.
- `không áp dụng`: metric không có ý nghĩa với phương pháp đó.
- Cột số để trống khi thiếu giá trị; lý do được ghi trong `notes` hoặc tài liệu đi kèm.

Trước khi gộp bảng, kiểm tra không có hai hàng trùng khóa `method_id + run_id + split`. Mọi tỷ lệ phần trăm phải dùng thang **0–100**, có tên cột kết thúc bằng `_percent` và nêu rõ mẫu số.

## 9. Đánh giá định tính và phân tích lỗi

Mỗi phương pháp cần có ví dụ bài thực tế cho:

- Chủ đề hoặc cụm nhất quán, dễ giải thích.
- Chủ đề hoặc cụm lẫn nhiều nội dung.
- Bài khó gán hoặc bị gán sai.
- Bài `-1` đối với phương pháp có cơ chế nhiễu.

Với HDBSCAN, M12 đã phân tích mẫu bài `-1` bằng các nhãn `data_noise`, `rare_valid`, `multi_topic`, `valid_outlier` và `uncertain`. Đây là nhận xét trên mẫu, không phải nhãn chuẩn của toàn bộ bài `-1`.

E05 / phần việc 4.7 quy định riêng đánh giá bởi ít nhất hai người chấm, thang điểm và cách giải quyết bất đồng. E01 chỉ quy định vị trí của đánh giá định tính trong quy trình chọn và báo cáo mô hình.

## 10. Kiểm tra trước khi công bố kết quả

- [ ] `article_id` đúng split, không trùng và không thiếu ngoài dự kiến.
- [ ] Số nhãn dự đoán bằng số bài được đánh giá.
- [ ] Không có NaN hoặc Inf trong đầu vào và metric cần tính.
- [ ] Các thành phần cần fit đã được fit trên train.
- [ ] Cấu hình validation không được chọn bằng test.
- [ ] Bài `-1` và mẫu số của các tỷ lệ được xử lý đúng quy tắc.
- [ ] ARI/NMI dùng cùng tập ID và cùng chính sách đối với `-1`.
- [ ] Runtime có đơn vị, phạm vi đo và điều kiện máy chạy.
- [ ] Một hàng metric ứng với đúng một `method_id + run_id + split`.
- [ ] Có config, seed và đường dẫn kết quả để tái tạo.
- [ ] Nhận xét định tính có `article_id` để kiểm tra lại.

## 11. Tình trạng áp dụng hiện tại

Các phương pháp đã có kết quả riêng, nhưng chưa phải tất cả đều xuất bảng theo schema chung ở Mục 8. Tài liệu này xác định protocol đề xuất; không tuyên bố năm phương pháp đã hoàn tất toàn bộ đánh giá trên validation và test.

Riêng HDBSCAN, đã kiểm tra bốn run trên cùng 1.391 bài validation:

- `mpnet_title_content + eom_15_5`: 528 bài `-1`; nhiễu 37,96%; ARI 0,0464; NMI 0,4101.
- `mpnet_title_content + leaf_15_5`: 677 bài `-1`; nhiễu 48,67%; ARI 0,0098; NMI 0,3562.
- `minilm_title_content + eom_15_5`: 484 bài `-1`; nhiễu 34,80%; ARI 0,1303; NMI 0,4218.
- `minilm_title_content + leaf_15_5`: 682 bài `-1`; nhiễu 49,03%; ARI 0,0093; NMI 0,3592.

Các nhãn train tạo lại đã khớp kết quả M11 đã lưu. Sau khi mở rộng runner cho MiniLM, bộ test HDBSCAN và HDBSCAN validation đạt **6 passed**; bộ test M10–M12 cùng validation HDBSCAN đạt **19 passed** ở phiên bản hiện tại.

ARI, NMI, tỷ lệ nhiễu và tỷ lệ bài được gán cụm được tính từ assignments bằng `experiments/hdbscan/evaluate_m11_validation.py` và lưu tại `results/hdbscan/validation/m11_validation_metrics.csv`. Phương pháp validation được trình bày trong `docs/m11_validation_guide.md`.

Các run HDBSCAN hiện **chưa có runtime được đo theo Mục 7**. Không ước lượng runtime từ những lần chạy trước hoặc điền `0`; việc đo đầy đủ có thể thực hiện trong E08.

Để chốt E01, TV4 và các thành viên phụ trách KMeans, NMF, LDA, BERTopic cần xác nhận: metric riêng, cách tạo assignments trên validation/test, ngoại lệ đối với metric chung và schema đầu ra. E03 sẽ dùng protocol đã chốt để lập bảng so sánh năm phương pháp.

## 12. Tài liệu và kết quả liên quan

- Quy tắc chia dữ liệu: `docs/d05_guide.md`.
- KMeans: `docs/m02_kmeans_guide.md`, `results/kmeans/`.
- NMF: `results/nmf/`.
- LDA: `docs/m08_lda_experiments.md`, `results/lda/`.
- HDBSCAN train: `docs/m11_hdbscan_guide.md`, `results/hdbscan/`.
- HDBSCAN validation: `docs/m11_validation_guide.md`, `results/hdbscan/validation/`.
- Phân tích outlier: `docs/m12_outlier_analysis.md`, `results/hdbscan/outlier_analysis/`.
- BERTopic: `docs/m14_bertopic_experiments.md`, `results/bertopic_experiments/`.