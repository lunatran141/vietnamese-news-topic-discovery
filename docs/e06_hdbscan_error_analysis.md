# E06 — Phân tích lỗi và giới hạn của HDBSCAN

**Người phụ trách:** TV4 — Vân
**Phạm vi:** Phần HDBSCAN đóng góp cho Error Analysis chung của nhóm

## 1. Mục tiêu và nguồn bằng chứng

Phần này tổng hợp các ca khó và giới hạn của HDBSCAN sau các bước so sánh cấu hình (E03), kiểm tra độ ổn định (E04) và đánh giá định tính (E05/4.7). Mục tiêu là giải thích *vì sao* một số cụm khó diễn giải hoặc một số bài bị gán nhãn nhiễu `-1`.

Bảng ca được lưu ở `results/hdbscan/error_analysis/hdbscan_error_cases.csv`, gồm:

- **6 bài validation** từ đánh giá định tính cấu hình `minilm_title_content + eom_15_5`. Các bài này thuộc các cụm T01, T03, T04 và T05 trong `results/hdbscan/human_eval/`.
- **4 bài train** từ phân tích outlier M12 của cấu hình `mpnet_title_content + eom_15_5`.

Các bài được chọn để minh họa từng dạng trường hợp đã được đọc trước đó. Đây **không phải mẫu ngẫu nhiên đại diện** để ước lượng tỷ lệ lỗi trên toàn bộ 1.391 bài validation hoặc 6.493 bài train. Kết quả của hai split và hai embedding được trình bày riêng, không cộng thành một tỷ lệ lỗi chung.

Nguồn đối chiếu chính:

- `docs/e03_hdbscan_contribution.md`: chỉ số validation và phân tích cụm.
- `docs/e04_hdbscan_stability.md`: độ nhạy với seed UMAP.
- `docs/hdbscan_human_eval.md`: phiếu chấm R1/R2 và giải thích các cụm.
- `docs/m12_outlier_analysis.md`: phân loại các bài train được gán `-1`.
- `results/hdbscan/error_analysis/hdbscan_error_cases.csv`: danh sách 10 ca dùng trong phần này.

Trong `hdbscan_error_cases.csv`, cột `source_file` của các ca HS lấy từ bộ bài mẫu `evaluation_samples.csv`. Nội dung cột `evidence` được lấy từ `reviewer_R1_documents.csv` và `reviewer_R2_documents.csv`, cột `interpretation` được rà và điền sau khi tạo bảng ca.

## 2. Các nhóm trường hợp trên validation

### 2.1. Tin trùng sự kiện trong một cụm có ranh giới rộng

**HS001 — `VNE_4e84cfb652e9`** và **HS004 — `VNN_f0d233a65f6a`** đều đưa tin về vụ kiện sáng chế Taptic Engine của Apple. Đây là hai bài từ hai nguồn khác nhau về gần như cùng một sự kiện. Trong phiếu đánh giá, cả R1 và R2 đều chấm mỗi bài **2/5** về mức phù hợp với chủ đề được đề xuất cho T01.

`duplicate` trong bảng ca mô tả **quan hệ trùng sự kiện giữa HS001 và HS004**. Bản thân việc hai bài cùng sự kiện nằm chung cụm không phải lỗi. Vấn đề về ý nghĩa của T01 là cụm còn gộp tin về vụ kiện này với các bài có trọng tâm iPhone Duo màn hình gập và các tin Apple khác. Các bài có chung thực thể “Apple/iPhone” nhưng không cùng chủ đề cụ thể.

**Nhận xét:** Khi diễn giải T01, cần xem nội dung và sự kiện của từng bài, thay vì chỉ dựa vào thực thể chung hoặc từ khóa của cụm.

### 2.2. Cụm bao phủ nhiều trọng tâm

**HS013 — `TTR_42079ce07920`** là bài phỏng vấn về quản trị và phát triển Cần Thơ. Bài có nhắc tới phát triển hạ tầng, nhưng phạm vi rộng hơn các tin về luật đô thị và metro trong T03. R1 và R2 đều chấm bài **2/5**. Ca này minh họa một bài chỉ liên quan gián tiếp với trọng tâm của cụm.

T05 cũng có ranh giới rộng trong lĩnh vực y tế:

- **HS023 — `TTR_02e6c53cad15`:** tin lễ kỷ niệm bệnh viện, liên quan ngành y nhưng khác trọng tâm tư vấn sức khỏe, bệnh lý hoặc điều trị của nhiều bài khác. R1 chấm **2/5**, R2 chấm **3/5**.
- **HS024 — `TTR_07f28315c701`:** trọng tâm là trẻ sơ sinh bị bỏ rơi; bệnh viện chỉ xuất hiện trong diễn biến sự việc. Cả R1 và R2 đều chấm **1/5**. Đây là ca lạc trọng tâm rõ nhất trong các bài T05 được chọn.

Nội dung HS024 có dấu hiệu xuất hiện thêm một mẩu tin khác ở cuối. Đây là **nghi vấn cần kiểm tra với bài nguồn**, chưa đủ căn cứ để kết luận chắc chắn lỗi nằm ở dữ liệu gốc, crawler hay bước tiền xử lý.

**Nhận xét:** Một tên cụm rộng như “y tế và sức khỏe” có thể bao quát nhiều bài, nhưng kém hữu ích nếu mục tiêu là khám phá một bệnh, phương pháp điều trị hoặc sự kiện cụ thể. HS024 còn cho thấy việc xuất hiện từ “bệnh viện” không đồng nghĩa trọng tâm bài là y tế.

### 2.3. Bài có nhiều trọng tâm nhưng vẫn phù hợp với cụm

**HS016 — `VNE_29086a9d7dfc`** bàn về cả giá vàng và chứng khoán Mỹ sau quyết định lãi suất của Fed. Bài được đưa vào T04 về biến động giá vàng. R1 chấm **5/5**, R2 chấm **4/5** về mức phù hợp.

Đây là một ca `multi_topic` cần đọc kỹ, **chưa phải bằng chứng rằng HDBSCAN gán sai cụm**: giá vàng vẫn là nội dung đáng kể của bài. Trường hợp này cho thấy ranh giới chủ đề có thể không rạch ròi đối với tin kinh tế đề cập nhiều thị trường.

## 3. Các dạng outlier trên train

Bốn ca sau đến từ M12, dùng **MPNet trên train** và đều có `cluster_id = -1`. Chúng không thuộc bộ phiếu validation MiniLM ở mục 2.

### 3.1. Bài hợp lệ nhưng bị gán `-1`

**`TTR_607809186810`** là tin lịch thi đấu ASIAD ngày 28/9. Các bài gần được đối chiếu cũng nói về lịch thi đấu hoặc thành tích ASIAD, nhưng bài này vẫn bị gán `-1`. M12 ghi nhãn `valid_outlier`.

Ca này cho thấy `-1` không đồng nghĩa bài báo vô nghĩa hoặc dữ liệu hỏng. Cần kiểm tra nội dung bài và các láng giềng trước khi diễn giải nhãn nhiễu.

### 3.2. Chủ đề có thể quá nhỏ hoặc khó đặt vào cụm rộng

**`TTR_607b52b568c2`** nói về việc vận chuyển 384 tuýp kem đánh răng không có chứng từ. M12 tạm gán `rare_valid` nếu xét tiểu chủ đề vận chuyển hàng không rõ nguồn gốc. Tuy vậy, sự khác nhau giữa các vụ việc riêng lẻ chưa đủ để khẳng định chắc chắn đây là một chủ đề hiếm.

Đây là nhãn phân tích **tạm thời**; không dùng một bài để kết luận mô hình luôn bỏ sót các chủ đề nhỏ.

### 3.3. Đầu vào thiếu ngữ cảnh

**`VNN_14f9af2f4e8b`** là bài về sức mua tại chợ truyền thống và chuyển đổi số. Theo phân tích M12, phần `content` dùng cho embedding thiếu câu dẫn cần thiết và bắt đầu bằng cụm phụ thuộc ngữ cảnh “Trong bối cảnh đó”. M12 ghi nhãn `data_noise` theo nghĩa **văn bản đầu vào thiếu thông tin để biểu diễn bài**.

Ca này không chứng minh toàn bộ bài nguồn hoặc crawler bị lỗi. Cần phân biệt chất lượng văn bản được đưa vào mô hình với chất lượng của bài báo gốc.

### 3.4. Một bản ghi chứa nhiều tin độc lập

**`VNN_a9503ec90cf8`** chứa các mẩu riêng về nhiều nhân vật giải trí, không cùng một sự kiện. M12 ghi nhãn `multi_topic`. Khi một bản ghi chứa nhiều tin độc lập, việc ép nó vào một cụm chủ đề hẹp có thể không phù hợp.

Dạng này khác HS016: HS016 là **một bài kinh tế có nhiều mảng liên quan**, còn bản ghi này gồm **nhiều mẩu tin độc lập**. Hai trường hợp không nên được gộp thành cùng một kết luận về chất lượng gán cụm.

## 4. Chỉ số định lượng và ý nghĩa chủ đề

Theo E03, trong **bốn run HDBSCAN đã so sánh trên validation**, `minilm_title_content + eom_15_5` có các chỉ số tốt nhất: ARI **0,1303**, NMI **0,4218**, tỷ lệ nhiễu **34,80%** và độ bao phủ **65,20%**.

“Đứng đầu trong bốn run” là một so sánh **tương đối**; ARI 0,1303 không được mô tả là cao theo nghĩa tuyệt đối. ARI/NMI ở đây đối chiếu với `category_gold`, tức **chuyên mục báo**, trong khi một chuyên mục có thể gồm nhiều chủ đề hoặc sự kiện.

Các ca T01, T03 và T05 cho thấy ngay cả run được chọn theo validation vẫn có cụm gộp bài khác trọng tâm. Ngược lại, HS016 cho thấy một bài nhiều trọng tâm vẫn có thể phù hợp với cụm. Vì vậy, kết luận về chất lượng chủ đề cần đọc đồng thời chỉ số, phiếu chấm và bài báo cụ thể.

E04 còn ghi nhận độ nhạy với seed UMAP của cấu hình MiniLM + EOM: trên 10 seed, pairwise ARI trung bình là **0,5165 trên train** và **0,4774 trên validation**. Đây là ARI **giữa các lần chạy**, khác ARI so với `category_gold` nêu trên. Kết quả này nhắc rằng số cụm và ranh giới nhiễu của seed 42 không phải giá trị cố định cho mọi lần fit cùng cấu hình.

## 5. Những dạng lỗi chưa được xác nhận

Danh sách yêu cầu E06 của nhóm còn có trường hợp **`narrow` — một chủ đề bị chia thành nhiều cụm quá hẹp**. Mười ca trong bảng hiện tại chưa chứng minh được điều đó. Để báo cáo dạng lỗi này, cần tìm ít nhất hai cụm cụ thể, đối chiếu các `article_id` và nội dung bài, rồi chỉ ra chúng thực sự thuộc cùng một chủ đề ở mức chi tiết đã thống nhất.

Không nên ghi `narrow` là lỗi HDBSCAN đã quan sát được chỉ để điền đủ danh mục lỗi chung. Tương tự, `duplicate` là thuộc tính của cặp HS001/HS004; lỗi ranh giới T01 cần được giải thích riêng.

## 6. Kết luận và giới hạn

Các ca đã đọc cho thấy ba vấn đề đáng chú ý đối với HDBSCAN:

1. **Ranh giới ngữ nghĩa của một số cụm còn rộng:** bài có chung thực thể hoặc cùng lĩnh vực vẫn có thể khác trọng tâm, như T01 và T05.
2. **Bài có nhiều trọng tâm cần được đánh giá theo ngữ cảnh:** HS016 là ca khó nhưng không đủ bằng chứng để gọi là gán sai.
3. **Nhãn `-1` có nhiều nguyên nhân:** bài hợp lệ chưa được gán cụm, tiểu chủ đề có thể hiếm, đầu vào thiếu ngữ cảnh hoặc bản ghi chứa nhiều tin độc lập.

Phạm vi của kết luận này là **10 ca được chọn để phân tích**, không phải tỷ lệ xuất hiện các lỗi trong toàn bộ dữ liệu. Sáu ca validation và bốn ca train dùng hai embedding khác nhau. Ngoài ra, độ nhạy với seed ở E04 được đo trên cùng dữ liệu, chưa thay thế đánh giá trên dữ liệu khác thời gian hoặc nguồn.

Phần HDBSCAN này là đầu vào cho E06 chung của nhóm. Khi so sánh với KMeans, NMF, LDA và BERTopic, nhóm cần giữ cùng định nghĩa từng nhóm lỗi và dẫn chứng bài cụ thể; không cộng số ca của các mẫu được chọn theo cách khác nhau thành một tỷ lệ lỗi chung.