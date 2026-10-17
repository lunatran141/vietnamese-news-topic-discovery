# Paper notes — Röder et al. (2015): Exploring the Space of Topic Coherence Measures

## 1. Người cung cấp

- Thành viên: TV3 — Tuấn.
- Phương pháp phụ trách: LDA, topic coherence và topic quality evaluation.
- Ngày đọc/review: 17/10/2026.

## 2. Citation info — Thông tin trích dẫn

- Citation key: `roder_2015_topic_coherence`.
- Tên đầy đủ của tài liệu: *Exploring the Space of Topic Coherence Measures*.
- Tác giả: Michael Röder, Andreas Both, Alexander Hinneburg.
- Năm xuất bản: 2015.
- Hội nghị: *Proceedings of the Eighth ACM International Conference on Web Search and Data Mining (WSDM '15)*, pages 399–408.
- DOI: https://doi.org/10.1145/2684822.2685324
- Bản PDF: https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf
- Loại tài liệu: Scientific paper — peer-reviewed conference paper.

## 3. Method — Phương pháp

- **Bài toán và mục tiêu:** Nghiên cứu các thước đo tự động đánh giá topic coherence, nhằm tìm những thước đo có mức độ tương quan cao với đánh giá của con người về khả năng diễn giải và mức độ dễ hiểu của topic.
- **Phương pháp:** Đề xuất một framework thống nhất cho topic coherence, cho phép biểu diễn các thước đo đã có và khám phá những tổ hợp mới.
- **Representation:** Mỗi topic được biểu diễn bằng một tập các từ khóa quan trọng nhất (top words). Coherence được tính dựa trên mối liên hệ thống kê giữa các từ trong tập này.
- **Cơ chế chính:** Framework gồm bốn thành phần: segmentation, probability estimation, confirmation measure và aggregation.
- **Segmentation:** Xác định các cặp hoặc tập con từ cần so sánh trong một topic.
- **Probability estimation:** Ước lượng xác suất từ và đồng xuất hiện từ dựa trên reference corpus, như Wikipedia hoặc corpus gốc khi có dữ liệu.
- **Confirmation measure:** Định lượng mức độ liên hệ giữa các từ hoặc nhóm từ, thông qua các phương pháp trực tiếp và gián tiếp.
- **Aggregation:** Tổng hợp các giá trị confirmation thành một điểm coherence cho topic.
- **Các thước đo được phân tích:** UMass, UCI, NPMI, C_V, C_P và những tổ hợp khác trong không gian cấu hình coherence.

Bài báo tập trung vào **đánh giá chất lượng topic representation**, không đề xuất thuật toán huấn luyện LDA hoặc NMF mới.

## 4. Dataset — Dữ liệu

- **Tên/nguồn:** Nghiên cứu sử dụng sáu benchmark: 20 Newsgroups (20NG), Genomics, New York Times (NYT), RTL-NYT, RTL-Wiki và Movie.
- **Ngôn ngữ và miền nội dung:** Các bộ dữ liệu tiếng Anh, bao gồm bài đăng diễn đàn, bài báo khoa học, tin tức và văn bản Wikipedia.
- **20NG:** 19.952 tài liệu, 100 topic được đánh giá, mỗi topic gồm 10 top words.
- **Genomics:** 29.833 tài liệu, 100 topic được đánh giá, mỗi topic gồm 10 top words; corpus chứa các bài viết khoa học từ 49 tạp chí MEDLINE.
- **NYT:** 100 topic được đánh giá, mỗi topic gồm 10 top words. Các topic được công bố dựa trên 47.229 bài New York Times, nhưng tác giả nghiên cứu không có quyền truy cập corpus NYT gốc, vì vậy Table 1 không báo cáo trực tiếp số tài liệu xử lý cho corpus này.
- **RTL-NYT:** Table 1 có 1.095 tập top words được đưa vào đánh giá, mỗi tập gồm 5 từ. Nghiên cứu nguồn ban đầu tạo 900 topic trước khi lấy các tập con từ để đánh giá.
- **RTL-Wiki:** Table 1 có 1.096 tập top words được đưa vào đánh giá, mỗi tập gồm 5 từ. Nghiên cứu nguồn ban đầu tạo 900 topic và thu thập đánh giá trên các tập từ.
- **Movie:** 108.952 tài liệu trong corpus được tái tạo, 100 topic được đánh giá, mỗi topic gồm 5 top words.
- **Reference corpus:** Wikipedia được sử dụng để ước lượng các xác suất từ; một số cấu hình còn được đánh giá bằng chính corpus gốc khi corpus đó có sẵn.
- **Preprocessing:** Các corpus và Wikipedia tham chiếu được xử lý bằng lemmatization, loại stopwords và một số bước lọc tài liệu.
- **Experimental protocol:** Tính các coherence measures trên các tập top words và so sánh thứ hạng của chúng với human topic ratings. Không sử dụng Train/Validation/Test protocol kiểu supervised classification.

Cần phân biệt số tập từ được đánh giá với số topic gốc do mô hình tạo ra. Các số 1.095 và 1.096 của RTL-NYT/RTL-Wiki không được mô tả là số topic độc lập ban đầu.

## 5. Metric — Đánh giá

- **Metric chính:** Pearson correlation coefficient (r), đo mức tương quan giữa điểm coherence tự động và đánh giá topic của con người.
- **Ý nghĩa:** Giá trị tương quan dương cao hơn cho thấy coherence measure có xu hướng phù hợp hơn với human ratings trong thiết lập đánh giá.
- **Các coherence measures:** C_V, C_P, C_UMass, C_one-any, C_UCI, C_NPMI và C_A.
- **Quy mô tìm kiếm:** Nghiên cứu khảo sát 237.912 cấu hình coherence và parameterizations.
- **Human evaluation:** Các benchmark cung cấp human ratings về khả năng hiểu và diễn giải topic, được tạo từ các nghiên cứu trước.
- **Kết quả chính:** C_V đạt hệ số tương quan trung bình cao nhất trong các thước đo được trình bày tại Table 2 khi sử dụng Wikipedia làm reference corpus.
- **Số liệu từ Table 2:** C_V đạt Pearson r trung bình 0,731; C_P đạt 0,700; C_NPMI đạt 0,653; C_UCI đạt 0,599; C_UMass đạt 0,358.
- **Điều kiện của số liệu:** Các giá trị trên là trung bình của các hệ số tương quan qua sáu benchmark, với hai nhóm có 10 và 5 top words. Đây không phải điểm topic coherence của LDA trên một corpus cụ thể.
- **Reference corpus:** Trong nhiều bộ đánh giá, các coherence measures dùng thống kê xác suất từ Wikipedia đạt tương quan tốt hơn khi dùng corpus gốc, nhưng không phải trong mọi trường hợp.
- **Runtime:** Section 6 và Table 6 nghiên cứu chi phí tính toán. Trên 100 topic NYT với Wikipedia làm reference corpus, thời gian được báo cáo cho C_V là 315,2 giây và C_UMass là 13,7 giây trong thiết lập thực nghiệm của tác giả.
- **Ý nghĩa kết quả:** Coherence tự động có giá trị hỗ trợ đánh giá topic, nhưng không thể xem là sự thay thế tuyệt đối cho human evaluation.

Các runtime và giá trị correlation của nghiên cứu gốc không được sử dụng để dự đoán trực tiếp kết quả trên Dataset v1.0 tiếng Việt của nhóm.

## 6. Gap — Hạn chế hoặc khoảng trống

- **Phụ thuộc reference corpus:** Coherence dựa trên thống kê xác suất và đồng xuất hiện từ nên phụ thuộc dữ liệu tham chiếu, preprocessing, lựa chọn top words và cách ước lượng xác suất.
- **Không đánh giá toàn bộ đặc tính của mô hình:** Coherence chỉ tập trung vào mức độ liên hệ của các từ đại diện cho topic, không trực tiếp đánh giá độ chính xác gán tài liệu vào cụm hoặc chất lượng toàn bộ hệ thống topic discovery.
- **Không đo stability:** Bài báo không được thiết kế để kiểm tra mức độ nhất quán của topic giữa các lần huấn luyện với random seeds khác nhau.
- **Không trực tiếp đo topic diversity:** Coherence và diversity phản ánh những khía cạnh khác nhau; kết quả nghiên cứu không thay thế một phép đo riêng về mức độ trùng lặp từ giữa các topic.
- **Giới hạn ngôn ngữ:** Các benchmark được nghiên cứu không phải corpus tin tức tiếng Việt đã phân đoạn từ. Việc lựa chọn reference corpus và tokenization có thể ảnh hưởng đến metric khi áp dụng cho tiếng Việt.
- **Chi phí tính toán:** Một số phương pháp coherence cần thống kê đồng xuất hiện trên sliding windows hoặc ngữ liệu tham chiếu lớn nên có thể tốn thời gian hơn các phương pháp dựa trên document-level counts.

**Phân biệt nguồn nhận định:** Sự phụ thuộc vào cách ước lượng xác suất, so sánh reference corpora và runtime được khảo sát trong paper. Các nhận xét về tiếng Việt, diversity, document clustering và multi-seed stability là đánh giá về phạm vi áp dụng đối với đồ án của TV3.

## 7. Liên hệ với đồ án

- **M08 — LDA Experiments:** Cung cấp cơ sở học thuật để sử dụng topic coherence khi so sánh các cấu hình số topic K.
- **M08 — Topic Quality:** Giúp giải thích vì sao coherence cần được xem xét cùng topic diversity, topic keywords và representative documents.
- **M09 — LDA Stability:** Coherence phản ánh một chiều chất lượng topic, nhưng không thay thế các chỉ số stability tính qua nhiều random seeds.
- **3.7 — LDA/NMF Comparison:** Khi so sánh hai phương pháp, cần nhất quán về cách tính coherence, tập văn bản tham chiếu, preprocessing và số top words.
- **3.6 — Cross-method Evaluation:** Hỗ trợ yêu cầu thống nhất evaluation protocol giữa các phương pháp topic discovery.
- **3.10 — Human Evaluation:** Cung cấp cơ sở khoa học để đối chiếu đánh giá tự động với đánh giá bằng người.
- **R03 — Paper Writing:** Được sử dụng để giải thích lựa chọn và giới hạn của metric coherence trong Methodology, Experimental Setup và Discussion.

Trong đồ án, các giá trị coherence, diversity và stability phải được tính từ artifacts thực nghiệm của nhóm. Không được coi hệ số tương quan Pearson trong bài báo này là điểm coherence của mô hình LDA trên Dataset v1.0.

## 8. Vị trí bằng chứng

| Nhận định/số liệu | Section/trang/bảng/vị trí nguồn |
|---|---|
| Mục tiêu đánh giá topic coherence | Abstract; Section 1 |
| Các coherence measures đã công bố | Section 2 — Related Work |
| Framework gồm bốn thành phần | Section 3; Figure 1 |
| Segmentation | Section 3.1 |
| Probability estimation | Section 3.2 |
| Confirmation measures | Section 3.3 |
| Aggregation | Section 3.4 |
| Sáu benchmark, số topic và số top words | Section 4; Table 1 |
| Số tài liệu của 20NG, Genomics và Movie | Section 4; Table 1 |
| NYT và giới hạn truy cập corpus gốc | Section 4 — Evaluation and Data Sets |
| Nguồn RTL-NYT/RTL-Wiki và human ratings | Section 4 |
| Human ratings và Pearson correlation | Section 4 |
| Khảo sát 237.912 cấu hình coherence | Section 4 |
| Kết quả C_V, C_P, C_NPMI, C_UCI và C_UMass | Section 5; Table 2 |
| So sánh Wikipedia và corpus gốc | Section 5; Table 2 |
| Runtime của các coherence measures | Section 6 — Runtimes; Table 6 |
| Các ứng dụng của coherence evaluation | Section 7 — Applications |
| Kết luận nghiên cứu | Section 8 — Conclusions |

Nguồn chính: https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf

## 9. Điều cần xác minh thêm

- Kiểm tra metric coherence thực tế được sử dụng trong pipeline M08, gồm `c_v`, `u_mass` và các cấu hình liên quan.
- Xác nhận reference corpus, preprocessing variant và số top words được dùng khi tính coherence cho LDA.
- Khi đối chiếu LDA với NMF và các phương pháp khác, kiểm tra khả năng so sánh của metric giữa các representation và topic outputs.
- Không dùng bài báo này làm nguồn chứng minh trực tiếp cho kết quả multi-seed stability hoặc topic diversity.
- Khi trình bày runtime, chỉ so sánh số liệu từ những môi trường thực nghiệm tương đương.

Các thông tin dataset, top words, Pearson correlation, Table 2 và runtime được đối chiếu với PDF gốc của nghiên cứu.

## 10. Trạng thái và bàn giao

- Trạng thái: `Ready for TV3 review`.
- Link file hoặc PR bàn giao: Chưa tạo; bổ sung sau khi push branch.
- Nhận xét của TV3 sau review: Đã đối chiếu framework, dataset, human ratings, correlation và runtime với bài báo gốc. Chưa nghiệm thu vào Related Work Matrix.