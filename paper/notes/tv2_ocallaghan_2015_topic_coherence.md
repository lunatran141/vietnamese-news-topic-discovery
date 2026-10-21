# Paper notes - O’Callaghan et al. (2015): An analysis of the coherence of descriptors in topic modeling

## 1. Người cung cấp

- Thành viên: TV2 - Tiến.
- Phương pháp phụ trách: NMF/topic modeling và đánh giá topic.
- Ngày đọc/đối chiếu nguồn: 21/10/2026.

## 2. Citation info - Thông tin trích dẫn

- Citation key: `ocallaghan_2015_topic_coherence`.
- Tên đầy đủ: *An analysis of the coherence of descriptors in topic modeling*.
- Tác giả: Derek O’Callaghan, Derek Greene, Joe Carthy và Pádraig Cunningham.
- Năm xuất bản: 2015.
- Tạp chí: *Expert Systems with Applications*, volume 42, issue 13, trang 5645-5657.
- **DOI chính thức:** <https://doi.org/10.1016/j.eswa.2015.02.055>.
- Trang nhà xuất bản: <https://www.sciencedirect.com/science/article/pii/S0957417415001633>.
- PDF bản tác giả dùng đối chiếu: <https://researchrepository.ucd.ie/rest/bitstreams/18055/retrieve>.
- Loại tài liệu: Scientific paper - journal article.

**Quy ước bằng chứng:** Section, Eq., Table và Figure dưới đây được đối chiếu trên **bản tác giả 16 trang**, đề ngày 12/02/2015. “Trang” trong note này là số trang 1-16 của bản đó, không phải trang tạp chí 5645-5657. Không cộng một độ lệch để tự suy ra số trang bản xuất bản.

## 3. Method - Phương pháp

- **Bài toán và mục tiêu:** So sánh khả năng diễn giải của descriptor gồm các top terms từ NMF/LDA, đồng thời khảo sát generality, redundancy và mức phù hợp của document-topic membership với nhãn corpus. Trọng tâm là descriptor mà người dùng nhìn thấy, không chỉ độ khớp mô hình (Sections 1, 4-5).
- **Representation:** Corpus tiếng Anh được biểu diễn Bag-of-Words. Theo ký hiệu bài, ma trận term-document V ≈ WH có W là term-topic basis và H là topic-document weights; LDA có topic-word distribution φ và document-topic distribution θ (Section 2.1, trang 2-3).
- **NMF thực nghiệm:** Dùng mục tiêu squared error: tối thiểu hóa ‖V − WH‖² theo chuẩn Frobenius, với W và H không âm (Eq. (3)), triển khai scikit-learn theo phương pháp Lin (2007), alternating nonnegative least squares/projected gradient; khởi tạo xác định NNDSVD. Bài không đưa KL-NMF vào đánh giá này (Sections 2.1, 4.1, trang 3, 5).
- **LDA thực nghiệm:** MALLET với Gibbs sampling; α = 50/k, β = 0,01, đồng thời bật hyperparameter optimization cho asymmetric prior trên θ. Không nhầm phần related work về gensim/online variational Bayes với implementation dùng cho kết quả chính. Section 4.1 nói tăng số vòng so với code tham khảo nhưng không ghi con số cụ thể tại đó (trang 5).

**Bốn cách tạo descriptor**, mỗi topic lấy **10 terms** (Section 4.1, trang 5-6):

| Cấu hình | Dữ liệu/huấn luyện | Cách chọn top terms | Ý nghĩa của weighting |
|---|---|---|---|
| NMFw | Log-based TF-IDF, chuẩn hóa vector về độ dài 1 | 10 trọng số cao nhất trong topic basis Wₖ | IDF giảm ảnh hưởng từ phổ biến, tăng vai trò từ phân biệt |
| NMFu | TF, bỏ thành phần IDF | 10 trọng số cao nhất trong basis của mô hình này | Là mô hình được huấn luyện riêng, không chỉ đổi descriptor của NMFw |
| LDAu | Bag-of-Words/feature sequences trong MALLET | 10 từ có xác suất φ(k, v) lớn nhất | Descriptor theo xác suất gốc; “unweighted” không có nghĩa bỏ term counts |
| LDAw | Dùng lại mô hình LDAu | 10 từ có term-score lớn nhất theo Eq. (5) | Chỉ hậu xử lý thứ hạng từ, không huấn luyện lại; giữ nguyên θ |

Eq. (5), Section 4.1, trang 5:

> term-score(k, v) = β̂(k, v) × log[β̂(k, v) / G(v)].
>
> G(v) là trung bình nhân của xác suất từ v trên K topic: G(v) = [β̂(1, v) × β̂(2, v) × … × β̂(K, v)] mũ (1/K).

Ở đây β̂(k, v) là xác suất của từ v trong topic k, không phải hyperparameter β = 0,01. Cách xếp hạng này giảm vai trò từ có xác suất cao trên nhiều topic. Vì NMFw thay đổi **đầu vào huấn luyện**, còn LDAw thay đổi **đầu ra descriptor**, không được quy mọi chênh lệch coherence cho riêng thuật toán NMF/LDA.

## 4. Dataset - Dữ liệu

**Sáu corpus tiếng Anh sau preprocessing**, chép số liệu từ **Table 1(a), trang 5**. Cột “Terms” là kích thước từ vựng, không phải tổng token.

| Corpus | Documents | Terms | Classes | Nguồn/protocol xây dựng (Section 3, trang 4-5) |
|---|---:|---:|---:|---|
| BBC | 161.469 | 17.079 | 40 | Bài BBC trong giai đoạn 01/01/2009-31/12/2013, giữ 40 section lớn nhất; một section/bài |
| Guardian | 194.153 | 22.141 | 24 | Thu thập từ website Guardian theo quy trình tương tự; giữ 24 section lớn nhất, một section/bài |
| NYT 2003 | 70.134 | 20.429 | 20 | Bài năm 2003 từ New York Times Corpus |
| NYT 2000+ (10%) | 65.335 | 21.461 | 20 | Mẫu phân tầng 10% bài giai đoạn 2000-2007 từ cùng nguồn NYT |
| Wikipedia (high-level) | 5.682 | 28.699 | 6 | Chọn các category cấp cao từ WikiProject; nội dung từ dump tháng 01/2014 |
| Wikipedia (lower-level) | 4.970 | 24.265 | 10 | Chọn các sub-category cấp thấp hơn từ nguồn WikiProject/dump tương ứng |

- **Nguồn nhãn:** Section metadata của các báo và category của Wikipedia, không phải nhãn topic do mô hình tạo ra. NYT có thể có nhiều section/bài; loại các meta-section như Front Page, Corrections, Magazine. Wikipedia cũng có bài thuộc nhiều category. Tập ứng viên Wikipedia ban đầu gồm 42.170 page titles/194 categories; không nhầm với quy mô hai tập cuối ở Table 1 (Section 3, trang 4-5).
- **Tiền xử lý chung:** Bỏ 671 English stopwords, URI, từ chứa chữ số/tên miền; lowercase, chuẩn hóa ASCII và lemmatization WordNet qua NLTK. Không stemming vì cần descriptor dễ đọc. Bỏ từ xuất hiện trong ít hơn ℓ tài liệu, với ℓ = max(10, m/1000) và m là số tài liệu (Section 4.1, trang 5).
- **Protocol mô hình:** Chạy riêng trên mỗi corpus với k = 10, 20, …, 100; tính trung bình coherence của các topic cho từng cặp (descriptor method, k). Không được báo cáo chia Train/Validation/Test cho đánh giá coherence này (Section 5.1, trang 7).
- **Protocol membership:** Loại tài liệu đa nhãn, gán mỗi tài liệu vào topic có trọng số cao nhất rồi so với nhãn bằng AMI. Quy mô cụ thể sau bước loại đa nhãn không được báo cáo trong Section 5.3; không coi số Documents ở Table 1 là chắc chắn số mẫu AMI (trang 10-11).
- **Corpus tham chiếu metric:** Wikipedia dump tháng 01/2014 được tokenize, bỏ cùng 671 stopwords và lemmatize. Đây là nguồn tính metric, cần phân biệt với hai tập Wikipedia nhỏ để huấn luyện/đánh giá mô hình. Co-occurrence dùng cửa sổ 20 terms. Word2vec Skip-gram có 200 chiều, context window tối đa 5, hierarchical softmax bật, negative sampling tắt, sample threshold 0,001 (Section 5.1, trang 7).

## 5. Metric - Đánh giá

### 5.1. Ba coherence measures thực sự được báo cáo

Mỗi descriptor gồm N từ theo thứ hạng, với N = 10. Ký hiệu wᵢ là từ ở vị trí i và C = N × (N − 1) / 2 là số cặp từ. Điểm topic là trung bình trên các cặp; điểm mô hình là trung bình điểm topic (Section 4.2, trang 6).

| Tên | Định nghĩa | Chiều diễn giải | Vị trí |
|---|---|---|---|
| **TC-NPMI** | Normalized pointwise mutual information; đồng xuất hiện so với xác suất biên của cả hai từ | Cao hơn là coherence cao hơn theo metric này | Eq. (6), trang 6 |
| **TC-LCP** | Mean pairwise log conditional probability, điều kiện trên từ xếp hạng cao hơn | Cao hơn, tức thường ít âm hơn, là coherence cao hơn theo metric này | Eq. (7), trang 6 |
| **TC-W2V** | Trung bình cosine similarity của vector từ Skip-gram | Cao hơn là gần nhau hơn trong không gian ngữ nghĩa tham chiếu | Eq. (8), trang 6 |

Công thức theo bản tác giả:

> TC-NPMI = (1/C) × Σ Σ [log((P(wⱼ, wᵢ) + ε) / (P(wᵢ) × P(wⱼ))) / (−log P(wᵢ, wⱼ) + ε)].

> TC-LCP = (1/C) × Σ Σ log[(P(wⱼ, wᵢ) + ε) / P(wᵢ)].

> TC-W2V = (1/C) × Σ Σ cosine(wvⱼ, wvᵢ).
>
> Trong cả ba công thức, lấy tổng qua j từ 2 đến N và i từ 1 đến j − 1. Vector wvᵢ và wvⱼ lần lượt biểu diễn hai từ; cosine là tích vô hướng của hai vector chia cho tích độ dài của chúng.

**Lưu ý ký hiệu/smoothing:** Eq. (6)-(7) ở trên giữ cách đặt ε của bản tác giả; Eq. (8) dùng cosine theo phần mô tả. Tác giả gọi ε = 1 là **smoothing count** để tránh log của 0. Không diễn giải thành cộng xác suất 1 vào mọi joint probability trong một implementation mới; khi tái lập phải xác định rõ cách làm trơn counts rồi chuẩn hóa xác suất. Khi không làm trơn, NPMI chuẩn là PMI / [−log P(wᵢ, wⱼ)]. Không tự thay công thức có smoothing của bài bằng một biến thể trong thư viện mà không giải thích.

TC-LCP có thể ưu ái các cặp từ chung hơn vì chỉ xét xác suất biên của từ đứng trước trong thứ hạng. Table 2, trang 7 minh họa: (school, year) đạt TC-LCP -0,85 so với -1,01 của (school, education), nhưng TC-NPMI lần lượt 0,12 so với 0,38 và TC-W2V 0,20 so với 0,54. Đây là **điểm hai cặp từ**, không phải điểm mô hình hay trung bình corpus.

### 5.2. Generality, redundancy và document-topic membership

- **Overlap/dependency giữa descriptor:** Mean pairwise Jaccard (MPJ), Eq. (9), Section 4.2, trang 6:

> MPJ(m, k) = (1 / số cặp topic) × Σ Σ Jaccard(TDᵢ, TDⱼ).
>
> Số cặp topic bằng k × (k − 1) / 2. Lấy tổng qua j từ 2 đến k và i từ 1 đến j − 1. Jaccard là số từ chung của hai tập chia cho số từ trong hợp của chúng.

TDᵢ là tập top-10 terms của topic i. Giá trị cao hơn biểu thị trùng lặp/phụ thuộc nhiều hơn, tức bất lợi nếu muốn các descriptor ít dư thừa. Đây không phải coherence score và không tự động đánh giá toàn bộ chất lượng mô hình.

- **Tần suất lặp từ/generality:** Tạo phân phối số lần từ xuất hiện qua các descriptor, phân tích số từ nằm trong 2, 3, 4 hoặc ít nhất 5 descriptor tại k = 100; đọc các từ phổ biến để nhận diện tính chung. Không có một điểm “generality” duy nhất thay cho các phân tích này (Section 4.2, trang 6; Section 5.2, trang 8-10; Figures 5-6, Table 3).
- **Membership:** Adjusted Mutual Information (**AMI**, không phải ARI/NMI), đo phù hợp giữa hard assignment topic và class labels có hiệu chỉnh kỳ vọng do ngẫu nhiên; cao hơn thể hiện phù hợp hơn. Bài không đưa công thức AMI, nên không gán thêm số phương trình cho metric này. LDAw giữ nguyên document-topic distributions của LDAu nên không có đường AMI riêng (Section 5.3, trang 10-11; Figure 7, trang 13).
- **Đánh giá định tính:** Đọc từ khóa, ví dụ Syria/Middle East trong Introduction và các từ lặp ở Table 3 hỗ trợ diễn giải. Bài chưa thực hiện user survey để xác nhận trực tiếp tương quan với human judgements; đây là future work (Section 6, trang 14).

### 5.3. Kết quả và điều kiện so sánh

| Khía cạnh | Phát hiện trong thiết lập bài gốc | Bằng chứng |
|---|---|---|
| TC-NPMI | Các phương pháp có weighting mạnh hơn; **NMFw thường** đạt coherence cao, LDAw cũng mạnh; không kết luận mọi cấu hình NMF đều thắng | Section 5.1, trang 7; Figure 2(a-f), trang 8 |
| TC-W2V | Xu hướng có lợi cho NMFw/weighted descriptors tương tự TC-NPMI | Section 5.1, trang 7; Figure 4(a-f), trang 10 |
| TC-LCP | **LDAu có lợi thế**, NMF kém hơn theo metric này; LDAw thường ở giữa | Section 5.1, trang 7-8; Figure 3(a-f), trang 9 |
| Redundancy/generality | LDAu thường lặp từ và descriptor nhiều hơn; LDAw giảm hiện tượng đó, NMFw nhìn chung ít hơn; phụ thuộc k và corpus | Section 5.2, trang 8-10; Figure 5, trang 11; Figure 6, trang 12 |
| Document-topic membership | Với các corpus ngoài Wikipedia, LDAu có AMI nhỉnh hơn NMFw, NMFu kém; hai tập Wikipedia có mức phù hợp cao hơn và ít chênh lệch giữa phương pháp | Section 5.3, trang 10-11; Figure 7(a-f), trang 13 |

**Điều kiện của các kết luận:** Sáu corpus trong Table 1; k từ 10 đến 100, bước 10; descriptor top-10; NMF squared error + NNDSVD; LDA MALLET; các cách weighting ở mục 3; metric dùng Wikipedia tham chiếu và cấu hình ở mục 4. Các đường coherence là trung bình topic trong mô hình, không phải accuracy hay kết quả một user study. Bài chủ yếu trình bày đường đồ thị; không tự đọc thành các điểm số chính xác hoặc tuyên bố ý nghĩa thống kê khi không có bảng/kiểm định tương ứng.

**Kết luận cân bằng:** Hai phép đo thường có lợi cho NMF là **TC-NPMI và TC-W2V**, đặc biệt với NMFw; **TC-LCP và AMI** cho kết quả có lợi cho LDA trong các điều kiện trên. LDA còn có thể phù hợp khi cần mô tả topic rộng, khái quát (Section 6, trang 14). Không kết luận NMF luôn tốt hơn LDA.

## 6. Gap - Hạn chế hoặc khoảng trống

**Phạm vi/hạn chế do tác giả nêu:**

- Chỉ xét các biến thể phổ biến; không đánh giá toàn bộ họ NMF/LDA. Loại KL-NMF khỏi thực nghiệm (Sections 2.1, 4.1, trang 3, 5).
- Chỉ đánh giá unigram, dù n-grams có thể cải thiện coherence; cần đọc kỹ topic được sinh ra (Section 6, trang 14).
- Điểm phụ thuộc smoothing và cách tổng hợp: ε = 10⁻¹² tạo outliers với cặp từ không đồng xuất hiện; trung bình nhạy với outliers. Tác giả chọn ε = 1 như một thỏa hiệp và thảo luận khả năng dùng median (Section 5.4, trang 12-13).
- Cần khảo sát coherence measures khác và thực hiện user survey so sánh descriptor với human judgements; chưa thể coi nghiên cứu đã xác thực đánh giá con người trực tiếp (Section 6, trang 14).
- Membership chỉ phân tích gán đơn topic, loại tài liệu đa nhãn dù hai mô hình cho phép mixture; đây là giới hạn protocol được nêu rõ (Section 5.3, trang 10).

**Nhận xét riêng của TV2 đối với đồ án tin tức tiếng Việt:**

- Sáu corpus và Wikipedia tham chiếu đều là tiếng Anh; chưa kiểm tra phân đoạn từ tiếng Việt, từ ghép sau tokenization hay chất lượng embedding/co-occurrence tham chiếu tiếng Việt.
- Coherence, độ trùng lặp descriptor và phù hợp với chuyên mục báo phản ánh các khía cạnh khác nhau. Chuyên mục không mặc nhiên trùng với sự kiện hoặc topic mong muốn.
- Cần đối chiếu nhiều seed, lựa chọn k, từ khóa và bài đại diện theo protocol của nhóm; bài không cung cấp kết quả stability hay khám phá sự kiện tiếng Việt.
- Đây là nhận xét về **khả năng áp dụng của bài này**, không phải tuyên bố các vấn đề đó chưa từng được nghiên cứu.

## 7. Liên hệ với đồ án

- **M04/M05 - NMF và đánh giá:** Cơ sở khảo sát TF-IDF, số topic, top terms và kết hợp coherence với đọc topic/bài đại diện.
- **So sánh LDA/NMF:** Ghi rõ khác biệt đầu vào và hậu xử lý descriptor; giữ nhất quán số top words, metric, tokenization, reference corpus và cấu hình khi so sánh.
- **Đánh giá định tính/diversity:** Có thể kế thừa cách xem từ lặp, Jaccard overlap và nội dung descriptor. Nếu dùng topic diversity khác, phải định nghĩa riêng; không gọi đó là cùng metric với MPJ.
- **Tin tức tiếng Việt sau preprocessing P3:** Cần tự tính kết quả; không chuyển các đường TC-W2V tiếng Anh hoặc AMI với class labels của bài thành kết quả Dataset V1. Nếu nhóm không có nhãn thích hợp, không mặc nhiên bổ sung AMI.

Đây là thiết kế thí nghiệm dự kiến, không phải kết quả nhóm đã xác nhận.

## 8. Vị trí bằng chứng

| Nhận định/số liệu | Section/trang/bảng/vị trí nguồn trong bản tác giả |
|---|---|
| Citation và DOI chính thức | Trang nhà xuất bản; DOI 10.1016/j.eswa.2015.02.055 |
| Mục tiêu coherence/generality; ví dụ descriptor Syria | Abstract, trang 1; Section 1, trang 1-2 |
| Cơ chế LDA; NMF term-document và loss | Section 2.1, trang 2-3; Eqs. (1)-(4) |
| Nguồn BBC/Guardian/NYT/Wikipedia và cách lấy mẫu | Section 3, trang 4-5 |
| Quy mô, từ vựng, số lớp sáu corpus | Table 1(a), trang 5; thống kê độ dài ở Table 1(b) |
| Preprocessing, lọc từ, MALLET, hyperparameters, NMF/NNDSVD | Section 4.1, trang 5 |
| Bốn descriptor; khác biệt tiền/hậu xử lý weighting | Section 4.1, trang 5-6; Eq. (5), trang 5 |
| Tên/định nghĩa/công thức ba coherence measures | Section 4.2, trang 6; Eqs. (6)-(8) |
| Trung bình cặp từ/topic; smoothing count | Section 4.2, trang 6, đoạn sau Eq. (8) |
| Jaccard và phân phối từ lặp | Section 4.2, trang 6; Eq. (9) |
| k, top-10, Wikipedia tham chiếu, window và word2vec | Section 5.1, trang 7 |
| Hai cặp school/year và school/education | Section 5.1; Table 2, trang 7 |
| NMF thường có lợi ở TC-NPMI/TC-W2V, LDA ở TC-LCP | Section 5.1, trang 7-8; Figures 2-4, trang 8-10 |
| Generality, redundancy; từ phổ biến trong BBC với k = 100 | Section 5.2, trang 8-10; Table 3, trang 7; Figures 5-6, trang 11-12 |
| Hard assignment, bỏ đa nhãn, AMI; LDAw chung membership | Section 5.3, trang 10-11; Figure 7, trang 13 |
| Smoothing/outliers và mean/median | Section 5.4, trang 12-13 |
| Phạm vi unigram, mô tả rộng của LDA, future user survey | Section 6, trang 13-14, đặc biệt trang 14 |

## 9. Điều cần xác minh thêm

- Khi triển khai lại metric, kiểm tra mã nguồn được dẫn tại <http://mlg.ucd.ie/topiccoherence/> để xác định quy trình counts/probabilities và smoothing, thay vì suy ra trực tiếp từ ký hiệu P + ε.
- Xác minh corpus tham chiếu tiếng Việt, OOV trong embedding, số top words và metric thực tế trong pipeline nhóm; không mặc định TC-W2V tương đương C_V.
- Số vòng LDA cụ thể và số tài liệu sau loại đa nhãn không được báo cáo tại phần protocol tương ứng; nếu cần tái lập chính xác phải tra code/artifacts gốc. Chưa kiểm tra mã nguồn đó trong lần đối chiếu này.
- Không bổ sung điểm số đọc ước lượng từ đồ thị như số liệu chính xác; cần dữ liệu kết quả gốc nếu muốn lập bảng số.

Các mục method, sáu corpus, metric, chiều diễn giải, phát hiện chính và vị trí bằng chứng đã đối chiếu toàn văn/hình của bản tác giả.

## 10. Trạng thái và bàn giao

- Trạng thái: `Ready for TV3 review`.
- Link file hoặc PR bàn giao: `paper/notes/tv2_ocallaghan_2015_topic_coherence.md`.
- Nhận xét của TV3 sau review: Chưa có; chờ TV3 review và nghiệm thu vào Related Work Matrix.
