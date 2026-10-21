# Khám phá chủ đề bằng BERTopic

## Phương pháp

### Tổng quan

BERTopic là phương pháp mô hình hóa chủ đề dựa trên biểu diễn ngữ nghĩa, kết
hợp bốn thành phần chính: sentence embeddings, giảm chiều UMAP, phân cụm mật
độ HDBSCAN và biểu diễn từ khóa bằng class-based TF-IDF (c-TF-IDF). Khác với
các mô hình xác suất như LDA, BERTopic không giả định mỗi tài liệu được sinh
ra từ một phân phối xác suất trên các chủ đề. Thay vào đó, tài liệu được ánh xạ
vào không gian vector ngữ nghĩa, sau đó các vùng có mật độ cao được xác định
như những cụm chủ đề. Các tài liệu không thuộc vùng mật độ đủ ổn định có thể
được giữ lại dưới dạng ngoại lai thay vì bị buộc vào một chủ đề không phù hợp
[Grootendorst, 2022].

Pipeline được sử dụng trong nghiên cứu gồm năm bước:

\[
\text{Văn bản}
\rightarrow \text{Sentence Embedding}
\rightarrow \text{UMAP}
\rightarrow \text{HDBSCAN}
\rightarrow \text{c-TF-IDF}.
\]

Cách tổ chức mô-đun này cho phép thay đổi mô hình embedding và tham số phân
cụm mà không phải thay đổi toàn bộ pipeline. Đây là đặc điểm phù hợp với mục
tiêu khảo sát ảnh hưởng của biểu diễn ngôn ngữ chuyên biệt tiếng Việt, cấu
trúc lân cận của UMAP và kích thước cụm tối thiểu của HDBSCAN.

### Dữ liệu đầu vào

Các thí nghiệm sử dụng Dataset v1.0 sau bước chuẩn hóa P1. Mỗi bài báo được
định danh bằng `article_id` và liên kết với một tập chia cố định. Tập train gồm
6.493 bài được sử dụng để huấn luyện mô hình, trong khi tập validation gồm
1.391 bài được dùng để lựa chọn cấu hình. Tập test được giữ khóa trong giai
đoạn điều chỉnh tham số nhằm hạn chế rò rỉ dữ liệu.

Đầu vào chính là chuỗi được tạo bằng cách ghép tiêu đề với nội dung bài báo đã
chuẩn hóa. Văn bản không bị loại stopword trước khi tạo embedding nhằm bảo
toàn ngữ cảnh câu. Danh sách stopword chỉ được áp dụng tại bước c-TF-IDF để
giảm các từ ít thông tin trong phần mô tả chủ đề. Ngoài cấu hình đầu vào đầy
đủ, một thí nghiệm ablation chỉ sử dụng tiêu đề đã được thực hiện để đánh giá
vai trò của phần nội dung.

### Biểu diễn ngữ nghĩa

Với mỗi tài liệu \(d_i\), mô hình sentence-transformer tạo một vector ngữ
nghĩa:

\[
\mathbf{e}_i=f_{\theta}(d_i),
\]

trong đó \(f_{\theta}\) là mô hình embedding đã được huấn luyện trước. Nghiên
cứu khảo sát hai mô hình: `paraphrase-multilingual-MiniLM-L12-v2`, được sử
dụng làm baseline đa ngôn ngữ, và `keepitreal/vietnamese-sbert`, được huấn
luyện chuyên biệt cho tiếng Việt. Các vector được chuẩn hóa trước khi đưa vào
bước giảm chiều.

### Giảm chiều và phân cụm mật độ

Embedding ban đầu có số chiều lớn nên được chiếu xuống không gian 5 chiều bằng
UMAP. Cấu hình cơ sở sử dụng `n_neighbors=15`, `n_components=5`,
`min_dist=0.0`, khoảng cách cosine và seed 42. Tham số `n_neighbors` kiểm soát
mức độ cân bằng giữa cấu trúc cục bộ và cấu trúc rộng hơn của không gian dữ
liệu. Một cấu hình bổ sung tăng giá trị này lên 30 để khảo sát ảnh hưởng của
vùng lân cận lớn hơn.

Sau UMAP, HDBSCAN được sử dụng để xác định các vùng có mật độ cao mà không cần
quy định trước số lượng topic. Cấu hình cơ sở sử dụng
`min_cluster_size=15`, `min_samples=5`, khoảng cách Euclidean và phương pháp
chọn cụm Excess of Mass. Một thí nghiệm khác tăng `min_cluster_size` lên 30
nhằm đánh giá tác động của việc ưu tiên các cụm lớn hơn. Những tài liệu không
thuộc cụm ổn định được gán nhãn \(-1\). Trong nghiên cứu này, các tài liệu đó
được xem là ứng viên ngoại lai; chúng có thể là nhiễu, bài hiếm, bài đa chủ đề
hoặc nội dung mới chưa hình thành cụm rõ ràng.

### Biểu diễn từ khóa bằng c-TF-IDF

Sau khi phân cụm, toàn bộ tài liệu thuộc cùng một topic được ghép thành một
siêu tài liệu. Trọng số c-TF-IDF của từ \(t\) trong topic \(c\) được mô tả
khái quát như sau:

\[
w_{t,c}=tf_{t,c}\log\left(1+\frac{\bar{A}}{tf_t}\right),
\]

trong đó \(tf_{t,c}\) là tần suất của từ \(t\) trong siêu tài liệu của topic
\(c\), \(tf_t\) là tần suất của từ đó trên toàn bộ các topic và \(\bar{A}\)
là độ dài trung bình của các siêu tài liệu. Vectorizer sử dụng unigram và
bigram với `min_df=2`, `max_df=0.95`. Mười từ có trọng số cao nhất được dùng
để diễn giải mỗi topic.

Đối với từng topic, hệ thống đồng thời lưu tối đa năm bài báo đại diện. Tên
topic gợi ý được hình thành từ các từ khóa có trọng số cao, sau đó được đối
chiếu với tiêu đề và nội dung của các bài đại diện. Việc đổi tên chỉ tác động
đến lớp trình bày, không thay đổi assignment hay output gốc của mô hình.

## Thiết lập thực nghiệm

Năm cấu hình BERTopic được xây dựng để khảo sát riêng tác động của embedding,
UMAP, HDBSCAN và loại văn bản đầu vào. Các thí nghiệm sử dụng cùng dataset,
split và seed 42; mỗi thí nghiệm chỉ thay đổi thành phần được quan tâm.

| Run | Mục tiêu | Embedding | `n_neighbors` | `min_cluster_size` | Đầu vào |
|---|---|---|---:|---:|---|
| E0 | Baseline đa ngôn ngữ | Multilingual MiniLM | 15 | 15 | Tiêu đề + nội dung |
| E1 | Embedding tiếng Việt | Vietnamese SBERT | 15 | 15 | Tiêu đề + nội dung |
| E2 | Khảo sát vùng lân cận rộng hơn | Vietnamese SBERT | 30 | 15 | Tiêu đề + nội dung |
| E3 | Khảo sát cụm tối thiểu lớn hơn | Vietnamese SBERT | 15 | 30 | Tiêu đề + nội dung |
| E4 | Ablation loại văn bản đầu vào | Vietnamese SBERT | 15 | 15 | Chỉ tiêu đề |

Các mô hình được đánh giá bằng số lượng topic, tỷ lệ ngoại lai, độ phủ, topic
diversity, NPMI coherence, Adjusted Rand Index (ARI) và Normalized Mutual
Information (NMI). Độ phủ được xác định bởi:

\[
\text{Coverage}=1-\text{Outlier Rate}.
\]

Topic diversity đo tỷ lệ từ khóa không trùng lặp trong tập các từ khóa đứng
đầu, trong khi NPMI phản ánh mức đồng xuất hiện của từ khóa trên tập
validation. ARI và NMI được tính theo `category_gold`. Do `category_gold` là
chuyên mục báo chí rộng chứ không phải nhãn chủ đề chi tiết, hai chỉ số này
chỉ được xem là tín hiệu tham chiếu bên ngoài, không được diễn giải như độ
chính xác phân loại.

Runtime được đo cho toàn bộ pipeline, bao gồm tải embedding model, tạo biểu
diễn, khởi tạo mô hình, huấn luyện và dự đoán trên validation. Vì vậy, runtime
không đồng nhất với thời gian fit của BERTopic.

## Kết quả và thảo luận

### So sánh các cấu hình

Bảng dưới trình bày kết quả của năm cấu hình trên tập validation. Các giá trị
test không được sử dụng trong quá trình lựa chọn mô hình.

| Run | Số topic | Outlier train | Outlier validation | Coverage | Diversity | ARI | NMI | NPMI | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | 111 | 31,14% | 42,27% | 57,73% | 0,7000 | 0,0205 | 0,3827 | -0,0681 | 186,02 |
| E1 | 96 | 24,61% | **35,87%** | **64,13%** | 0,6813 | 0,0720 | 0,4213 | -0,0668 | 1.086,07 |
| E2 | 93 | 33,61% | 51,91% | 48,09% | 0,6978 | 0,0201 | 0,3689 | -0,0753 | 1.070,10 |
| E3 | 57 | 29,59% | 39,11% | 60,89% | 0,7456 | **0,0732** | **0,4266** | **-0,0494** | 1.747,92 |
| E4 | 93 | 34,88% | 46,66% | 53,34% | **0,8602** | 0,0288 | 0,3088 | -0,1066 | 139,31 |

### Ảnh hưởng của mô hình embedding

So với baseline đa ngôn ngữ E0, việc sử dụng Vietnamese SBERT trong E1 làm
giảm tỷ lệ ngoại lai validation từ 42,27% xuống 35,87%, tương ứng tăng độ phủ
từ 57,73% lên 64,13%. NMI tăng từ 0,3827 lên 0,4213 và ARI tăng từ 0,0205 lên
0,0720. Kết quả cho thấy biểu diễn chuyên biệt tiếng Việt giúp tạo ra không
gian thuận lợi hơn cho HDBSCAN trong các cấu hình được khảo sát. Tuy nhiên,
topic diversity của E1 thấp hơn nhẹ so với E0, cho thấy độ phủ được cải thiện
không đồng nghĩa với mức độ khác biệt từ khóa giữa các topic luôn tăng.

E1 được chọn làm cấu hình đại diện do có tỷ lệ ngoại lai validation thấp nhất
trong nhóm E0-E3. Tiêu chí này phù hợp với mục tiêu ứng dụng, trong đó hệ thống
cần cung cấp topic cho tỷ lệ bài báo lớn hơn nhưng vẫn cho phép từ chối các
trường hợp chưa thuộc cụm ổn định. Việc lựa chọn E1 không hàm ý rằng cấu hình
này vượt trội trên mọi chỉ số.

### Ảnh hưởng của UMAP và HDBSCAN

Khi tăng `n_neighbors` từ 15 lên 30 trong E2, số topic giảm nhẹ từ 96 xuống
93, nhưng tỷ lệ ngoại lai validation tăng lên 51,91%. ARI, NMI và NPMI cũng
giảm so với E1. Trong trường hợp này, việc mở rộng vùng lân cận UMAP không tạo
ra biểu diễn thuận lợi hơn cho cấu hình HDBSCAN hiện tại. Kết quả nhấn mạnh
rằng tham số UMAP và HDBSCAN có quan hệ phụ thuộc và không nên được tối ưu
hoàn toàn tách biệt.

E3 tăng `min_cluster_size` từ 15 lên 30, làm số topic giảm từ 96 xuống 57.
Topic diversity tăng từ 0,6813 lên 0,7456; ARI, NMI và NPMI cũng cao hơn nhẹ
so với E1. Đổi lại, tỷ lệ ngoại lai validation tăng từ 35,87% lên 39,11%.
Điều này thể hiện sự đánh đổi giữa mức độ khái quát và độ phủ: E3 tạo ít topic
hơn với từ khóa phân biệt hơn, trong khi E1 gán topic cho nhiều bài báo hơn.
Do đó, việc lựa chọn cấu hình phụ thuộc vào mục tiêu sử dụng thay vì một chỉ
số đơn lẻ.

### Ảnh hưởng của tiêu đề và nội dung

E4 giữ nguyên cấu hình mô hình của E1 nhưng chỉ sử dụng tiêu đề. So với E1,
độ phủ validation của E4 giảm từ 64,13% xuống 53,34%; NMI giảm từ 0,4213
xuống 0,3088; ARI giảm từ 0,0720 xuống 0,0288; và NPMI giảm từ -0,0668 xuống
-0,1066. Kết quả cho thấy tiêu đề không cung cấp đủ ngữ cảnh để tái tạo cấu
trúc chủ đề của tập tin tức, đồng thời làm tăng số tài liệu không được gán vào
cụm ổn định.

E4 đạt topic diversity 0,8602, cao hơn E1. Tuy nhiên, diversity cao cần được
đọc đồng thời với coherence và coverage. Các topic có thể sử dụng những từ
khóa ít trùng nhau nhưng không nhất thiết có mức liên kết nội tại hoặc độ phủ
tốt hơn. Do đó, kết quả ablation không ủng hộ việc thay thế nội dung đầy đủ
bằng tiêu đề trong bài toán khám phá chủ đề tổng quát.

### Khả năng diễn giải chủ đề

Từ khóa và bài đại diện cho thấy BERTopic phát hiện được cả chủ đề rộng lẫn
những nhóm gắn với sự kiện hoặc sản phẩm cụ thể. Các nhóm tiêu biểu liên quan
đến điều tra - pháp luật, sức khỏe - dinh dưỡng, giáo dục, bóng đá nữ, âm
nhạc, thiết bị iPhone và trí tuệ nhân tạo. Topic theo sự kiện hoặc sản phẩm có
thể hữu ích đối với khám phá tin tức ngắn hạn, nhưng thường nhạy hơn với sự
thay đổi của phân phối dữ liệu theo thời gian.

Các run BERTopic đều có NPMI âm. Kết quả này cho thấy sự khác biệt từ khóa
giữa các topic không đồng nghĩa với mức đồng xuất hiện cao của từ khóa trong
cùng topic. Một số nguyên nhân có thể gồm số topic lớn, sự xuất hiện của các
topic sự kiện hẹp và cách tính đồng xuất hiện ở mức tài liệu. Vì vậy, đánh giá
BERTopic cần kết hợp metric tự động với đánh giá con người dựa trên từ khóa
và bài báo đại diện.

### Chi phí tính toán

Thời gian tạo embedding chiếm phần lớn runtime của các cấu hình sử dụng
Vietnamese SBERT. Kết quả này cho thấy việc cache embedding là cần thiết khi
khảo sát tham số UMAP hoặc HDBSCAN. Đồng thời, runtime cần được tách thành các
thành phần representation, fit và prediction để tránh nhầm tổng thời gian
pipeline với thời gian huấn luyện mô hình.

Các giá trị runtime trong bảng được thu nhận từ từng lần chạy pipeline, chưa
phải benchmark lặp lại trên cùng trạng thái cache và tải hệ thống. Vì vậy,
chúng được sử dụng để nhận diện thành phần tốn thời gian, không được dùng để
kết luận chính xác rằng một cấu hình nhanh hơn cấu hình khác theo tỷ lệ tương
ứng.

### Đánh giá cuối trên tập test

Sau khi lựa chọn E1 hoàn toàn bằng kết quả validation, cấu hình này được giữ
nguyên và chạy một lần trên 1.390 bài thuộc tập test. Mô hình đạt tỷ lệ outlier
37,05%, tương ứng coverage 62,95%; ARI đạt 0,0559, NMI đạt 0,4072 và NPMI đạt
-0,0767. Các giá trị test gần với validation (outlier 35,87%, ARI 0,0720, NMI
0,4213 và NPMI -0,0668), cho thấy chất lượng không suy giảm đột ngột trên phần
dữ liệu được giữ khóa. Tuy nhiên, ARI/NMI vẫn chỉ đối chiếu với chuyên mục báo
rộng và NPMI vẫn âm, nên kết quả test không loại bỏ nhu cầu đánh giá thủ công.

Thời gian pipeline của lượt đánh giá cuối là 1.191,65 giây, trong đó bước tạo
embedding chiếm 1.084,73 giây, fit BERTopic 57,50 giây, dự đoán validation
23,48 giây và dự đoán test 1,38 giây. Runtime này bao gồm cả train,
validation và test embedding; vì vậy không được diễn giải là riêng thời gian
huấn luyện.

## Tích hợp vào hệ thống

Kết quả BERTopic được chuyển sang một schema artifact chung gồm
`manifest.json`, `config_used.yaml`, `model_summary.json`,
`document_topics.csv`, `topic_keywords.csv` và `topic_summary.csv`. Cấu trúc
này tách giao diện khỏi object mô hình trong bộ nhớ và cho phép Streamlit đọc
kết quả của BERTopic cùng các phương pháp K-Means, NMF, LDA và HDBSCAN theo
một hợp đồng dữ liệu thống nhất.

Giao diện hỗ trợ xem tổng quan mô hình, khám phá chi tiết topic, tìm kiếm bài
báo, trực quan hóa phép chiếu hai chiều, theo dõi phân bố theo thời gian, so
sánh run và xem các trường hợp cần phân tích lỗi. Tên hiển thị và nhóm topic
được lưu dưới dạng mapping riêng, do đó không làm thay đổi kết quả phân cụm
gốc. Cách thiết kế này bảo toàn khả năng tái lập và cho phép đối chiếu trực
tiếp giữa giao diện với artifact thực nghiệm.

API nội bộ hiện cung cấp endpoint kiểm tra trạng thái hệ thống. Chức năng suy
luận cho bài báo mới chỉ được xem là hoàn chỉnh khi endpoint dự đoán, model đã
lưu và kiểm thử đầu-cuối được xác nhận. Các chức năng phát hiện sự kiện hoặc
xu hướng cũng không được xem là kết quả nghiên cứu của BERTopic nếu chưa có
thuật toán và quy trình đánh giá độc lập.

## Hạn chế

Thứ nhất, chỉ E1 được đánh giá trên test sau khi khóa cấu hình. E0, E2, E3 và
E4 vẫn chỉ là các thí nghiệm lựa chọn/ablation trên validation; không được dùng
kết quả test để xếp hạng lại hoặc chọn cấu hình khác.

Thứ hai, nhãn chuyên mục của nguồn báo chỉ phản ánh lĩnh vực rộng và không đại
diện hoàn toàn cho cấu trúc topic hoặc sự kiện. ARI và NMI vì thế chỉ cung cấp
một góc nhìn tham chiếu. Thứ ba, các cấu hình E0-E3 tạo từ 57 đến 111 topic,
làm tăng chi phí đánh giá thủ công và có nguy cơ sinh các chủ đề quá hẹp.

Thứ tư, tỷ lệ ngoại lai validation vẫn dao động từ 35,87% đến 51,91%. Một số
ngoại lai có thể là dữ liệu nhiễu, nhưng một số khác có thể là bài hiếm hoặc
bài đa chủ đề hợp lệ. Việc phân loại thủ công là cần thiết trước khi xem toàn
bộ nhóm ngoại lai là lỗi. Cuối cùng, tên topic hiện được gợi ý từ từ khóa và
bài đại diện; chất lượng tên vẫn phụ thuộc vào đánh giá con người và cần được
xác nhận bởi nhiều người đánh giá độc lập.

## Tài liệu tham khảo

- Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a
  class-based TF-IDF procedure*. arXiv:2203.05794.
  https://doi.org/10.48550/arXiv.2203.05794
- Janssens, R., Carlee, J., Herten, M., Meyer, F., & Fuchs, G. (2025).
  *BERTopic for topic modeling of Dutch legislative proceedings*. PeerJ
  Computer Science, 11, e2778. https://doi.org/10.7717/peerj-cs.2778
