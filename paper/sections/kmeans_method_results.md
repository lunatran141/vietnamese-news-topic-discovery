# 4.2. Phương pháp biểu diễn TF-IDF và phân cụm K-Means

## 4.2.1. Biểu diễn văn bản bằng trọng số TF-IDF dưới tuyến tính

Trong nhánh tiếp cận phân cụm khoảng cách cổ điển, mỗi bài báo sau khi qua chuỗi tiền xử lý `P3` được ánh xạ vào không gian vector thực nhiều chiều thông qua trọng số TF-IDF. Một hạn chế phổ biến của công thức tần suất từ tuyến tính truyền thống là các bài phóng sự dài thường lặp lại một số từ khóa nhiều lần, khiến trọng số của từ đó tăng vọt và lấn át các thuật ngữ quan trọng khác. Để khắc phục hiện tượng này, nhóm nghiên cứu áp dụng biến đổi tần suất từ dưới tuyến tính (*sublinear TF scaling*) kết hợp với nghịch đảo tần suất văn bản có làm trơn:

$$\text{tf}(t, d) = \begin{cases} 1 + \ln f_{t,d}, & \text{nếu } f_{t,d} > 0 \\ 0, & \text{nếu } f_{t,d} = 0 \end{cases}, \qquad \text{idf}(t) = \ln\left(\frac{1 + N_{\text{train}}}{1 + \text{df}(t)}\right) + 1 \quad (4.1)$$

trong đó $f_{t,d}$ là số lần từ $t$ xuất hiện trong văn bản $d$, $N_{\text{train}} = 6.493$ là tổng số văn bản của tập Train, và $\text{df}(t)$ là số văn bản trong tập Train có chứa từ $t$. Trọng số thô của mỗi từ được tính bằng tích $w_{t,d} = \text{tf}(t, d) \cdot \text{idf}(t)$.

Do độ dài các bài báo trong ngữ liệu chênh lệch rất lớn (từ 50 đến 5.885 từ), hai bài viết cùng chủ đề nhưng khác độ dài sẽ có độ lớn vector rất khác nhau nếu giữ nguyên trọng số thô. Vì vậy, mỗi vector văn bản $\mathbf{w}_d = (w_{1,d}, w_{2,d}, \dots, w_{V,d})^\top$ được chuẩn hóa về độ dài đơn vị theo chuẩn $\ell_2$:

$$\mathbf{x}_d = \frac{\mathbf{w}_d}{\|\mathbf{w}_d\|_2} = \frac{\mathbf{w}_d}{\sqrt{\sum_{j=1}^{V} w_{j,d}^2}}, \qquad \|\mathbf{x}_d\|_2 = 1 \quad (4.2)$$

Sau bước chuẩn hóa này, toàn bộ các bài báo đều nằm trên mặt cầu đơn vị $(V - 1)$ chiều, giúp loại bỏ hoàn toàn sự thiên lệch do độ dài văn bản gây ra.

---

## 4.2.2. Phân cụm K-Means và mối liên hệ với độ tương đồng Cosine

Trên không gian đặc trưng TF-IDF đã chuẩn hóa, thuật toán K-Means phân hoạch tập $N$ văn bản thành $K$ cụm rời nhau $\mathcal{C} = \{C_1, C_2, \dots, C_K\}$ sao cho mỗi văn bản chỉ thuộc về một cụm duy nhất. Thuật toán tìm cách cực tiểu hóa tổng bình phương khoảng cách trong cụm (WCSS / Inertia):

$$\mathcal{J}(\mathcal{C}, \{\boldsymbol{\mu}_k\}_{k=1}^K) = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|_2^2, \qquad \text{với } \boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i \quad (4.3)$$

Mặc dù K-Means sử dụng khoảng cách Euclid, việc áp dụng trên các vector đã chuẩn hóa $\ell_2$ ($\|\mathbf{x}_i\|_2 = 1$) tạo ra một tính chất hình học quan trọng. Cụ thể, với hai vector đơn vị $\mathbf{x}_i$ và $\mathbf{x}_j$ bất kỳ, bình phương khoảng cách Euclid giữa chúng có mối quan hệ tuyến tính nghịch biến với độ tương đồng Cosine [1]:

$$\|\mathbf{x}_i - \mathbf{x}_j\|_2^2 = \|\mathbf{x}_i\|_2^2 + \|\mathbf{x}_j\|_2^2 - 2\mathbf{x}_i^\top \mathbf{x}_j = 2\left(1 - \cos(\mathbf{x}_i, \mathbf{x}_j)\right) \quad (4.4)$$

Tương tự, khi xét khoảng cách từ một văn bản $\mathbf{x}_i$ đến tâm cụm $\boldsymbol{\mu}_k$, ta có $\|\mathbf{x}_i - \boldsymbol{\mu}_k\|_2^2 = 1 + \|\boldsymbol{\mu}_k\|_2^2 - 2\|\boldsymbol{\mu}_k\|_2 \cos(\mathbf{x}_i, \boldsymbol{\mu}_k)$. Như vậy, việc gán văn bản vào tâm cụm có khoảng cách Euclid ngắn nhất cũng tương ứng với việc gom các bài báo có hướng ngữ nghĩa gần nhau nhất theo góc Cosine [1].

Để hạn chế việc thuật toán hội tụ về các cực tiểu cục bộ kém chất lượng trên không gian thưa nhiều chiều, nhóm sử dụng phương pháp khởi tạo tâm cụm `k-means++`. Các tâm cụm ban đầu được chọn ngẫu nhiên có trọng số tỷ lệ thuận với bình phương khoảng cách tới tâm gần nhất đã chọn, giúp các tâm khởi tạo phân bố trải đều khắp không gian từ vựng trước khi thực hiện vòng lặp cập nhật Lloyd. Sau khi mô hình hội tụ, các từ khóa đại diện cho mỗi chủ đề $k$ được trích xuất trực tiếp bằng cách xếp hạng các chiều từ vựng có giá trị lớn nhất trong vector tâm cụm $\boldsymbol{\mu}_k$.

---

# 5. Thiết lập thực nghiệm cho nhánh TF-IDF + K-Means

## 5.1. Cấu hình trích xuất đặc trưng và huấn luyện

Không gian đặc trưng TF-IDF chuẩn được xây dựng trên biến thể tiền xử lý `P3` (văn bản đã tách từ bằng `pyvi` và lọc 388 từ dừng). Nhờ bước tách từ đã nối sẵn các từ ghép tiếng Việt bằng dấu gạch dưới `_`, bộ trích xuất đặc trưng chỉ cần sử dụng cấu hình unigram (`ngram_range = (1, 1)`) mà vẫn thu được đầy đủ các khái niệm đa âm tiết (như `bất_động_sản`, `chứng_khoán`, `trí_tuệ_nhân_tạo`). Để loại bỏ các từ gõ sai hoặc cực hiếm đồng thời lọc bớt các từ xuất hiện tràn lan, nhóm thiết lập ngưỡng tần suất tài liệu tối thiểu `min_df = 2`, tần suất tài liệu tối đa `max_df = 0.85` và giới hạn kích thước từ điển ở mức $V = 20.000$ đặc trưng có trọng số cao nhất. Bộ trích xuất TF-IDF chỉ được học (`fit`) trên 6.493 bài thuộc tập Train, tạo ra ma trận đặc trưng thưa có độ thưa (*sparsity*) khoảng 99,05%.

Đối với thuật toán K-Means, mỗi cấu hình được huấn luyện với `n_init = 10` lần khởi tạo `k-means++` độc lập và chọn nghiệm có giá trị Inertia thấp nhất, số vòng lặp tối đa `max_iter = 300`, ngưỡng hội tụ $\text{tol} = 10^{-4}$ và hạt giống ngẫu nhiên chuẩn `seed = 42`.

---

## 5.2. Giao thức tìm kiếm số cụm $K$ và đánh giá độ ổn định

Vì số lượng chủ đề thực sự trong dòng tin tức không được biết trước và có độ phân giải mịn hơn so với 7 chuyên mục biên tập lớn của tòa soạn, nhóm khảo sát lưới giá trị $K \in \{3, 5, 7, 9, 11, 14, 18\}$ trên tập Validation (1.391 bài). Trong đó, $K = 7$ đóng vai trò là mốc đối chiếu trùng với số chuyên mục gốc.

Việc lựa chọn $K$ tối ưu không dựa vào một chỉ số đơn lẻ mà kết hợp bốn tiêu chí trên tập Validation:
1. **Các chỉ số hình học nội tại:** Hệ số Silhouette (trên khoảng cách Euclid), chỉ số Davies-Bouldin (DBI) và chỉ số Calinski-Harabasz (CH), kèm theo mức độ giảm Inertia trên tập Train.
2. **Độ ổn định đa hạt giống (*multi-seed stability*):** Tại mỗi giá trị $K$, mô hình được huấn luyện độc lập với 5 hạt giống ngẫu nhiên khác nhau $\{42, 52, 62, 72, 82\}$. Độ ổn định được đo bằng giá trị ARI trung bình của $\binom{5}{2} = 10$ cặp kết quả phân cụm trên tập Validation.
3. **Phân bố kích thước cụm:** Kiểm tra kích thước cụm nhỏ nhất và lớn nhất để loại trừ các cấu hình sinh ra cụm suy biến (chỉ gồm vài bài báo lẻ loi) hoặc gom phần lớn dữ liệu vào một cụm.
4. **Đánh giá định tính thủ công:** Với các cấu hình ứng viên tốt nhất ($K \in \{7, 9, 11\}$), nhóm trích xuất 10 từ khóa đứng đầu cùng 5 bài báo gần tâm nhất và 5 bài báo ngẫu nhiên của từng cụm để chấm điểm độ mạch lạc ngữ nghĩa trên thang điểm từ 1 đến 5.

---

## 5.3. Thiết kế các thí nghiệm cắt giảm

Nhằm trả lời hai câu hỏi nghiên cứu cốt lõi của bài báo về đặc thù tiếng Việt, nhóm thực hiện hai thí nghiệm cắt giảm có kiểm soát chặt chẽ trên tập Validation:

- **Thí nghiệm cắt giảm tiền xử lý (RQ2):** Cố định hoàn toàn thuật toán K-Means ($K = 9$ và kiểm tra độ nhạy tại $K = 7$) cùng toàn bộ siêu tham số TF-IDF, chỉ thay đổi lần lượt 4 biến thể văn bản đầu vào `P0`, `P1`, `P2` và `P3`. Để đảm bảo phép thử công bằng tuyệt đối, chế độ tự động chuyển chữ thường và lọc từ dừng bên trong bộ vector hóa được tắt hoàn toàn nhằm giữ nguyên bản chất của từng biến thể tiền xử lý. Bên cạnh các chỉ số phân cụm (Silhouette, DBI, CH, Purity, NMI, ARI), nhóm đo thêm tỷ lệ từ ngoài từ điển (OOV rate) trên tập Validation, tỷ lệ từ ghép (`compound_ratio`) và tỷ lệ nhiễu từ dừng (`stopword_ratio`) trong 15 từ khóa đại diện hàng đầu của mỗi cụm.
- **Thí nghiệm phạm vi văn bản đầu vào (RQ3):** Cố định mức tiền xử lý tối ưu `P3` và cấu hình K-Means ($K = 9$ và $K = 7$), tiến hành đối chiếu trực tiếp giữa biến thể chỉ dùng tiêu đề (`Title-only`) và biến thể ghép tiêu đề với nội dung (`Title+Content`). Thí nghiệm này đánh giá cả chất lượng phân cụm, sự phân bố kích thước cụm lẫn thời gian chạy thực tế.



---

# 6. Kết quả thực nghiệm và thảo luận (Nhánh TF-IDF + K-Means)

## 6.1. Lựa chọn số cụm $K$ và kết quả đánh giá trên tập Test

Bảng 6.1 trình bày kết quả khảo sát lưới $K \in \{3, 5, 7, 9, 11, 14, 18\}$ trên tập Validation (1.391 bài báo).

**Bảng 6.1.** Kết quả khảo sát số cụm $K$ của mô hình TF-IDF + K-Means trên tập Validation.

| Số cụm $K$ | Silhouette | DBI | CH | Độ ổn định đa seed (ARI) | Cụm nhỏ nhất (bài) | Giảm Train Inertia | Purity | NMI | ARI |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 3 | 0,0147 | 7,149 | 16,65 | 0,849 ± 0,171 | 218 | — | 0,510 | 0,479 | 0,255 |
| 5 | 0,0189 | 7,194 | 14,19 | 0,917 ± 0,089 | 163 | 87,5 | 0,724 | 0,657 | 0,561 |
| 7 | 0,0205 | 6,618 | 12,48 | 0,828 ± 0,116 | 90 | 63,6 | 0,821 | 0,689 | 0,624 |
| **9** | **0,0190** | **6,157** | **11,37** | **0,822 ± 0,044** | **76** | **53,6** | **0,873** | **0,700** | **0,593** |
| 11 | 0,0205 | 5,826 | 10,59 | 0,806 ± 0,051 | 30 | 47,2 | 0,825 | 0,646 | 0,490 |
| 14 | 0,0194 | 5,629 | 9,68 | 0,699 ± 0,083 | 30 | 49,3 | 0,838 | 0,627 | 0,438 |
| 18 | 0,0191 | 5,603 | 8,77 | 0,713 ± 0,069 | 7 | 49,3 | 0,881 | 0,623 | 0,391 |

Số liệu thực nghiệm ở Bảng 6.1 cho thấy ba đặc điểm hình học điển hình của bài toán phân cụm văn bản trên không gian TF-IDF 20.000 chiều. Thứ nhất, hệ số Silhouette hầu như phẳng trên toàn dải $K$ (dao động hẹp từ 0,0147 đến 0,0205). Nguyên nhân là trên mặt cầu đơn vị nhiều chiều với độ thưa hơn 99%, khoảng cách từ một bài báo tới tâm cụm gần nhất chỉ nhỏ hơn đôi chút so với khoảng cách tới các tâm cụm khác (khoảng cách trung bình tới tâm trên toàn tập đạt $0{,}948 \pm 0{,}028$). Thứ hai, chỉ số DBI giảm đơn điệu theo $K$ (từ 7,194 xuống 5,603), trong khi mức giảm Inertia trên tập Train không tạo ra điểm gãy khuỷu tay (*elbow*) rõ rệt. Thứ ba, ở các giá trị $K$ lớn ($K = 14$ và $K = 18$), độ ổn định qua 5 hạt giống ngẫu nhiên sụt giảm mạnh (ARI trung bình rơi xuống 0,699 và 0,713), đồng thời xuất hiện cụm suy biến chỉ gồm 7 bài báo tại $K = 18$.

Khi đối chiếu ba cấu hình ứng viên ổn định nhất là $K \in \{7, 9, 11\}$, cấu hình **$K = 9$** được chọn làm mô hình tối ưu vì sự cân bằng giữa độ ổn định, kích thước cụm và khả năng phân tách ngữ nghĩa:
- **So với $K = 7$:** Mặc dù ngữ liệu có 7 chuyên mục gốc, hai chuyên mục Kinh doanh và Thể thao có quy mô lớn gấp đôi các chuyên mục khác và chứa các nhánh sự kiện rất khác biệt. Ở $K = 7$, mảng Khoa học công nghệ bị trộn lẫn vào cụm Doanh nghiệp — Đầu tư. Khi tăng lên $K = 9$, mô hình tách riêng được một cụm Công nghệ độc lập (Cụm 2: thiết bị Apple và AI, độ tinh khiết 92,1%), đồng thời tách mảng Thể thao thành hai cụm rõ rệt là Bóng đá (Cụm 4) và Đại hội thể thao ASIAD đa môn (Cụm 1), đồng thời tách mảng Kinh doanh thành Thị trường tài chính — Giá vàng (Cụm 8) và Doanh nghiệp vĩ mô (Cụm 6).
- **So với $K = 11$:** Cấu hình $K = 9$ có độ ổn định đa hạt giống cao hơn và phương sai nhỏ hơn ($0{,}822 \pm 0{,}044$ so với $0{,}806 \pm 0{,}051$), kích thước cụm nhỏ nhất đạt 76 bài (so với cụm quá nhỏ chỉ 30 bài ở $K = 11$), đồng thời đạt giá trị cao nhất trên cả Purity (0,873) và NMI (0,700).

**[HÌNH/BẢNG: Hình 6.1 — Biểu đồ đường cong Inertia, các chỉ số nội tại và độ ổn định đa hạt giống theo số cụm $K$.]**

Bảng 6.2 tổng hợp cấu trúc chi tiết và kết quả chấm điểm độ mạch lạc ngữ nghĩa thủ công (thang điểm 1–5) của 9 cụm ở cấu hình $K = 9$.

**Bảng 6.2.** Cấu trúc 9 cụm chủ đề và đánh giá định tính của mô hình TF-IDF + K-Means ($K = 9$).

| Cụm | Số bài (Train / Val) | Chuyên mục chiếm ưu thế (Val) | Từ khóa đại diện hàng đầu (Top keywords) | Nhãn chủ đề diễn giải | Điểm mạch lạc (1–5) |
| :---: | :---: | :--- | :--- | :--- | :---: |
| 0 | 1.135 / 251 | Giải trí (68,1%) | `cô, phim, anh, khán_giả, không, diễn_viên, nghệ_sĩ, tác_phẩm` | Giải trí, nghệ sĩ và đời sống | 3 |
| 1 | 395 / 98 | Thể thao (99,0%) | `asiad, nữ, u23, nam, giành, thi_đấu, việt_nam, trận, môn` | Thể thao: ASIAD và đội tuyển đa môn | 4 |
| 2 | 377 / 76 | KH-CN (92,1%) | `iphone, apple, pro, máy, dùng, màn_hình, max, hệ_thống, thiết_bị` | Công nghệ: Thiết bị Apple và AI | 4 |
| 3 | 722 / 159 | Sức khỏe (98,7%) | `bác_sĩ, bệnh, điều_trị, bệnh_viện, người_bệnh, khám, máu, thuốc` | Y tế, bệnh lý và điều trị | 5 |
| 4 | 832 / 174 | Thể thao (100,0%) | `trận, đội, bóng, cầu_thủ, hlv, cup, thắng, sân, fifa, bàn` | Bóng đá trong nước và quốc tế | 5 |
| 5 | 882 / 188 | Pháp luật (87,2%) | `công_an, điều_tra, án, vụ, cơ_quan, hành_vi, tiền, cảnh_sát` | Pháp luật hình sự và điều tra | 4 |
| 6 | 1.109 / 229 | Kinh doanh (74,7%) | `doanh_nghiệp, phát_triển, đầu_tư, công_nghệ, dự_án, thị_trường` | Doanh nghiệp, đầu tư và dịch vụ | 2 |
| 7 | 650 / 129 | Giáo dục (97,7%) | `trường, học_sinh, giáo_dục, học, đào_tạo, đại_học, nhà_trường, lớp` | Giáo dục và đào tạo | 5 |
| 8 | 391 / 87 | Kinh doanh (96,6%) | `tăng, giá, đồng, lãi_suất, thị_trường, giảm, usd, cổ_phiếu, bán` | Giá vàng, lãi suất và chứng khoán | 5 |

Điểm mạch lạc trung bình trên 9 cụm đạt **4,11 / 5,00**. Trong đó, 7/9 cụm có ranh giới từ vựng rất sắc nét và độ tinh khiết từ 87,2% đến 100%. Hai cụm có điểm mạch lạc thấp hơn là Cụm 6 (2/5 điểm, độ tinh khiết 74,7%) và Cụm 0 (3/5 điểm, độ tinh khiết 68,1%). Phân tích các bài viết ngẫu nhiên trong Cụm 6 cho thấy đây là một "cụm thu gom" (*catch-all cluster*): các bài báo viết về chuyển đổi số, hạ tầng hàng không, du học hay bán dẫn đều sử dụng chung lớp từ vựng kinh tế vĩ mô (`doanh_nghiệp`, `phát_triển`, `đầu_tư`, `dự_án`), nên bị kéo về cùng một tâm cụm. Tương tự, Cụm 0 thu hút các bài viết giàu đại từ nhân xưng và tính tự sự (`cô`, `anh`, `tôi`), khiến một số bài viết về phong cách sống hoặc bài quảng cáo thương hiệu bị gộp chung với tin tức nghệ sĩ.

**[HÌNH/BẢNG: Hình 6.2 — Phân bố kích thước 9 cụm và biểu đồ chiếu giảm chiều 2D của các cụm K-Means ($K = 9$).]**

Sau khi khóa cố định cấu hình $K = 9$ (`seed = 42`), mô hình được đánh giá một lần duy nhất trên tập Test độc lập (1.390 bài báo). Kết quả đối chiếu giữa tập Validation và tập Test được trình bày tại Bảng 6.3.

**Bảng 6.3.** Kết quả đánh giá mô hình TF-IDF + K-Means ($K = 9$) trên tập Validation và tập Test khóa kín.

| Tập dữ liệu | Số bài | Silhouette | DBI | CH | Purity | NMI | ARI | Topic Diversity | Tổng thời gian chạy (giây) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Validation | 1.391 | 0,0190 | 6,1566 | 11,3745 | 0,8728 | 0,7004 | 0,5925 | 0,9556 | — |
| **Test (Final)** | **1.390** | **0,0201** | **6,1954** | **11,5909** | **0,8597** | **0,6774** | **0,5649** | **0,9556** | **7,75** |

Trên tập Test, mô hình duy trì hiệu năng rất sát với tập Validation, đạt Purity = **85,97%**, NMI = **0,6774**, ARI = **0,5649** và độ đa dạng từ khóa Topic Diversity = **0,9556**. Tổng thời gian thực thi toàn trình chỉ mất **7,75 giây** (gồm 3,91 giây trích xuất đặc trưng TF-IDF, 2,84 giây huấn luyện K-Means, 0,0044 giây suy luận trên tập Test và 0,99 giây tính toán chỉ số). Kết quả này khẳng định mô hình có khả năng khái quát hóa tốt và không xảy ra hiện tượng quá khớp (*overfitting*).

---

## 6.2. Nghiên cứu cắt giảm tiền xử lý (Trả lời câu hỏi nghiên cứu RQ2)

Để định lượng đóng góp của từng bước tiền xử lý tiếng Việt, Bảng 6.4 trình bày kết quả thực nghiệm khi thay đổi lần lượt 4 biến thể `P0`, `P1`, `P2` và `P3` trên cùng cấu hình K-Means ($K = 9$).

**Bảng 6.4.** Ảnh hưởng của các cấp độ tiền xử lý từ `P0` đến `P3` đến chất lượng phân cụm K-Means ($K = 9$, tập Validation).

| Chỉ số đánh giá | `P0` (Tối thiểu) | `P1` (Chuẩn hóa) | `P2` (Tách từ) | **`P3` (Lọc từ dừng)** |
| :--- | :---: | :---: | :---: | :---: |
| Kích thước từ vựng huấn luyện | 17.983 | 15.285 | 20.000 | **20.000** |
| Độ thưa ma trận Validation (%) | 98,34% | 98,17% | 98,79% | **99,05%** |
| Tỷ lệ từ ngoài từ điển — OOV (%) | 9,86% | 11,04% | 11,97% | **2,56%** |
| Số từ kích hoạt trung bình / bài | 299,27 | 279,88 | 241,80 | **190,75** |
| Silhouette | 0,0193 | 0,0186 | 0,0181 | **0,0190** |
| Davies-Bouldin (DBI) | 5,7112 | **5,5708** | 6,1151 | 6,1566 |
| Calinski-Harabasz (CH) | 13,15 | **14,17** | 11,50 | 11,38 |
| **Purity (%)** | 83,61% | 81,52% | 81,60% | **87,28%** |
| **NMI** | 0,6652 | 0,6503 | 0,6494 | **0,7004** |
| **ARI** | 0,5457 | 0,5161 | 0,4958 | **0,5925** |
| Độ ổn định đa seed (ARI) | 0,8599 ± 0,0487 | 0,8267 ± 0,0612 | 0,8186 ± 0,0478 | **0,8211 ± 0,0442** |
| **Tỷ lệ từ ghép trong Top-15 (%)** | 0,00% | 0,00% | 32,59% | **43,70%** |
| **Tỷ lệ nhiễu từ dừng trong Top-15 (%)** | 9,63% | 9,63% | 18,52% | **0,00%** |
| Topic Diversity (Top-10) | 0,9000 | 0,8778 | 0,9222 | **0,9556** |

Kết quả thực nghiệm ở Bảng 6.4 làm sáng tỏ hai cơ chế ngôn ngữ học quan trọng khi xử lý văn bản tiếng Việt:

1. **Nghịch lý giữa chỉ số hình học nội tại và chất lượng ngữ nghĩa ngoại tại:** Nếu chỉ quan sát các chỉ số hình học nội tại, biến thể `P1` đạt điểm DBI thấp nhất (5,5708) và CH cao nhất (14,17), trong khi biến thể hoàn chỉnh `P3` lại có DBI cao hơn (6,1566). Nguyên nhân là ở `P0` và `P1`, văn bản chưa được tách từ và chưa lọc từ dừng, khiến các âm tiết đơn xuất hiện dày đặc ở mọi bài báo (như `các`, `của`, `người`, `và`, `được`, `trong`). Các âm tiết chung này đóng vai trò như một "lực hút giả", kéo vector của mọi bài viết co cụm lại gần nhau quanh một hướng chung, làm giảm khoảng cách hình học nội cụm một cách cơ học. Tuy nhiên, các cụm thu được lại bị nhiễu nặng bởi các âm tiết vô nghĩa (`0%` từ ghép). Khi sang `P3`, toàn bộ 388 từ dừng bị loại bỏ, giải phóng các vector bài báo tỏa ra đúng các trục thuật ngữ chuyên ngành. Dù các cụm giãn rộng hơn đôi chút về mặt hình học, độ chính xác theo chủ đề thực tế lại tăng vọt lên mức cao nhất trên mọi thước đo: Purity tăng từ 81,52% lên **87,28%** (+5,76 điểm phần trăm), NMI tăng từ 0,6503 lên **0,7004**, và ARI tăng từ 0,5161 lên **0,5925**.
2. **Sự cộng hưởng bắt buộc giữa tách từ (`P2`) và lọc từ dừng (`P3`):** Nếu chỉ tách từ bằng `pyvi` mà không lọc từ dừng (`P2`), hiệu năng phân cụm gần như giậm chân tại chỗ (Purity đạt 81,60%, ARI giảm nhẹ xuống 0,4958) và tỷ lệ nhiễu từ dừng trong Top-15 từ khóa tăng vọt lên 18,52% (25/135 từ khóa đại diện là từ dừng ghép như `nhưng_không`, `cho_biết`, `có_thể`). Lý do là khi các âm tiết nội dung được gom lại thành từ ghép (`bệnh_viện`, `học_sinh`), tần suất xuất hiện của chúng giảm xuống so với các từ dừng đơn âm tiết, khiến từ dừng càng chiếm ưu thế trong trọng số tâm cụm. Chỉ khi kết hợp tách từ với lọc từ dừng ở `P3`, các từ dừng mới bị triệt tiêu hoàn toàn (**0,00% nhiễu**), giúp tỷ lệ từ ghép có nghĩa trong tâm cụm nhảy vọt lên **43,70%** và giảm tỷ lệ OOV trên tập Validation từ 11,97% xuống chỉ còn **2,56%**. Kiểm tra độ nhạy tại $K = 7$ cũng cho kết quả hoàn toàn nhất quán (`P3` đạt Purity = 82,10% và NMI = 0,6889, vượt xa `P1` và `P2`).

**[HÌNH/BẢNG: Hình 6.3 — So sánh các chỉ số ngoại tại (Purity, NMI, ARI) và chất lượng từ vựng tâm cụm qua 4 biến thể tiền xử lý P0–P3.]**

---

## 6.3. Nghiên cứu phạm vi đầu vào: chỉ tiêu đề so với tiêu đề + nội dung (Trả lời RQ3)

Bảng 6.5 đối chiếu trực tiếp hiệu năng của mô hình K-Means ($K = 9$ và kiểm tra độ nhạy tại $K = 7$) khi chỉ sử dụng tiêu đề (`Title-only`) so với khi kết hợp tiêu đề và toàn văn (`Title+Content`) trên nền tiền xử lý `P3`.

**Bảng 6.5.** So sánh thực nghiệm giữa biến thể chỉ tiêu đề (`Title-only`) và tiêu đề kết hợp nội dung (`Title+Content`) trên tập Validation.

| Chỉ số đánh giá | `Title-only` ($K = 9$) | `Title+Content` ($K = 9$) | `Title-only` ($K = 7$) | `Title+Content` ($K = 7$) |
| :--- | :---: | :---: | :---: | :---: |
| Kích thước từ vựng hiệu dụng | 4.437 | **20.000** | 4.437 | **20.000** |
| Số từ kích hoạt trung bình / bài | 7,66 | **190,75** | 7,66 | **190,75** |
| Độ thưa ma trận Validation (%) | 99,83% | **99,05%** | 99,83% | **99,05%** |
| Tỷ lệ OOV tập Validation (%) | 10,45% | **2,56%** | 10,45% | **2,56%** |
| Silhouette | 0,0063 | **0,0190** | 0,0053 | **0,0205** |
| Davies-Bouldin (DBI) | 5,8107 | 6,1566 | 6,7098 | **6,6178** |
| **Purity (%)** | 30,77% | **87,28%** | 29,55% | **82,10%** |
| **NMI** | 0,1637 | **0,7004** | 0,1047 | **0,6889** |
| **ARI** | -0,0016 | **0,5925** | -0,0046 | **0,6239** |
| Độ ổn định đa seed (ARI) | 0,5496 ± 0,0894 | **0,8211 ± 0,0442** | 0,5983 ± 0,1139 | **0,8277 ± 0,1158** |
| Kích thước cụm (Nhỏ nhất / Lớn nhất) | 13 / **982** | **76 / 251** | 12 / **1.092** | **90 / 297** |
| Tỷ lệ từ ghép trong Top-15 (%) | 17,04% | **43,70%** | 19,05% | **42,86%** |
| Topic Diversity (Top-10) | 0,7556 | **0,9556** | 0,7429 | **0,9286** |
| Tổng thời gian chạy thực nghiệm (giây) | **2,27** | 25,38 | **1,85** | 21,40 |

Số liệu tại Bảng 6.5 chứng minh hiện tượng **sụp đổ cụm (*cluster collapse*)** nghiêm trọng khi áp dụng K-Means trên văn bản tiêu đề ngắn. Do mỗi tiêu đề sau khi lọc từ dừng chỉ còn trung bình 7,66 từ khóa khác không trên không gian 4.437 chiều (độ thưa lên tới 99,83% và tỷ lệ OOV tới 10,45%), phần lớn các cặp tiêu đề không chia sẻ chung từ khóa nào, khiến tích vô hướng giữa chúng bằng 0 ($\cos(\mathbf{x}_i, \mathbf{x}_j) = 0$) và khoảng cách Euclid giữa hầu hết các điểm đều xấp xỉ $\sqrt{2}$. Khi thiếu sự giao thoa từ vựng để định hình các tâm cụm chuyên biệt, thuật toán K-Means đẩy tới **982 / 1.391 bài báo (70,60% toàn tập Validation) dồn vào một cụm khổng lồ duy nhất** (tại $K = 7$, cụm khổng lồ này hút tới 1.092 bài, tương đương 78,50%). Hệ quả là chỉ số ARI của `Title-only` rơi xuống mức âm nhẹ (**-0,0016** tại $K = 9$ và **-0,0046** tại $K = 7$), tức là không tốt hơn việc gán nhãn ngẫu nhiên, còn Purity giảm sâu xuống 30,77%.

Ngược lại, việc bổ sung phần nội dung bài báo trong `Title+Content` nâng số từ kích hoạt trung bình lên 190,75 từ/bài (tăng gấp 24,9 lần), tạo ra mạng lưới đồng xuất hiện từ vựng đủ dày để neo giữ các tâm cụm. Nhờ đó, Purity tăng vọt thêm **+56,51 điểm phần trăm** (từ 30,77% lên 87,28%), NMI tăng từ 0,1637 lên 0,7004 và ARI tăng từ -0,0016 lên 0,5925. Xét theo từng chuyên mục, độ tinh khiết của mảng Thể thao tăng từ 14,26% lên 100%, Sức khỏe tăng từ 16,19% lên 98,74%, và Khoa học công nghệ tăng từ 7,03% lên 92,11%. Dù biến thể toàn văn tốn thời gian tính toán nhiều hơn so với chỉ dùng tiêu đề (25,38 giây so với 2,27 giây cho toàn bộ quy trình đánh giá đa hạt giống), mức chênh lệch vài chục giây là hoàn toàn không đáng kể so với sự cải thiện quyết định về chất lượng phân cụm.

**[HÌNH/BẢNG: Hình 6.4 — Đối chiếu hiệu năng phân cụm và sự suy sụp phân bố kích thước cụm giữa `Title-only` và `Title+Content`.]**

---

## 6.4. Thảo luận về ưu điểm và hạn chế của nhánh TF-IDF + K-Means

Từ toàn bộ kết quả thực nghiệm, có thể rút ra những nhận định khách quan về vai trò của nhánh mô hình cổ điển TF-IDF + K-Means trong bài toán khám phá chủ đề báo chí tiếng Việt:

- **Ưu điểm nổi bật:** Khi được kết hợp với quy trình tách từ tiếng Việt và lọc từ dừng hợp lý (`P3`) trên văn bản toàn văn (`Title+Content`), TF-IDF + K-Means là một đường cơ sở (*baseline*) cực kỳ mạnh mẽ, tốc độ cao (huấn luyện và suy luận chỉ mất vài giây trên CPU) và có độ tinh khiết ngoại tại rất cao (Test Purity đạt 85,97%, Topic Diversity đạt 0,9556). Trọng số tâm cụm phản ánh trực diện các thuật ngữ đặc trưng của từng lĩnh vực mà không đòi hỏi tài nguyên GPU.
- **Hạn chế cố hữu:**
  1. *Cơ chế phân cụm cứng (*hard assignment*):* Mỗi bài báo bị ép buộc gán vào một cụm duy nhất. Với các bài báo giao thoa nhiều lĩnh vực (ví dụ: chính sách thu phí học đường của ngành y, hoặc vụ án lừa đảo tài chính ngân hàng), mô hình không thể biểu diễn tỷ trọng đa chủ đề như LDA hay NMF, dẫn đến sự hình thành của cụm thu gom chung chung (như Cụm 6).
  2. *Hạn chế của biểu diễn túi từ:* TF-IDF dựa hoàn toàn vào sự trùng khớp bề mặt từ vựng nên không tự nhận diện được các từ đồng nghĩa hoặc cách diễn đạt khác nhau của cùng một khái niệm như các mô hình nhúng ngữ cảnh (PhoBERT/SBERT), khiến phương pháp này hoàn toàn thất bại trên văn bản cực ngắn (`Title-only`).
  3. *Tính nhạy cảm với sự kiện ngắn hạn:* Do ngữ liệu thu thập trong tháng 09/2026, K-Means dễ bị thu hút bởi các từ khóa sự kiện có tần suất đột biến trong tháng (như `asiad`, `iphone`), tạo ra các cụm sự kiện bên cạnh các cụm chủ đề bền vững.

---

# Tài liệu tham khảo (Phần của TV1)

[1] M. Steinbach, G. Karypis, and V. Kumar, "A comparison of document clustering techniques," in *Proc. KDD Workshop on Text Mining*, Boston, MA, USA, 2000, pp. 1–20.

[2] T. Vu, D. Q. Nguyen, D. Q. Nguyen, M. Dras, and M. Johnson, "VnCoreNLP: A Vietnamese natural language processing toolkit," in *Proc. 2018 Conf. North American Chapter of the Association for Computational Linguistics: Demonstrations (NAACL-HLT)*, New Orleans, LA, USA, 2018, pp. 56–60.


---

# 6. Kết quả thực nghiệm và thảo luận (Nhánh TF-IDF + K-Means)

## 6.1. Lựa chọn số cụm $K$ và kết quả đánh giá trên tập Test

Bảng 6.1 trình bày kết quả khảo sát lưới $K \in \{3, 5, 7, 9, 11, 14, 18\}$ trên tập Validation (1.391 bài báo).

**Bảng 6.1.** Kết quả khảo sát số cụm $K$ của mô hình TF-IDF + K-Means trên tập Validation.

| Số cụm $K$ | Silhouette | DBI | CH | Độ ổn định đa seed (ARI) | Cụm nhỏ nhất (bài) | Giảm Train Inertia | Purity | NMI | ARI |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 3 | 0,0147 | 7,149 | 16,65 | 0,849 ± 0,171 | 218 | — | 0,510 | 0,479 | 0,255 |
| 5 | 0,0189 | 7,194 | 14,19 | 0,917 ± 0,089 | 163 | 87,5 | 0,724 | 0,657 | 0,561 |
| 7 | 0,0205 | 6,618 | 12,48 | 0,828 ± 0,116 | 90 | 63,6 | 0,821 | 0,689 | 0,624 |
| **9** | **0,0190** | **6,157** | **11,37** | **0,822 ± 0,044** | **76** | **53,6** | **0,873** | **0,700** | **0,593** |
| 11 | 0,0205 | 5,826 | 10,59 | 0,806 ± 0,051 | 30 | 47,2 | 0,825 | 0,646 | 0,490 |
| 14 | 0,0194 | 5,629 | 9,68 | 0,699 ± 0,083 | 30 | 49,3 | 0,838 | 0,627 | 0,438 |
| 18 | 0,0191 | 5,603 | 8,77 | 0,713 ± 0,069 | 7 | 49,3 | 0,881 | 0,623 | 0,391 |

Số liệu thực nghiệm ở Bảng 6.1 cho thấy ba đặc điểm hình học điển hình của bài toán phân cụm văn bản trên không gian TF-IDF 20.000 chiều. Thứ nhất, hệ số Silhouette hầu như phẳng trên toàn dải $K$ (dao động hẹp từ 0,0147 đến 0,0205). Nguyên nhân là trên mặt cầu đơn vị nhiều chiều với độ thưa hơn 99%, khoảng cách từ một bài báo tới tâm cụm gần nhất chỉ nhỏ hơn đôi chút so với khoảng cách tới các tâm cụm khác (khoảng cách trung bình tới tâm trên toàn tập đạt $0{,}948 \pm 0{,}028$). Thứ hai, chỉ số DBI giảm đơn điệu theo $K$ (từ 7,194 xuống 5,603), trong khi mức giảm Inertia trên tập Train không tạo ra điểm gãy khuỷu tay (*elbow*) rõ rệt. Thứ ba, ở các giá trị $K$ lớn ($K = 14$ và $K = 18$), độ ổn định qua 5 hạt giống ngẫu nhiên sụt giảm mạnh (ARI trung bình rơi xuống 0,699 và 0,713), đồng thời xuất hiện cụm suy biến chỉ gồm 7 bài báo tại $K = 18$.

Khi đối chiếu ba cấu hình ứng viên ổn định nhất là $K \in \{7, 9, 11\}$, cấu hình **$K = 9$** được chọn làm mô hình tối ưu vì sự cân bằng giữa độ ổn định, kích thước cụm và khả năng phân tách ngữ nghĩa:
- **So với $K = 7$:** Mặc dù ngữ liệu có 7 chuyên mục gốc, hai chuyên mục Kinh doanh và Thể thao có quy mô lớn gấp đôi các chuyên mục khác và chứa các nhánh sự kiện rất khác biệt. Ở $K = 7$, mảng Khoa học công nghệ bị trộn lẫn vào cụm Doanh nghiệp — Đầu tư. Khi tăng lên $K = 9$, mô hình tách riêng được một cụm Công nghệ độc lập (Cụm 2: thiết bị Apple và AI, độ tinh khiết 92,1%), đồng thời tách mảng Thể thao thành hai cụm rõ rệt là Bóng đá (Cụm 4) và Đại hội thể thao ASIAD đa môn (Cụm 1), đồng thời tách mảng Kinh doanh thành Thị trường tài chính — Giá vàng (Cụm 8) và Doanh nghiệp vĩ mô (Cụm 6).
- **So với $K = 11$:** Cấu hình $K = 9$ có độ ổn định đa hạt giống cao hơn và phương sai nhỏ hơn ($0{,}822 \pm 0{,}044$ so với $0{,}806 \pm 0{,}051$), kích thước cụm nhỏ nhất đạt 76 bài (so với cụm quá nhỏ chỉ 30 bài ở $K = 11$), đồng thời đạt giá trị cao nhất trên cả Purity (0,873) và NMI (0,700).

**[HÌNH/BẢNG: Hình 6.1 — Biểu đồ đường cong Inertia, các chỉ số nội tại và độ ổn định đa hạt giống theo số cụm $K$.]**

Bảng 6.2 tổng hợp cấu trúc chi tiết và kết quả chấm điểm độ mạch lạc ngữ nghĩa thủ công (thang điểm 1–5) của 9 cụm ở cấu hình $K = 9$.

**Bảng 6.2.** Cấu trúc 9 cụm chủ đề và đánh giá định tính của mô hình TF-IDF + K-Means ($K = 9$).

| Cụm | Số bài (Train / Val) | Chuyên mục chiếm ưu thế (Val) | Từ khóa đại diện hàng đầu (Top keywords) | Nhãn chủ đề diễn giải | Điểm mạch lạc (1–5) |
| :---: | :---: | :--- | :--- | :--- | :---: |
| 0 | 1.135 / 251 | Giải trí (68,1%) | `cô, phim, anh, khán_giả, không, diễn_viên, nghệ_sĩ, tác_phẩm` | Giải trí, nghệ sĩ và đời sống | 3 |
| 1 | 395 / 98 | Thể thao (99,0%) | `asiad, nữ, u23, nam, giành, thi_đấu, việt_nam, trận, môn` | Thể thao: ASIAD và đội tuyển đa môn | 4 |
| 2 | 377 / 76 | KH-CN (92,1%) | `iphone, apple, pro, máy, dùng, màn_hình, max, hệ_thống, thiết_bị` | Công nghệ: Thiết bị Apple và AI | 4 |
| 3 | 722 / 159 | Sức khỏe (98,7%) | `bác_sĩ, bệnh, điều_trị, bệnh_viện, người_bệnh, khám, máu, thuốc` | Y tế, bệnh lý và điều trị | 5 |
| 4 | 832 / 174 | Thể thao (100,0%) | `trận, đội, bóng, cầu_thủ, hlv, cup, thắng, sân, fifa, bàn` | Bóng đá trong nước và quốc tế | 5 |
| 5 | 882 / 188 | Pháp luật (87,2%) | `công_an, điều_tra, án, vụ, cơ_quan, hành_vi, tiền, cảnh_sát` | Pháp luật hình sự và điều tra | 4 |
| 6 | 1.109 / 229 | Kinh doanh (74,7%) | `doanh_nghiệp, phát_triển, đầu_tư, công_nghệ, dự_án, thị_trường` | Doanh nghiệp, đầu tư và dịch vụ | 2 |
| 7 | 650 / 129 | Giáo dục (97,7%) | `trường, học_sinh, giáo_dục, học, đào_tạo, đại_học, nhà_trường, lớp` | Giáo dục và đào tạo | 5 |
| 8 | 391 / 87 | Kinh doanh (96,6%) | `tăng, giá, đồng, lãi_suất, thị_trường, giảm, usd, cổ_phiếu, bán` | Giá vàng, lãi suất và chứng khoán | 5 |

Điểm mạch lạc trung bình trên 9 cụm đạt **4,11 / 5,00**. Trong đó, 7/9 cụm có ranh giới từ vựng rất sắc nét và độ tinh khiết từ 87,2% đến 100%. Hai cụm có điểm mạch lạc thấp hơn là Cụm 6 (2/5 điểm, độ tinh khiết 74,7%) và Cụm 0 (3/5 điểm, độ tinh khiết 68,1%). Phân tích các bài viết ngẫu nhiên trong Cụm 6 cho thấy đây là một "cụm thu gom" (*catch-all cluster*): các bài báo viết về chuyển đổi số, hạ tầng hàng không, du học hay bán dẫn đều sử dụng chung lớp từ vựng kinh tế vĩ mô (`doanh_nghiệp`, `phát_triển`, `đầu_tư`, `dự_án`), nên bị kéo về cùng một tâm cụm. Tương tự, Cụm 0 thu hút các bài viết giàu đại từ nhân xưng và tính tự sự (`cô`, `anh`, `tôi`), khiến một số bài viết về phong cách sống hoặc bài quảng cáo thương hiệu bị gộp chung với tin tức nghệ sĩ.

**[HÌNH/BẢNG: Hình 6.2 — Phân bố kích thước 9 cụm và biểu đồ chiếu giảm chiều 2D của các cụm K-Means ($K = 9$).]**

Sau khi khóa cố định cấu hình $K = 9$ (`seed = 42`), mô hình được đánh giá một lần duy nhất trên tập Test độc lập (1.390 bài báo). Kết quả đối chiếu giữa tập Validation và tập Test được trình bày tại Bảng 6.3.

**Bảng 6.3.** Kết quả đánh giá mô hình TF-IDF + K-Means ($K = 9$) trên tập Validation và tập Test khóa kín.

| Tập dữ liệu | Số bài | Silhouette | DBI | CH | Purity | NMI | ARI | Topic Diversity | Tổng thời gian chạy (giây) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Validation | 1.391 | 0,0190 | 6,1566 | 11,3745 | 0,8728 | 0,7004 | 0,5925 | 0,9556 | — |
| **Test (Final)** | **1.390** | **0,0201** | **6,1954** | **11,5909** | **0,8597** | **0,6774** | **0,5649** | **0,9556** | **7,75** |

Trên tập Test, mô hình duy trì hiệu năng rất sát với tập Validation, đạt Purity = **85,97%**, NMI = **0,6774**, ARI = **0,5649** và độ đa dạng từ khóa Topic Diversity = **0,9556**. Tổng thời gian thực thi toàn trình chỉ mất **7,75 giây** (gồm 3,91 giây trích xuất đặc trưng TF-IDF, 2,84 giây huấn luyện K-Means, 0,0044 giây suy luận trên tập Test và 0,99 giây tính toán chỉ số). Kết quả này khẳng định mô hình có khả năng khái quát hóa tốt và không xảy ra hiện tượng quá khớp (*overfitting*).

---

## 6.2. Nghiên cứu cắt giảm tiền xử lý (Trả lời câu hỏi nghiên cứu RQ2)

Để định lượng đóng góp của từng bước tiền xử lý tiếng Việt, Bảng 6.4 trình bày kết quả thực nghiệm khi thay đổi lần lượt 4 biến thể `P0`, `P1`, `P2` và `P3` trên cùng cấu hình K-Means ($K = 9$).

**Bảng 6.4.** Ảnh hưởng của các cấp độ tiền xử lý từ `P0` đến `P3` đến chất lượng phân cụm K-Means ($K = 9$, tập Validation).

| Chỉ số đánh giá | `P0` (Tối thiểu) | `P1` (Chuẩn hóa) | `P2` (Tách từ) | **`P3` (Lọc từ dừng)** |
| :--- | :---: | :---: | :---: | :---: |
| Kích thước từ vựng huấn luyện | 17.983 | 15.285 | 20.000 | **20.000** |
| Độ thưa ma trận Validation (%) | 98,34% | 98,17% | 98,79% | **99,05%** |
| Tỷ lệ từ ngoài từ điển — OOV (%) | 9,86% | 11,04% | 11,97% | **2,56%** |
| Số từ kích hoạt trung bình / bài | 299,27 | 279,88 | 241,80 | **190,75** |
| Silhouette | 0,0193 | 0,0186 | 0,0181 | **0,0190** |
| Davies-Bouldin (DBI) | 5,7112 | **5,5708** | 6,1151 | 6,1566 |
| Calinski-Harabasz (CH) | 13,15 | **14,17** | 11,50 | 11,38 |
| **Purity (%)** | 83,61% | 81,52% | 81,60% | **87,28%** |
| **NMI** | 0,6652 | 0,6503 | 0,6494 | **0,7004** |
| **ARI** | 0,5457 | 0,5161 | 0,4958 | **0,5925** |
| Độ ổn định đa seed (ARI) | 0,8599 ± 0,0487 | 0,8267 ± 0,0612 | 0,8186 ± 0,0478 | **0,8211 ± 0,0442** |
| **Tỷ lệ từ ghép trong Top-15 (%)** | 0,00% | 0,00% | 32,59% | **43,70%** |
| **Tỷ lệ nhiễu từ dừng trong Top-15 (%)** | 9,63% | 9,63% | 18,52% | **0,00%** |
| Topic Diversity (Top-10) | 0,9000 | 0,8778 | 0,9222 | **0,9556** |

Kết quả thực nghiệm ở Bảng 6.4 làm sáng tỏ hai cơ chế ngôn ngữ học quan trọng khi xử lý văn bản tiếng Việt:

1. **Nghịch lý giữa chỉ số hình học nội tại và chất lượng ngữ nghĩa ngoại tại:** Nếu chỉ quan sát các chỉ số hình học nội tại, biến thể `P1` đạt điểm DBI thấp nhất (5,5708) và CH cao nhất (14,17), trong khi biến thể hoàn chỉnh `P3` lại có DBI cao hơn (6,1566). Nguyên nhân là ở `P0` và `P1`, văn bản chưa được tách từ và chưa lọc từ dừng, khiến các âm tiết đơn xuất hiện dày đặc ở mọi bài báo (như `các`, `của`, `người`, `và`, `được`, `trong`). Các âm tiết chung này đóng vai trò như một "lực hút giả", kéo vector của mọi bài viết co cụm lại gần nhau quanh một hướng chung, làm giảm khoảng cách hình học nội cụm một cách cơ học. Tuy nhiên, các cụm thu được lại bị nhiễu nặng bởi các âm tiết vô nghĩa (`0%` từ ghép). Khi sang `P3`, toàn bộ 388 từ dừng bị loại bỏ, giải phóng các vector bài báo tỏa ra đúng các trục thuật ngữ chuyên ngành. Dù các cụm giãn rộng hơn đôi chút về mặt hình học, độ chính xác theo chủ đề thực tế lại tăng vọt lên mức cao nhất trên mọi thước đo: Purity tăng từ 81,52% lên **87,28%** (+5,76 điểm phần trăm), NMI tăng từ 0,6503 lên **0,7004**, và ARI tăng từ 0,5161 lên **0,5925**.
2. **Sự cộng hưởng bắt buộc giữa tách từ (`P2`) và lọc từ dừng (`P3`):** Nếu chỉ tách từ bằng `pyvi` mà không lọc từ dừng (`P2`), hiệu năng phân cụm gần như giậm chân tại chỗ (Purity đạt 81,60%, ARI giảm nhẹ xuống 0,4958) và tỷ lệ nhiễu từ dừng trong Top-15 từ khóa tăng vọt lên 18,52% (25/135 từ khóa đại diện là từ dừng ghép như `nhưng_không`, `cho_biết`, `có_thể`). Lý do là khi các âm tiết nội dung được gom lại thành từ ghép (`bệnh_viện`, `học_sinh`), tần suất xuất hiện của chúng giảm xuống so với các từ dừng đơn âm tiết, khiến từ dừng càng chiếm ưu thế trong trọng số tâm cụm. Chỉ khi kết hợp tách từ với lọc từ dừng ở `P3`, các từ dừng mới bị triệt tiêu hoàn toàn (**0,00% nhiễu**), giúp tỷ lệ từ ghép có nghĩa trong tâm cụm nhảy vọt lên **43,70%** và giảm tỷ lệ OOV trên tập Validation từ 11,97% xuống chỉ còn **2,56%**. Kiểm tra độ nhạy tại $K = 7$ cũng cho kết quả hoàn toàn nhất quán (`P3` đạt Purity = 82,10% và NMI = 0,6889, vượt xa `P1` và `P2`).

**[HÌNH/BẢNG: Hình 6.3 — So sánh các chỉ số ngoại tại (Purity, NMI, ARI) và chất lượng từ vựng tâm cụm qua 4 biến thể tiền xử lý P0–P3.]**

---

## 6.3. Nghiên cứu phạm vi đầu vào: chỉ tiêu đề so với tiêu đề + nội dung (Trả lời RQ3)

Bảng 6.5 đối chiếu trực tiếp hiệu năng của mô hình K-Means ($K = 9$ và kiểm tra độ nhạy tại $K = 7$) khi chỉ sử dụng tiêu đề (`Title-only`) so với khi kết hợp tiêu đề và toàn văn (`Title+Content`) trên nền tiền xử lý `P3`.

**Bảng 6.5.** So sánh thực nghiệm giữa biến thể chỉ tiêu đề (`Title-only`) và tiêu đề kết hợp nội dung (`Title+Content`) trên tập Validation.

| Chỉ số đánh giá | `Title-only` ($K = 9$) | `Title+Content` ($K = 9$) | `Title-only` ($K = 7$) | `Title+Content` ($K = 7$) |
| :--- | :---: | :---: | :---: | :---: |
| Kích thước từ vựng hiệu dụng | 4.437 | **20.000** | 4.437 | **20.000** |
| Số từ kích hoạt trung bình / bài | 7,66 | **190,75** | 7,66 | **190,75** |
| Độ thưa ma trận Validation (%) | 99,83% | **99,05%** | 99,83% | **99,05%** |
| Tỷ lệ OOV tập Validation (%) | 10,45% | **2,56%** | 10,45% | **2,56%** |
| Silhouette | 0,0063 | **0,0190** | 0,0053 | **0,0205** |
| Davies-Bouldin (DBI) | 5,8107 | 6,1566 | 6,7098 | **6,6178** |
| **Purity (%)** | 30,77% | **87,28%** | 29,55% | **82,10%** |
| **NMI** | 0,1637 | **0,7004** | 0,1047 | **0,6889** |
| **ARI** | -0,0016 | **0,5925** | -0,0046 | **0,6239** |
| Độ ổn định đa seed (ARI) | 0,5496 ± 0,0894 | **0,8211 ± 0,0442** | 0,5983 ± 0,1139 | **0,8277 ± 0,1158** |
| Kích thước cụm (Nhỏ nhất / Lớn nhất) | 13 / **982** | **76 / 251** | 12 / **1.092** | **90 / 297** |
| Tỷ lệ từ ghép trong Top-15 (%) | 17,04% | **43,70%** | 19,05% | **42,86%** |
| Topic Diversity (Top-10) | 0,7556 | **0,9556** | 0,7429 | **0,9286** |
| Tổng thời gian chạy thực nghiệm (giây) | **2,27** | 25,38 | **1,85** | 21,40 |

Số liệu tại Bảng 6.5 chứng minh hiện tượng **sụp đổ cụm (*cluster collapse*)** nghiêm trọng khi áp dụng K-Means trên văn bản tiêu đề ngắn. Do mỗi tiêu đề sau khi lọc từ dừng chỉ còn trung bình 7,66 từ khóa khác không trên không gian 4.437 chiều (độ thưa lên tới 99,83% và tỷ lệ OOV tới 10,45%), phần lớn các cặp tiêu đề không chia sẻ chung từ khóa nào, khiến tích vô hướng giữa chúng bằng 0 ($\cos(\mathbf{x}_i, \mathbf{x}_j) = 0$) và khoảng cách Euclid giữa hầu hết các điểm đều xấp xỉ $\sqrt{2}$. Khi thiếu sự giao thoa từ vựng để định hình các tâm cụm chuyên biệt, thuật toán K-Means đẩy tới **982 / 1.391 bài báo (70,60% toàn tập Validation) dồn vào một cụm khổng lồ duy nhất** (tại $K = 7$, cụm khổng lồ này hút tới 1.092 bài, tương đương 78,50%). Hệ quả là chỉ số ARI của `Title-only` rơi xuống mức âm nhẹ (**-0,0016** tại $K = 9$ và **-0,0046** tại $K = 7$), tức là không tốt hơn việc gán nhãn ngẫu nhiên, còn Purity giảm sâu xuống 30,77%.

Ngược lại, việc bổ sung phần nội dung bài báo trong `Title+Content` nâng số từ kích hoạt trung bình lên 190,75 từ/bài (tăng gấp 24,9 lần), tạo ra mạng lưới đồng xuất hiện từ vựng đủ dày để neo giữ các tâm cụm. Nhờ đó, Purity tăng vọt thêm **+56,51 điểm phần trăm** (từ 30,77% lên 87,28%), NMI tăng từ 0,1637 lên 0,7004 và ARI tăng từ -0,0016 lên 0,5925. Xét theo từng chuyên mục, độ tinh khiết của mảng Thể thao tăng từ 14,26% lên 100%, Sức khỏe tăng từ 16,19% lên 98,74%, và Khoa học công nghệ tăng từ 7,03% lên 92,11%. Dù biến thể toàn văn tốn thời gian tính toán nhiều hơn so với chỉ dùng tiêu đề (25,38 giây so với 2,27 giây cho toàn bộ quy trình đánh giá đa hạt giống), mức chênh lệch vài chục giây là hoàn toàn không đáng kể so với sự cải thiện quyết định về chất lượng phân cụm.

**[HÌNH/BẢNG: Hình 6.4 — Đối chiếu hiệu năng phân cụm và sự suy sụp phân bố kích thước cụm giữa `Title-only` và `Title+Content`.]**

---

## 6.4. Thảo luận về ưu điểm và hạn chế của nhánh TF-IDF + K-Means

Từ toàn bộ kết quả thực nghiệm, có thể rút ra những nhận định khách quan về vai trò của nhánh mô hình cổ điển TF-IDF + K-Means trong bài toán khám phá chủ đề báo chí tiếng Việt:

- **Ưu điểm nổi bật:** Khi được kết hợp với quy trình tách từ tiếng Việt và lọc từ dừng hợp lý (`P3`) trên văn bản toàn văn (`Title+Content`), TF-IDF + K-Means là một đường cơ sở (*baseline*) cực kỳ mạnh mẽ, tốc độ cao (huấn luyện và suy luận chỉ mất vài giây trên CPU) và có độ tinh khiết ngoại tại rất cao (Test Purity đạt 85,97%, Topic Diversity đạt 0,9556). Trọng số tâm cụm phản ánh trực diện các thuật ngữ đặc trưng của từng lĩnh vực mà không đòi hỏi tài nguyên GPU.
- **Hạn chế cố hữu:**
  1. *Cơ chế phân cụm cứng (*hard assignment*):* Mỗi bài báo bị ép buộc gán vào một cụm duy nhất. Với các bài báo giao thoa nhiều lĩnh vực (ví dụ: chính sách thu phí học đường của ngành y, hoặc vụ án lừa đảo tài chính ngân hàng), mô hình không thể biểu diễn tỷ trọng đa chủ đề như LDA hay NMF, dẫn đến sự hình thành của cụm thu gom chung chung (như Cụm 6).
  2. *Hạn chế của biểu diễn túi từ:* TF-IDF dựa hoàn toàn vào sự trùng khớp bề mặt từ vựng nên không tự nhận diện được các từ đồng nghĩa hoặc cách diễn đạt khác nhau của cùng một khái niệm như các mô hình nhúng ngữ cảnh (PhoBERT/SBERT), khiến phương pháp này hoàn toàn thất bại trên văn bản cực ngắn (`Title-only`).
  3. *Tính nhạy cảm với sự kiện ngắn hạn:* Do ngữ liệu thu thập trong tháng 09/2026, K-Means dễ bị thu hút bởi các từ khóa sự kiện có tần suất đột biến trong tháng (như `asiad`, `iphone`), tạo ra các cụm sự kiện bên cạnh các cụm chủ đề bền vững.

---

# Tài liệu tham khảo (Phần của TV1)

[1] M. Steinbach, G. Karypis, and V. Kumar, "A comparison of document clustering techniques," in *Proc. KDD Workshop on Text Mining*, Boston, MA, USA, 2000, pp. 1–20.

[2] T. Vu, D. Q. Nguyen, D. Q. Nguyen, M. Dras, and M. Johnson, "VnCoreNLP: A Vietnamese natural language processing toolkit," in *Proc. 2018 Conf. North American Chapter of the Association for Computational Linguistics: Demonstrations (NAACL-HLT)*, New Orleans, LA, USA, 2018, pp. 56–60.

