# Paper notes - Lee & Seung (1999): Learning the parts of objects by non-negative matrix factorization

## 1. Người cung cấp

- Thành viên: TV2 - Tiến.
- Phương pháp phụ trách: NMF/topic modeling.
- Ngày đọc/đối chiếu nguồn: 21/10/2026.

## 2. Citation info - Thông tin trích dẫn

- Citation key: `lee_seung_1999_nmf`.
- Tên đầy đủ: *Learning the parts of objects by non-negative matrix factorization*.
- Tác giả: Daniel D. Lee và H. Sebastian Seung.
- Năm xuất bản: 1999.
- Tạp chí: *Nature*, volume 401, trang 788-791.
- DOI: <https://doi.org/10.1038/44565>.
- Trang nhà xuất bản: <https://www.nature.com/articles/44565>.
- PDF bài gốc: <https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/lee-nature-99.pdf>.
- Loại tài liệu: Scientific paper - journal article.

**Quy ước bằng chứng:** Trang in *Nature* 788-791 tương ứng trang PDF 1-4. Bài dạng letter không đánh số section; dùng tên mục, số công thức và hình để định vị. Đầu trang 788 còn chứa phần cuối của bài trước.

## 3. Method - Phương pháp

- **Bài toán và mục tiêu:** Học biểu diễn bằng cách kết hợp các bộ phận, minh họa trên khuôn mặt và ngữ nghĩa văn bản (trang 788; Figures 1, 4).
- **Representation:** V là ma trận không âm kích thước n × m, mỗi cột biểu diễn một ảnh hoặc tài liệu; n là số pixel hoặc số từ, m là số đối tượng. W là ma trận không âm kích thước n × r, mỗi cột là một basis; H là ma trận không âm kích thước r × m, mỗi cột là một encoding. Theo Eq. (1), trang 788:

> V(i, μ) ≈ (WH)(i, μ) = Σ W(i, a) × H(a, μ), với a chạy từ 1 đến r.
>
> Ràng buộc: W(i, a) ≥ 0 và H(a, μ) ≥ 0.

- **Cơ chế:** Tính không âm cho phép cộng nhiều basis, không cho phép triệt tiêu bằng phép trừ. Basis và encoding trong thí nghiệm có nhiều hệ số bằng không, tạo biểu diễn thưa. Không suy rộng quan sát này thành bảo đảm học được bộ phận có nghĩa với mọi dữ liệu (trang 789-790).

**Hàm mục tiêu:** Bài năm 1999 cực đại hóa Eq. (2), trang 789:

> F(W, H) = Σ Σ [V(i, μ) × log((WH)(i, μ)) − (WH)(i, μ)].
>
> Hai tổng lần lượt chạy qua i từ 1 đến n và μ từ 1 đến m.

Theo mô hình quan sát Poisson với trung bình (WH)(i, μ), đây là log-likelihood bỏ các hạng chỉ phụ thuộc dữ liệu. F lớn hơn biểu thị độ khớp tốt hơn theo tiêu chuẩn này. Bài có thảo luận squared error như một lựa chọn khác ở trang 789, nhưng đó không phải mục tiêu dùng trong Figure 2.

**Tối ưu/cập nhật:** Figure 2, trang 790, cho ba bước lặp theo thứ tự sau, từ khởi tạo không âm:

1. Cập nhật basis: W(i, a) ← W(i, a) × Σ [V(i, μ) × H(a, μ) / (WH)(i, μ)], lấy tổng theo μ.
2. Chuẩn hóa cột: W(i, a) ← W(i, a) / Σ W(j, a), lấy tổng theo j.
3. Cập nhật encoding: H(a, μ) ← H(a, μ) × Σ [W(i, a) × V(i, μ) / (WH)(i, μ)], lấy tổng theo i.

Cập nhật bảo toàn tính không âm; chuẩn hóa cột W về tổng 1 loại bất định tỉ lệ giữa hai factor. Tác giả mô tả hội tụ đơn điệu đến cực đại **cục bộ**, không bảo đảm cực đại toàn cục. Các quy tắc trong Figure 2 không có số phương trình riêng.

| Phương pháp | Ràng buộc/cơ chế | Ý nghĩa trong thí nghiệm |
|---|---|---|
| VQ | Mỗi cột H có một phần tử bằng 1, còn lại bằng 0; triển khai bằng k-means | Một prototype cho mỗi đối tượng; basis mang tính toàn thể |
| PCA | Cột W trực chuẩn, hàng H trực giao; hệ số có thể âm hoặc dương | Eigenfaces kết hợp bằng cộng và trừ; từng basis khó diễn giải trực quan |
| NMF | W và H đều không âm; nhiều basis có thể cùng hoạt động | Kết hợp cộng tính; học các vùng bộ phận trên ảnh đã căn chỉnh |

Bằng chứng so sánh: trang 788-789; Figure 1, trang 789; Methods, trang 791.

## 4. Dataset - Dữ liệu

| Thí nghiệm | Nguồn và quy mô | Biểu diễn/cấu hình | Bằng chứng |
|---|---|---|---|
| Khuôn mặt | **2.429 ảnh** chính diện; ảnh do C. Papageorgiou và T. Poggio cung cấp, không nêu tên benchmark riêng | **19 × 19 pixel** (361 chiều); **49 basis** cho cả ba phương pháp | Figure 1, trang 789; Methods và Acknowledgements, trang 791 |
| Văn bản | **Grolier encyclopedia**, **30.991 bài viết** tiếng Anh | **15.276 từ**, ma trận đếm từ 15.276 × 30.991; **200 đặc trưng ngữ nghĩa** | Văn bản trang 790; Figure 4 và Methods, trang 791 |

- **Tiền xử lý ảnh/protocol:** Căn chỉnh thủ công trên lưới 19 × 19; biến đổi tuyến tính mức xám để mean và standard deviation đều bằng 0,25, rồi cắt về [0,1]. NMF khởi tạo ngẫu nhiên, gần hội tụ sau dưới 50 vòng; hình báo cáo dùng 500 vòng. PCA lấy 49 eigenvectors có eigenvalues lớn nhất của VVᵀ; VQ dùng k-means với khởi tạo ngẫu nhiên (Methods, trang 791).
- **Tiền xử lý văn bản/protocol:** Bỏ 430 từ phổ biến nhất, chọn 15.276 từ có tần suất cao nhất còn lại. Dùng **word counts**, không phải TF-IDF. Kết quả sau 50 vòng từ khởi tạo ngẫu nhiên. Figure 4 biểu diễn mỗi feature được minh họa bằng tám từ có trọng số cao nhất (Methods và Figure 4, trang 791).
- **Kiểm tra tên/quy mô:** Tên Grolier có trong Methods/Acknowledgements; số 30.991 có trong chú thích Figure 4. Không ghi “không được báo cáo” cho hai thông tin này. Phiên bản/năm phát hành cụ thể của corpus: **Không được báo cáo**.
- **Chia dữ liệu:** Không được báo cáo Train/Validation/Test hoặc đánh giá held-out. Hai thí nghiệm phân rã dữ liệu để phân tích biểu diễn; không phải benchmark phân loại khuôn mặt hoặc phân cụm có nhãn.

## 5. Metric - Đánh giá

- **Tiêu chuẩn tối ưu định lượng:** Eq. (2) đánh giá độ khớp/tái tạo trong huấn luyện; chiều tốt là F lớn hơn. Bài không công bố bảng giá trị F hay reconstruction error để xếp hạng NMF/PCA/VQ. Số vòng lặp là thông tin hội tụ, không phải điểm chất lượng topic.
- **Kết quả ảnh chủ yếu định tính:** Figure 1 so sánh basis, encoding và ảnh tái tạo với cùng 49 thành phần. NMF tạo basis cục bộ giống bộ phận khuôn mặt, VQ/PCA thiên về biểu diễn toàn thể. Không chuyển nhận xét quan sát hình thành accuracy nhận diện hoặc kiểm định thống kê (trang 789).
- **Kết quả văn bản chủ yếu định tính:** Figure 4 minh họa các nhóm từ liên quan và cách phối hợp feature để biểu diễn bài về Hiến pháp Hoa Kỳ. Hai feature chứa “lead” minh họa hai cách dùng từ theo ngữ cảnh. Đây không phải điểm word-sense disambiguation trên tập kiểm tra (trang 790-791).
- **Metric không được báo cáo:** ARI, NMI, NPMI/topic coherence, accuracy phân loại hoặc human-rating có protocol và thang điểm. Không tự thêm các chỉ số này vào kết quả bài gốc.
- **Điều kiện/phạm vi:** Kết quả phụ thuộc hai dữ liệu và cấu hình ở mục 4. Bài không so sánh NMF với LDA và không đánh giá tin tức tiếng Việt.

## 6. Gap - Hạn chế hoặc khoảng trống

**Tác giả nêu trực tiếp** (đoạn bắt đầu “Although NMF is successful…”, trang 790):

- Thành công trên khuôn mặt và văn bản không bảo đảm học được bộ phận từ mọi dữ liệu, chẳng hạn ảnh có góc nhìn rất khác nhau hoặc đối tượng có cấu hình phức tạp.
- Những trường hợp đó có thể cần mô hình phân cấp nhiều tầng biến ẩn; tính không âm tự nó chưa được khẳng định là đủ.
- NMF không học quan hệ “cú pháp” giữa các bộ phận; ngoài tính không âm, không đặt thêm giả định về phụ thuộc thống kê giữa các biến ẩn.
- Ngoài ra, thuật toán chỉ được mô tả hội tụ đến cực đại cục bộ (trang 789; Figure 2, trang 790).

**Nhận xét riêng của TV2 cho đồ án:** Bài chưa khảo sát tin tức tiếng Việt, phân đoạn từ, TF-IDF P3, chọn số topic bằng coherence/diversity hay độ ổn định nhiều seed. Nhóm cần tự kiểm chứng các khía cạnh này. Đây là giới hạn phạm vi của **bài này**, không phải tuyên bố chưa có nghiên cứu khác. Tính thưa và khả năng diễn giải quan sát được không thay thế đánh giá trên dữ liệu nhóm.

## 7. Liên hệ với đồ án

- **M04 - NMF pipeline:** Cơ sở cho phân rã dữ liệu không âm. Đồ án dùng document-term TF-IDF, nên W trong code có thể là document-topic và H là topic-word, ngược vai trò ký hiệu term-document trong bài gốc.
- **M05 - Thực nghiệm/đánh giá:** Khảo sát số topic, từ khóa và bài đại diện; mô tả loss, solver và initialization theo cấu hình thực tế. Không mặc nhiên gán thuật toán Figure 2 cho triển khai dùng squared-error NMF.
- **R03/R06 - Viết bài:** Dùng làm nguồn nền tảng về biểu diễn cộng tính, kết hợp O’Callaghan cho coherence. Kết quả nhóm trên tin tức tiếng Việt sau preprocessing P3 phải được báo cáo riêng.

Đây là hướng kế thừa dự kiến, không phải kết quả đồ án đã được xác nhận.

## 8. Vị trí bằng chứng

| Nhận định/số liệu | Section/trang/bảng/vị trí nguồn |
|---|---|
| Thông tin thư mục, mục tiêu | Đầu bài, trang 788; DOI/trang Nature |
| V ≈ WH, kích thước và vai trò các factor | Trang 788, Eq. (1) |
| VQ unary; PCA trực giao/cho phép âm; NMF cộng tính | Trang 788-789; Figure 1, trang 789 |
| Hàm mục tiêu, mô hình Poisson; lựa chọn squared error | Trang 789, Eq. (2) và các đoạn ngay sau |
| Ba bước cập nhật, chuẩn hóa cột W, cực đại cục bộ | Figure 2 và chú thích, trang 790 |
| 2.429 ảnh, 19 × 19 pixel, 49 basis | Figure 1 và chú thích, trang 789 |
| Căn chỉnh/chuẩn hóa ảnh; 500 vòng; PCA và k-means | Methods, trang 791 |
| 200 semantic features, ví dụ phối hợp và đa nghĩa | Trang 790; Figure 4, trang 791 |
| Grolier, 30.991 bài, 15.276 từ, tám từ/feature | Figure 4 và chú thích, trang 791 |
| Bỏ 430 từ phổ biến; 50 vòng cho văn bản | Methods, trang 791 |
| Nguồn cung cấp ảnh và corpus Grolier | Acknowledgements, trang 791 |
| Giới hạn học bộ phận, mô hình nhiều tầng, quan hệ cú pháp | Đoạn “Although NMF is successful…”, trang 790 |
| Không báo cáo bảng metric nhận diện/clustering/coherence | Phạm vi toàn bài, trang 788-791; Figures 1-4 và Methods |

## 9. Điều cần xác minh thêm

- Đối chiếu loss, solver, normalization và cách sắp trục trong code/config của nhóm trước khi viết Methodology.
- Nếu cần phiên bản Grolier hoặc tên benchmark ảnh, phải tìm nguồn bổ sung; bài không báo cáo các chi tiết này. Không suy đoán từ số ảnh.
- Không dùng tính thưa trong Figure 1 như định lý bảo đảm sparsity/interpretability cho mọi cấu hình.

Thông tin phương pháp, quy mô hai thí nghiệm và vị trí bằng chứng đã được đối chiếu toàn văn, gồm chú thích Figure 4.

## 10. Trạng thái và bàn giao

- Trạng thái: `Ready for TV3 review`.
- Link file hoặc PR bàn giao: `paper/notes/tv2_lee_seung_1999_nmf.md`
- Nhận xét của TV3 sau review: Chưa có; chờ TV3 review và nghiệm thu vào Related Work Matrix.
