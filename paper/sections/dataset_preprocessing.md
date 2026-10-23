# 3. Tập dữ liệu và quy trình tiền xử lý

## 3.1. Thu thập và đặc tả ngữ liệu

Để phục vụ bài toán khám phá chủ đề không giám sát, nhóm nghiên cứu xây dựng bộ ngữ liệu báo chí tiếng Việt từ ba cổng thông tin điện tử lớn gồm Tuổi Trẻ, VnExpress và VietNamNet. Dữ liệu được thu thập tự động trong tháng 09/2026 nhằm đảm bảo tính đồng nhất về bối cảnh thời sự. Tập dữ liệu thô ban đầu gồm 9.304 bài báo. Mỗi bản ghi lưu trữ đầy đủ các thông tin gồm: mã bài viết, tiêu đề, nội dung toàn văn, thời gian xuất bản, nguồn báo, đường dẫn gốc và nhãn chuyên mục của tòa soạn.

Thống kê mô tả bằng EDA cho thấy tỷ lệ đóng góp tương đối đồng đều giữa ba nguồn tin. Cụ thể, Tuổi Trẻ chiếm 40,61% (3.778 bài), VnExpress chiếm 31,91% (2.969 bài) và VietNamNet chiếm 27,48% (2.557 bài). Ngữ liệu bao phủ 7 chuyên mục gồm Kinh doanh, Thể thao, Sức khỏe, Pháp luật, Giải trí, Giáo dục và Khoa học công nghệ. Trong đó, Kinh doanh là chuyên mục lớn nhất với 1.995 bài, còn Khoa học công nghệ có số lượng ít nhất với 702 bài, tương ứng tỷ lệ mất cân bằng ở mức vừa phải là $2{,}84$ lần. Độ dài bài viết có trung vị đạt 658 từ, thấp nhất là 50 từ đối với tin vắn và cao nhất lên tới 5.885 từ đối với các bài phóng sự dài.

Xuyên suốt nghiên cứu này, toàn bộ các mô hình gồm TF-IDF + K-Means, LDA, NMF và BERTopic đều được huấn luyện hoàn toàn không giám sát. Nhãn chuyên mục của tòa soạn không tham gia vào bất kỳ bước tiền xử lý, trích xuất đặc trưng hay huấn luyện nào. Nhãn này chỉ được dùng để chia tập dữ liệu theo phương pháp phân tầng và làm mốc tham chiếu khi tính các chỉ số đánh giá ngoại tại gồm Purity, NMI và ARI.

**[HÌNH/BẢNG: Hình 3.1 — Phân bố số lượng bài viết theo 7 chuyên mục và tỷ trọng của 3 nguồn báo.]**

---

## 3.2. Kiểm định chất lượng và khử trùng lặp

Quá trình kiểm tra dữ liệu thô cho thấy hai vấn đề phổ biến của văn bản báo mạng thu thập tự động. Vấn đề thứ nhất là nhiễu giao diện (*boilerplate*). Trong tập dữ liệu có 1.940 bài chứa các câu kêu gọi tương tác mạng xã hội ("chia sẻ"), 1.263 bài chứa cụm từ "theo dõi" tòa soạn, 941 bài lẫn chú thích ảnh hoặc video, và 276 bài dính nội dung quảng cáo. Ngoài ra còn có 180 bài chứa đường dẫn web thô và 62 bài chứa địa chỉ email. Vấn đề thứ hai là hiện tượng trùng lặp nội dung do các báo đăng lại tin từ hãng thông tấn hoặc cập nhật cùng một bản tin. Dù không có cặp bài viết nào trùng khớp hoàn toàn 100% nội dung, việc đối chiếu tiêu đề đã phát hiện 12 nhóm bài viết (gồm 25 bài) có tiêu đề trùng hoặc gần trùng nhau.

Để loại bỏ các bài trùng lặp mà không xóa nhầm những bài báo khai thác các góc nhìn khác nhau của cùng một sự kiện, nhóm nghiên cứu áp dụng quy trình khử trùng lặp hai giai đoạn. Ở giai đoạn đầu, văn bản sau chuẩn hóa được tách thành tập hợp các từ và đưa qua thuật toán MinHash LSH (128 hàm băm hoán vị, ngưỡng lọc $0{,}70$) kết hợp gom nhóm theo tiêu đề chuẩn hóa (Broder, 1997). Bước này giúp sàng lọc nhanh các cặp bài viết có khả năng giống nhau cao mà không cần so sánh chéo toàn bộ tập dữ liệu với độ phức tạp $\mathcal{O}(N^2)$. Ở giai đoạn hai, 2.122 cặp ứng viên được tính độ tương đồng Jaccard chính xác $J(A, B) = |A \cap B| / |A \cup B|$ riêng cho tập từ của tiêu đề ($S_{\text{title}}$) và tập từ của nội dung ($S_{\text{content}}$), kết hợp với chênh lệch ngày đăng $\Delta t = |t_i - t_j|$. Mỗi cặp bài viết $(d_i, d_j)$ không xung đột chuyên mục được phân loại theo quy tắc ngưỡng:

$$\text{Quyết định}(d_i, d_j) = \begin{cases} \text{Loại bỏ}, & \text{nếu } S_{\text{content}} \ge 0{,}85 \text{ hoặc } \left(S_{\text{title}} \ge 0{,}60 \land S_{\text{content}} \ge 0{,}70 \land \Delta t \le 2\right) \\ \text{Kiểm tra thủ công}, & \text{nếu } 0{,}80 \le S_{\text{content}} < 0{,}85 \\ \text{Giữ cả hai}, & \text{trong các trường hợp còn lại} \end{cases} \quad (3.1)$$

Các cặp thuộc diện loại bỏ được gom thành các cụm liên thông bằng cấu trúc Union-Find. Trong mỗi cụm trùng lặp, hệ thống giữ lại duy nhất bài viết có nội dung dài nhất; nếu độ dài bằng nhau sẽ ưu tiên bài có ngày đăng hợp lệ và mã bài viết đứng trước theo thứ tự từ điển. Kết quả xử lý trên 2.122 cặp ứng viên đã giữ lại 2.074 cặp phản ánh cùng sự kiện nhưng khác nội dung, giữ lại 18 cặp sau khi rà soát thủ công và loại bỏ 30 bài trùng lặp. Tập ngữ liệu sạch cuối cùng gồm **9.274 bài báo**.

**[HÌNH/BẢNG: Hình 3.2 — Phân phối độ tương đồng Jaccard của các cặp ứng viên và kết quả khử trùng lặp.]**

---

## 3.3. Chuỗi tiền xử lý đa mức độ cho tiếng Việt

Trong chữ viết tiếng Việt, khoảng trắng dùng để tách các âm tiết chứ không phải ranh giới từ. Bên cạnh đó, mỗi nhóm mô hình học máy lại đòi hỏi cách xử lý văn bản khác nhau. Các mô hình dựa trên túi từ như K-Means, LDA và NMF cần văn bản đã được ghép từ phức và lọc từ dừng để tránh nhiễu tần suất. Ngược lại, các mô hình dựa trên kiến trúc Transformer như PhoBERT hay SBERT trong nhánh BERTopic lại cần giữ nguyên câu văn tự nhiên, dấu câu và chữ viết hoa để nắm bắt ngữ cảnh.

Nhằm đáp ứng yêu cầu riêng của từng mô hình và đánh giá ảnh hưởng của việc tiền xử lý (câu hỏi nghiên cứu RQ2), nhóm xây dựng chuỗi tiền xử lý gồm 4 mức nối tiếp nhau từ `P0` đến `P3` trên toàn bộ 9.274 bài báo:

1. **Mức `P0` (Làm sạch tối thiểu):** Giải mã các ký tự HTML, xóa thẻ đánh dấu còn sót và chuẩn hóa bảng mã tiếng Việt về chuẩn **Unicode NFC**. Bước chuẩn hóa NFC giúp đồng nhất các ký tự tiếng Việt có dấu về cùng một cách biểu diễn byte. Toàn bộ chữ hoa và dấu câu được giữ nguyên để làm đầu vào cho BERTopic.
2. **Mức `P1` (Chuẩn hóa hình thái và lọc nhiễu báo chí):** Kế thừa từ `P0`, văn bản được đưa về chữ thường. Các đường dẫn web, địa chỉ email và con số được thay thế bằng các ký hiệu chung `<URL>`, `<EMAIL>`, `<NUM>`. Đồng thời, các câu rác kêu gọi chia sẻ, theo dõi hoặc chú thích ảnh ở cuối bài cũng được cắt bỏ bằng biểu thức chính quy.
3. **Mức `P2` (Tách từ tiếng Việt):** Sử dụng công cụ `pyvi` để tách từ trên văn bản `P1`. Theo quy ước chuẩn của cộng đồng xử lý ngôn ngữ tự nhiên tiếng Việt VLSP (Vu et al., 2018), các âm tiết tạo thành một từ ghép được nối lại bằng dấu gạch dưới `_` (ví dụ: `"kinh_doanh"`, `"trí_tuệ_nhân_tạo"`). Bước này giúp mô hình túi từ nhận diện đúng đơn vị từ vựng thay vì các âm tiết rời rạc.
4. **Mức `P3` (Lọc từ dừng):** Loại bỏ các từ hư, liên từ và từ xuất hiện quá phổ biến dựa trên danh sách 388 từ dừng tiếng Việt. Tuy nhiên, bộ lọc vẫn giữ lại các từ mang nghĩa phủ định (như `không`, `chưa`, `chẳng`) và các thuật ngữ chuyên ngành để tránh làm sai lệch ngữ nghĩa. Văn bản ở mức `P3` là đầu vào chuẩn cho nhánh TF-IDF + K-Means và NMF.

Bảng 3.1 tổng hợp sự thay đổi về kích thước từ vựng và độ dài bài viết qua bốn mức tiền xử lý.

**Bảng 3.1.** Thống kê từ vựng và độ dài văn bản qua 4 giai đoạn tiền xử lý ($N = 9.274$ bài báo).

| Biến thể | Số bài báo | Số bài rỗng | Kích thước từ vựng | Tổng số token | Tỷ lệ giữ lại token (%) | Độ dài TB (ký tự) | Số từ TB / bài | Số từ trung vị / bài | Khoảng phân vị P5 - P95 (từ) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Gốc (`original`) | 9.274 | 0 | 115.033 | 6.756.487 | 100,00% | 3.362,39 | 728,54 | 671,0 | 312,0 - 1.331,0 |
| `P0` | 9.274 | 0 | 113.175 | 6.756.482 | 100,00% | 3.361,24 | 728,54 | 671,0 | 312,0 - 1.331,0 |
| `P1` | 9.274 | 0 | 100.997 | 6.742.822 | 99,80% | 3.353,31 | 727,07 | 671,0 | 310,0 - 1.328,0 |
| `P2` | 9.274 | 0 | 61.108 | 5.989.915 | 88,65% | 3.444,85 | 645,88 | 593,0 | 284,0 - 1.174,0 |
| **`P3`** | **9.274** | **0** | **60.861** | **4.327.312** | **64,05%** | **2.638,24** | **466,61** | **432,5** | **212,0 - 831,0** |

Số liệu ở Bảng 3.1 phản ánh rõ tác dụng thu gọn từ điển của từng bước xử lý. Khi chuyển từ `P0` sang `P1`, việc đưa về chữ thường và gom nhóm các con số, đường dẫn đã giúp giảm 12.178 biến thể từ (từ 113.175 xuống 100.997) mà hầu như không làm hao hụt tổng số token (giữ lại 99,80%). Sự thay đổi lớn nhất diễn ra ở mức `P2`: việc ghép các âm tiết thành từ phức làm tổng số token giảm từ 6,74 triệu xuống 5,99 triệu (88,65%), đồng thời thu hẹp kích thước từ vựng thêm 39,49% (xuống còn 61.108 từ). Độ dài ký tự trung bình ở `P2` tăng nhẹ lên 3.444,85 ký tự do cách tách dấu câu và thêm dấu gạch dưới của bộ tách từ. Cuối cùng, bước lọc từ dừng ở `P3` đã loại bỏ hơn 1,66 triệu lượt xuất hiện của các từ hư, đưa tỷ lệ token giữ lại về mức 64,05% (trung bình 466,61 từ/bài) và không làm phát sinh bất kỳ văn bản rỗng nào.

**[HÌNH/BẢNG: Hình 3.3 — Tỷ lệ giữ lại token và phân phối độ dài bài viết qua các biến thể P0–P3.]**

---

## 3.4. Thiết kế biến thể phạm vi đầu vào: tiêu đề so với toàn văn

Trong các hệ thống phân tích tin tức thực tế, việc chỉ sử dụng tiêu đề hay sử dụng toàn bộ bài báo là một bài toán đánh đổi giữa tốc độ tính toán và chất lượng ngữ nghĩa (câu hỏi nghiên cứu RQ3). Tiêu đề bài báo rất ngắn gọn và đi thẳng vào sự kiện chính, nhưng thường thiếu ngữ cảnh hoặc dùng cách đặt câu ẩn dụ. Ngược lại, phần nội dung bài báo cung cấp đầy đủ các thuật ngữ chuyên môn và thông tin chi tiết, nhưng làm tăng đáng kể kích thước không gian đặc trưng.

Để so sánh công bằng hai cách tiếp cận này, nhóm trích xuất hai biến thể đầu vào từ cùng tập văn bản ở mức `P3` trên cùng một danh sách bài báo:
- **Biến thể chỉ tiêu đề (`Title-only`):** Chỉ lấy phần tiêu đề đã xử lý ở mức `P3`. Độ dài trung bình của mỗi tiêu đề chỉ đạt **9,98 từ** (trung bình 7,66 từ khóa khác không trên mỗi vector TF-IDF, với từ điển hiệu dụng gồm 4.437 từ).
- **Biến thể tiêu đề kết hợp nội dung (`Title+Content`):** Ghép phần tiêu đề với toàn bộ nội dung bài báo đã xử lý ở mức `P3`. Độ dài trung bình đạt **466,61 từ/bài** (trung bình 190,75 từ khóa khác không trên mỗi vector TF-IDF, với không gian từ điển 20.000 từ).

Việc giữ cố định mức tiền xử lý `P3`, danh sách bài báo và cấu hình thuật toán giúp đánh giá chính xác tác động của độ dài văn bản đầu vào trong thực nghiệm tại Mục 6.3.

---

## 3.5. Giao thức chia tập dữ liệu và chống rò rỉ thông tin

Để đảm bảo tính nhất quán khi so sánh giữa các mô hình, tập dữ liệu sạch gồm 9.274 bài báo được chia cố định một lần duy nhất thành ba tập Train, Validation và Test theo tỷ lệ 70% / 15% / 15%. Nhóm sử dụng phương pháp lấy mẫu phân tầng theo nhãn chuyên mục với hạt giống ngẫu nhiên `seed = 42`. Bảng 3.2 trình bày số lượng và tỷ lệ bài viết của từng chuyên mục trên ba tập dữ liệu.

**Bảng 3.2.** Phân phối số lượng và tỷ lệ bài viết theo chuyên mục trên ba tập Train, Validation và Test.

| Chuyên mục | Train | Tỷ lệ Train (%) | Validation | Tỷ lệ Val (%) | Test | Tỷ lệ Test (%) | Tổng cộng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kinh doanh** | 1.388 | 21,38% | 297 | 21,35% | 297 | 21,37% | 1.982 |
| **Thể thao** | 1.373 | 21,15% | 293 | 21,06% | 293 | 21,08% | 1.959 |
| **Sức khỏe** | 865 | 13,32% | 185 | 13,30% | 185 | 13,31% | 1.235 |
| **Pháp luật** | 829 | 12,77% | 177 | 12,72% | 178 | 12,81% | 1.184 |
| **Giải trí** | 825 | 12,71% | 178 | 12,80% | 177 | 12,73% | 1.180 |
| **Giáo dục** | 723 | 11,14% | 156 | 11,21% | 155 | 11,15% | 1.034 |
| **Khoa học công nghệ** | 490 | 7,55% | 105 | 7,55% | 105 | 7,55% | 700 |
| **Tổng cộng** | **6.493** | **70,01%** | **1.391** | **15,00%** | **1.390** | **14,99%** | **9.274** |

Số liệu ở Bảng 3.2 cho thấy tỷ lệ các chuyên mục được giữ đồng đều giữa cả ba tập Train (6.493 bài), Validation (1.391 bài) và Test (1.390 bài), với độ lệch không quá 0,09 điểm phần trăm. Nhằm ngăn chặn hiện tượng rò rỉ dữ liệu từ tập kiểm tra sang quá trình huấn luyện, quy trình thực nghiệm tuân thủ ba quy tắc:
1. **Trên tập Train:** Mọi bộ trích xuất đặc trưng (như từ điển BoW, trọng số TF-IDF) và tham số mô hình chỉ được học (`fit`) trên 6.493 bài của tập Train.
2. **Trên tập Validation:** 1.391 bài của tập Validation được biến đổi (`transform`) dựa trên từ điển đã khóa của tập Train. Tập này dùng để chọn số lượng chủ đề $K$ tối ưu, kiểm tra độ ổn định qua nhiều hạt giống ngẫu nhiên và chạy các thí nghiệm so sánh tiền xử lý.
3. **Trên tập Test:** 1.390 bài của tập Test được giữ độc lập hoàn toàn và chỉ được chạy đánh giá một lần duy nhất sau khi cấu hình tốt nhất của từng mô hình đã được chốt trên tập Validation. Quy tắc đánh giá được khóa cố định nghiêm ngặt: Validation dùng chọn cấu hình; Test chỉ mở sau khi chốt mô hình.

**[HÌNH/BẢNG: Hình 3.4 — Phân bố tỷ lệ 7 chuyên mục biên tập trên ba tập Train, Validation và Test.]**

---

## References — Tài liệu tham khảo (APA 7)

Broder, A. Z. (1997). On the resemblance and containment of documents. In *Proceedings. Compression and Complexity of SEQUENCES 1997* (pp. 21–29). IEEE. https://doi.org/10.1109/SEQUEN.1997.666900

Vu, T., Nguyen, D. Q., Nguyen, D. Q., Dras, M., & Johnson, M. (2018). VnCoreNLP: A Vietnamese natural language processing toolkit. In *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Demonstrations* (pp. 56–60). Association for Computational Linguistics. https://doi.org/10.18653/v1/N18-5012
