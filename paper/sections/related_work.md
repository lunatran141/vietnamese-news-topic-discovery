# Related Work — Nghiên cứu liên quan

Các phương pháp khám phá chủ đề văn bản đã phát triển theo nhiều hướng, từ phân cụm dựa trên đặc trưng từ vựng đến mô hình hóa chủ đề xác suất và các phương pháp khai thác biểu diễn ngữ nghĩa. Các hướng tiếp cận này khác nhau về cách biểu diễn tài liệu, cơ chế hình thành chủ đề, khả năng diễn giải và chi phí tính toán. Phần này tổng hợp những nghiên cứu tiêu biểu nhằm xây dựng cơ sở khoa học cho việc so sánh các phương pháp khám phá chủ đề trên dữ liệu tin tức tiếng Việt.

## 2.1 Classical Clustering — Phân cụm truyền thống

Các phương pháp phân cụm văn bản truyền thống thường bắt đầu bằng việc biểu diễn tài liệu trong không gian vector từ vựng. Những kỹ thuật như Bag-of-Words và Term Frequency–Inverse Document Frequency (TF-IDF) chuyển nội dung văn bản thành các vector đặc trưng, từ đó cho phép đo mức độ tương đồng giữa các tài liệu. Trên biểu diễn này, K-Means và các biến thể có thể tổ chức tài liệu thành những cụm dựa trên mức độ tương đồng với tâm cụm.

Steinbach et al. (2000) thực hiện nghiên cứu so sánh K-Means tiêu chuẩn, Bisecting K-Means và các phương pháp phân cụm phân cấp trên tám bộ dữ liệu văn bản tiếng Anh, bao gồm Reuters-21578, WebACE và các tập thuộc TREC. Nghiên cứu sử dụng entropy, F-measure và độ tương đồng nội cụm để đánh giá kết quả. Theo các điều kiện thí nghiệm được báo cáo, Bisecting K-Means cho kết quả tốt hơn những phương pháp đối chứng trong nhiều trường hợp. Đáng chú ý, nghiên cứu chỉ ra rằng K-Means tiêu chuẩn có thể nhạy cảm với khởi tạo ngẫu nhiên và tạo ra các cụm có kích thước mất cân bằng, ảnh hưởng đến chất lượng phân nhóm. Vì vậy, kết quả phân cụm cần được phân tích cùng cấu hình thuật toán và đặc điểm dữ liệu, thay vì chỉ dựa trên một lần chạy.

Khi áp dụng các phương pháp dựa trên từ vựng cho tiếng Việt, chất lượng tiền xử lý có vai trò quan trọng. Vu et al. (2018) giới thiệu VnCoreNLP, bộ công cụ hỗ trợ tách từ, gán nhãn từ loại, nhận diện thực thể và phân tích cú pháp phụ thuộc tiếng Việt. Nghiên cứu nhấn mạnh rằng khoảng trắng trong tiếng Việt có thể phân tách các âm tiết thuộc cùng một từ, do đó việc xác định ranh giới từ cần được xử lý riêng. Kết quả của nghiên cứu cung cấp cơ sở cho việc lựa chọn tách từ trước khi xây dựng biểu diễn từ vựng. Tuy nhiên, VnCoreNLP là công cụ được khảo sát trong nghiên cứu gốc, còn đồ án sử dụng PyVi để thực hiện tách từ tiếng Việt; vì vậy, không áp dụng trực tiếp các kết quả đánh giá của VnCoreNLP cho pipeline của nhóm.

Hai công trình trên hỗ trợ những khía cạnh khác nhau của phương pháp phân cụm truyền thống. Steinbach et al. (2000) cung cấp cơ sở thực nghiệm cho việc sử dụng và đánh giá K-Means trong phân cụm tài liệu, trong khi Vu et al. (2018) làm rõ vai trò của tiền xử lý tiếng Việt. Tuy nhiên, các thước đo chất lượng phân cụm trên dữ liệu tiếng Anh không thể được chuyển trực tiếp sang tập tin tức tiếng Việt. Tương tự, độ chính xác tách từ trên bộ dữ liệu gán nhãn không đồng nghĩa với việc chất lượng phân cụm chủ đề sẽ tăng tương ứng.

Từ đó, đồ án lựa chọn TF-IDF kết hợp K-Means làm hướng đối chứng truyền thống, đồng thời xem xét ảnh hưởng của các biến thể tiền xử lý và tách từ trên cùng bộ dữ liệu tin tức tiếng Việt. Cách tiếp cận này cho phép đánh giá vai trò của biểu diễn từ vựng trong điều kiện thực nghiệm thống nhất với những phương pháp khám phá chủ đề còn lại.

## 2.2 Topic Modeling — Mô hình hóa chủ đề

Khác với phân cụm truyền thống, mô hình hóa chủ đề hướng đến việc xác định các thành phần chủ đề tiềm ẩn trong tập tài liệu. Trong số những hướng tiếp cận tiêu biểu, Non-negative Matrix Factorization (NMF) dựa trên phân rã ma trận không âm, trong khi Latent Dirichlet Allocation (LDA) dựa trên mô hình xác suất sinh.

Lee và Seung (1999) nghiên cứu NMF thông qua việc xấp xỉ một ma trận không âm bằng tích của hai ma trận không âm có số chiều thấp hơn. Ràng buộc không âm tạo ra cơ chế kết hợp cộng tính giữa các thành phần biểu diễn. Bên cạnh thí nghiệm trên hình ảnh, công trình còn minh họa việc khám phá các đặc trưng ngữ nghĩa từ tập văn bản Grolier Encyclopedia. Các kết quả cung cấp nền tảng cho việc sử dụng NMF để trích xuất những thành phần có thể diễn giải trong dữ liệu văn bản. Tuy nhiên, nghiên cứu gốc không phải một benchmark so sánh NMF với các mô hình chủ đề hiện đại bằng ARI, NMI hoặc topic coherence.

Blei et al. (2003) đề xuất LDA, mô hình xác suất sinh trong đó mỗi tài liệu được biểu diễn bằng một phân phối xác suất trên các chủ đề và mỗi chủ đề tương ứng với một phân phối xác suất trên từ vựng. Theo cơ chế này, một tài liệu có thể đồng thời liên quan đến nhiều chủ đề. Nghiên cứu sử dụng variational inference để xấp xỉ quá trình suy luận và đánh giá khả năng mô hình hóa tài liệu thông qua held-out perplexity trên các corpus tiếng Anh, bao gồm văn bản khoa học và tin tức Associated Press. LDA đạt kết quả perplexity thuận lợi so với những mô hình xác suất đối chứng được khảo sát. Tuy vậy, mô hình cơ bản vẫn dựa trên giả định Bag-of-Words và cần xác định trước số lượng chủ đề.

NMF và LDA vì thế cung cấp hai cơ chế khác nhau để khai thác cấu trúc chủ đề. NMF biểu diễn dữ liệu thông qua các thành phần không âm, còn LDA mô hình hóa phân phối xác suất của chủ đề và từ vựng. Khác biệt về nguyên lý này không tự động chứng minh một phương pháp có chất lượng cao hơn phương pháp còn lại. Việc so sánh cần sử dụng dữ liệu, tiền xử lý và chỉ số đánh giá phù hợp.

Một vấn đề quan trọng khác là chất lượng những từ khóa đại diện cho chủ đề. O'Callaghan et al. (2015) khảo sát các bộ từ mô tả chủ đề do NMF và LDA tạo ra trên nhiều tập dữ liệu tiếng Anh, gồm BBC, Guardian, New York Times và Wikipedia. Nghiên cứu phân tích ảnh hưởng của biểu diễn dữ liệu và cách lựa chọn từ khóa đến độ mạch lạc của mô tả chủ đề. Kết quả cho thấy cách xây dựng topic descriptor có thể ảnh hưởng đáng kể đến những chỉ số đánh giá. Do đó, khi phân tích chênh lệch giữa NMF và LDA, cần tách biệt ảnh hưởng của thuật toán học với ảnh hưởng của biểu diễn đầu vào và phương pháp xếp hạng từ khóa.

Röder et al. (2015) đề xuất một khung thống nhất để nghiên cứu các thước đo topic coherence, bao gồm các bước phân chia tập từ cần so sánh, ước lượng xác suất, đo mức độ liên hệ và tổng hợp kết quả. Công trình khảo sát nhiều cấu hình đánh giá và đối chiếu với nhận xét của con người về khả năng diễn giải chủ đề. Các thước đo như C_V và NPMI vì vậy có cơ sở để sử dụng trong thực nghiệm. Tuy nhiên, giá trị coherence còn phụ thuộc vào corpus tham chiếu, cách tiền xử lý và phương pháp ước lượng xác suất; đây không phải phép đo đầy đủ mọi khía cạnh của chất lượng mô hình.

Những nghiên cứu trên hình thành cơ sở cho việc xây dựng và đánh giá NMF, LDA trong đồ án. Trên cùng bộ dữ liệu tin tức tiếng Việt, nhóm xem xét chất lượng các từ khóa chủ đề, phân bố tài liệu theo chủ đề, các chỉ số định lượng phù hợp, độ đa dạng và tính ổn định của kết quả. Việc phân tích tài liệu đại diện và đánh giá của con người cũng cần được xem xét bổ sung, bởi một mô hình có điểm coherence tốt chưa chắc tạo ra mọi chủ đề đều rõ nghĩa trong thực tế.

## 2.3 Semantic Topic Discovery — Khám phá chủ đề dựa trên ngữ nghĩa

Những phương pháp dựa trên Bag-of-Words hoặc TF-IDF chủ yếu khai thác thông tin xuất hiện từ vựng. Khi hai văn bản diễn đạt những nội dung tương tự bằng các từ khác nhau, biểu diễn từ vựng có thể chưa phản ánh đầy đủ mối quan hệ ngữ nghĩa. Các phương pháp embedding tạo ra biểu diễn vector của tài liệu để hỗ trợ nhận diện những quan hệ này, từ đó mở rộng khả năng khám phá chủ đề.

Grootendorst (2022) đề xuất BERTopic, một hướng mô hình hóa chủ đề kết hợp embedding tài liệu, giảm chiều, phân cụm và tạo từ khóa đại diện bằng class-based TF-IDF (c-TF-IDF). Trong thiết lập của nghiên cứu, tài liệu được biểu diễn bằng embedding, giảm chiều với UMAP và gom cụm bằng HDBSCAN. Các tài liệu trong cùng cụm sau đó được tổng hợp để xây dựng biểu diễn từ khóa đặc trưng bằng c-TF-IDF. Nghiên cứu đánh giá trên 20 Newsgroups, BBC News và Trump Tweets, sử dụng các chỉ số như NPMI topic coherence, topic diversity và thời gian thực thi.

Điểm đáng chú ý của BERTopic là quá trình hình thành cụm được tách khỏi bước lựa chọn từ khóa đại diện, giúp hỗ trợ việc thay đổi biểu diễn hoặc tinh chỉnh chủ đề. Tuy nhiên, Grootendorst (2022) cũng thảo luận những hạn chế liên quan đến việc một tài liệu có thể chứa nhiều chủ đề và sự dư thừa của các từ khóa được tạo bởi c-TF-IDF. Bên cạnh đó, kết quả và chi phí tính toán phụ thuộc vào mô hình embedding, phương pháp phân cụm và điều kiện phần cứng. Do vậy, ưu điểm của biểu diễn ngữ nghĩa cần được đánh giá đồng thời với tính dễ diễn giải và hiệu quả tính toán.

Trong lĩnh vực tin tức, khám phá chủ đề còn có liên hệ với phát hiện sự kiện. Nakshatri et al. (2023) nghiên cứu hệ thống KeyEvents kết hợp tín hiệu thời gian, biểu diễn ngữ nghĩa, HDBSCAN và mô hình ngôn ngữ lớn để hỗ trợ phát hiện và tóm tắt các sự kiện nổi bật. Công trình sử dụng thông tin về thời điểm xuất bản nhằm lựa chọn các bài báo liên quan, sau đó tổ chức và tổng hợp thành những nhóm sự kiện. Việc đánh giá được thực hiện qua coverage, entity purity, entity coherence và đánh giá thủ công. Kết quả chỉ ra sự đánh đổi giữa mức độ bao phủ tài liệu và độ nhất quán của các nhóm sự kiện.

Nghiên cứu này cũng làm rõ sự khác biệt giữa topic discovery và event discovery. Một chủ đề rộng có thể bao gồm nhiều sự kiện riêng biệt, trong khi bài toán phát hiện sự kiện thường cần bổ sung thông tin thời gian và tiêu chí xác định cùng một sự kiện. Do đó, các chủ đề được khám phá bằng mô hình phân cụm không nên mặc nhiên được diễn giải là những sự kiện cụ thể.

Một hướng nghiên cứu bổ sung là điều chỉnh số lượng chủ đề sau khi phân cụm. Janssens et al. (2025) so sánh các kỹ thuật topic reduction dành cho BERTopic, gồm điều chỉnh tham số HDBSCAN và gộp các chủ đề dựa trên những biểu diễn tương đồng. Nghiên cứu phân tích các thay đổi về NPMI, topic diversity, tỷ lệ outlier, thời gian thực thi và chi phí sử dụng mô hình ngôn ngữ lớn. Kết quả cho thấy quá trình giảm số chủ đề có thể tạo ra đánh đổi giữa độ mạch lạc, độ đa dạng và chi phí tính toán. Các kết luận được rút ra từ những tập dữ liệu và cấu hình cụ thể, nên không thể xem là bảo đảm hiệu quả cho mọi ngôn ngữ hoặc miền văn bản.

Ba nghiên cứu trên hỗ trợ việc khảo sát embedding, HDBSCAN và BERTopic trong bài toán khám phá chủ đề tin tức. Tuy nhiên, các thực nghiệm gốc chủ yếu sử dụng dữ liệu tiếng Anh, mạng xã hội hoặc những miền văn bản khác dữ liệu của đồ án. Vì vậy, nhóm cần kiểm chứng ảnh hưởng của lựa chọn embedding, khả năng diễn giải từ khóa, số lượng cụm, các tài liệu outlier và chi phí xử lý trên corpus tin tức tiếng Việt. Những chỉ số này cũng cần được so sánh trong điều kiện thống nhất với các phương pháp TF-IDF/K-Means, NMF và LDA.

Nhìn chung, các nghiên cứu được khảo sát phản ánh ba hướng tiếp cận có cơ chế và ưu điểm khác nhau: phân cụm truyền thống dựa trên đặc trưng từ vựng, mô hình hóa các chủ đề tiềm ẩn và khám phá chủ đề dựa trên biểu diễn ngữ nghĩa. Do sự khác biệt về bộ dữ liệu, quy trình thực nghiệm và thước đo, các kết quả đã công bố không đủ để xếp hạng trực tiếp các phương pháp cho tiếng Việt. Đồ án Vietnamese News Topic Discovery vì vậy tập trung vào so sánh thực nghiệm trên một bộ dữ liệu tin tức tiếng Việt thống nhất, kết hợp những đánh giá định lượng và khả năng diễn giải để phân tích sự phù hợp của từng hướng tiếp cận.

## References — Tài liệu tham khảo (APA 7)

Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation. *Journal of Machine Learning Research, 3*, 993–1022. https://www.jmlr.org/papers/v3/blei03a.html

Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a class-based TF-IDF procedure* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2203.05794

Janssens, W., Bogaert, M., & Van den Poel, D. (2025). A comparative analysis of topic reduction techniques for BERTopic. *IEEE Access, 13*, 204087–204103. https://doi.org/10.1109/ACCESS.2025.3638956

Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. *Nature, 401*, 788–791. https://doi.org/10.1038/44565

Nakshatri, N., Liu, S., Chen, S., Roth, D., Goldwasser, D., & Hopkins, D. (2023). Using LLM for improving key event discovery: Temporal-guided news stream clustering with event summaries. In *Findings of the Association for Computational Linguistics: EMNLP 2023* (pp. 4162–4173). https://doi.org/10.18653/v1/2023.findings-emnlp.274

O'Callaghan, D., Greene, D., Carthy, J., & Cunningham, P. (2015). An analysis of the coherence of descriptors in topic modeling. *Expert Systems with Applications, 42*(13), 5645–5657. https://doi.org/10.1016/j.eswa.2015.02.055

Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic coherence measures. In *Proceedings of the Eighth ACM International Conference on Web Search and Data Mining* (pp. 399–408). https://doi.org/10.1145/2684822.2685324

Steinbach, M., Karypis, G., & Kumar, V. (2000). *A comparison of document clustering techniques* (Technical Report No. TR 00-034). University of Minnesota. https://hdl.handle.net/11299/215421

Vu, T., Nguyen, D. Q., Nguyen, D. Q., Dras, M., & Johnson, M. (2018). VnCoreNLP: A Vietnamese natural language processing toolkit. In *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Demonstrations* (pp. 56–60). https://doi.org/10.18653/v1/N18-5012
