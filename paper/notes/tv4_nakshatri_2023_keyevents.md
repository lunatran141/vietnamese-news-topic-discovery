# Paper notes — Nakshatri et al. (2023): KeyEvents

## 1. Người cung cấp

- Thành viên: TV4 — Vân
- Phương pháp phụ trách: Embeddings, HDBSCAN, event detection
- Ngày đọc: 09/10/2026

## 2. Citation info — Thông tin trích dẫn

- Citation key: nakshatri_2023_keyevents
- Tên đầy đủ của tài liệu: Using LLM for Improving Key Event Discovery: Temporal-Guided News Stream Clustering with Event Summaries
- Tác giả: Nishanth Nakshatri; Siyi Liu; Sihao Chen; Daniel J. Hopkins; Dan Roth; Dan Goldwasser (theo thứ tự trên trang đầu PDF).
- Năm xuất bản: 2023
- Nơi xuất bản: *Findings of the Association for Computational Linguistics: EMNLP 2023* (Findings of EMNLP 2023), trang 4162–4173; bài trong kỷ yếu hội nghị.
- DOI: [`10.18653/v1/2023.findings-emnlp.274`](https://doi.org/10.18653/v1/2023.findings-emnlp.274).
- Nguồn: [ACL Anthology](https://aclanthology.org/2023.findings-emnlp.274/) · [PDF gốc](https://aclanthology.org/2023.findings-emnlp.274.pdf).

## 3. Method — Phương pháp

- **Bài toán:** Phát hiện sự kiện thu hút truyền thông và xác định bài báo thuộc từng sự kiện trong luồng tin, theo *issue* (vấn đề/chủ đề) người dùng quan tâm.
- **Truy xuất bài:** GPT-3.5 tạo từ khóa cho issue; BM25 lấy bài liên quan từ NELA (Appendix A).
- **Lọc theo thời gian:** Tìm các đỉnh trong chuỗi số bài theo ngày, rồi xử lý các bài quanh từng đỉnh (Section 2.1; Appendix B).
- **Tạo cụm ứng viên:** Mã hóa **tiêu đề và bốn dòng đầu** bằng dense retriever; giảm chiều embedding bằng **UMAP** trước khi chạy **HDBSCAN** trong từng đỉnh (Section 2.1; Appendix C).
- **Mô tả và làm sạch sự kiện:** GPT-3.5 tạo tóm tắt đa văn bản từ hai bài gần tâm cụm nhất. Hệ thống so cosine similarity giữa embedding tóm tắt và năm bài đại diện để phát hiện cụm không nhất quán; GPT-3.5 cũng xét các cặp tóm tắt để gộp cụm cùng sự kiện (Section 2.1; Tables 1–2). Loại cụm ứng viên **không có nghĩa** loại hẳn các bài của cụm khỏi bước gán tiếp theo.
- **Ánh xạ bài → sự kiện:** Mã hóa tóm tắt sự kiện đã cập nhật và bài bằng cùng encoder; dùng **cosine similarity và ngưỡng** để quyết định gán bài. Xét thêm các bài trong cửa sổ **một ngày trước/sau đỉnh** (Section 2.2). Nhãn HDBSCAN ban đầu chưa phải kết quả gán cuối cùng.

## 4. Dataset — Dữ liệu

- **Kho gốc:** NELA-GT-2021 có khoảng **1,8 triệu** bài từ **367 nguồn** trong năm 2021 (Appendix A). Đây là quy mô kho nguồn, không phải số bài đưa thẳng vào phân cụm.
- **Phạm vi nghiên cứu:** Tác giả tập trung vào **207 nguồn tại Mỹ**. Bước truy xuất thu khoảng **335 nghìn** bài liên quan đến **11 issue**; lọc theo đỉnh còn khoảng **90 nghìn** bài cho phân tích (Section 3; Appendix A).
- **Sản phẩm KeyEvents:** Khoảng **40 nghìn** bài, **611 sự kiện**, **11 issue/chủ đề** (Abstract). Cần phân biệt quy mô bộ dữ liệu được tạo ra với kho NELA và tập trung gian 335 nghìn/90 nghìn bài.
- **Miền dữ liệu:** Tin tiếng Anh từ các nguồn Mỹ. Bài báo không đánh giá trực tiếp trên tin tiếng Việt hoặc dữ liệu ngoài thời gian/nguồn này.

## 5. Metric — Đánh giá

- **Coverage:** Tỷ lệ bài được tính đến trong phép gán cụm/sự kiện. Nên đọc cùng chất lượng cụm: gán nhiều bài hơn không tự động có kết quả tốt hơn.
- **Entity purity:** Trong mỗi sự kiện, tỷ lệ bài nhắc ít nhất một trong **10 thực thể trung tâm** chọn bằng TF-IDF.
- **Entity coherence:** Đo đồng xuất hiện của các cặp thực thể trung tâm trong bài cùng sự kiện; theo công thức của bài, giá trị **cao hơn và gần 0 hơn** là tốt hơn vì điểm thường âm (Section 3; Appendix D). Đây không phải purity theo nhãn chuyên mục hay topic coherence theo từ khóa của đồ án.

**Table 3 — trung bình trên 11 issue:**

| Phương pháp | Coverage (%) | Entity purity (%) | Entity coherence |
|---|---:|---:|---:|
| HDBSCAN | 81.78 | 62.55 | -776.80 |
| BERTopic | 84.04 | 66.00 | -726.11 |
| Đề xuất, vòng 1 | 44.29 | 82.69 | -477.89 |
| Đề xuất, vòng 2 | 56.83 | 77.49 | -579.48 |

Vòng 1 có purity/coherence cao hơn các baseline trên nhưng coverage thấp. Vòng 2 tăng coverage bằng cách cập nhật embedding tóm tắt sự kiện, đổi lại purity/coherence giảm. **Phương pháp đề xuất không tốt nhất trên mọi metric** (Section 4.1; Table 3).

**Table 4 — human evaluation trên mẫu thuộc issue *Climate Change*:**

| Phương pháp | Event coherence (%) | Mapping quality, precision (%) |
|---|---:|---:|
| HDBSCAN | 84.90 | 62.27 |
| BERTopic | 85.48 | 69.87 |
| Đề xuất | 91.07 | 72.19 |

Người đánh giá xem ba bài đại diện có cùng sự kiện và bài được gán có phù hợp tóm tắt sự kiện hay không. Từ nhãn thủ công, tác giả lập các cặp bài đúng/sai để so sánh **mapping precision** giữa phương pháp; đây không phải tỷ lệ gán đúng trên toàn bộ kho tin (Section 4.3; Appendix E). Mẫu chỉ thuộc **một issue**, không phải đánh giá thủ công toàn bộ 11 issue/KeyEvents. Chênh lệch mapping precision với BERTopic là 2,32 điểm phần trăm; không nên gọi là “vượt trội hơn hẳn”.

## 6. Gap — Hạn chế và nhận xét riêng

**Tác giả nêu trực tiếp:**

- Bước lọc theo thời gian **cần timestamp/ngày xuất bản** của bài (mục *Limitations*, trang 4167).
- Pipeline dùng **GPT-3.5** để tạo tóm tắt và gộp cụm. Tác giả thảo luận chi phí, giới hạn tài nguyên và để ngỏ đánh đổi giữa hiệu quả tính toán với chất lượng. Chi phí API dưới 5 USD là số liệu cho **thí nghiệm trong bài**, không phải dự toán cho tập khác (mục *Limitations*).
- Tác giả quan sát một trường hợp **lọc nhầm cụm tốt** bằng kiểm tra similarity (Section 4.2; Table 7). Đây là phân tích lỗi cụ thể, không phải tuyên bố trong mục *Limitations*.

**Suy luận của TV4, chưa phải kết luận đã được tác giả kiểm chứng:**

- Sự kiện diễn ra từ từ, không có đỉnh đưa tin rõ, **có thể** khó được bước lọc theo đỉnh chọn. Đây là rủi ro suy ra từ thiết kế (Section 2.1; Appendix B); cần thử nghiệm riêng để biết mức bỏ sót.
- Chuyển sang báo tiếng Việt, nguồn tin mới hoặc thời gian khác **có thể** cần điều chỉnh truy xuất, encoder, ngưỡng similarity và cách xử lý timestamp. Bài báo không có kết quả OOD/tiếng Việt để kết luận mức ảnh hưởng.

## 7. Liên hệ với đồ án

- **M10–M11:** Bài báo gợi ý luồng embedding → UMAP → HDBSCAN; cần phân biệt phân cụm trên tập train của đồ án với cách tác giả **lọc theo issue và đỉnh thời gian trước khi phân cụm**.
- **M12:** Việc tác giả phát hiện cụm thiếu nhất quán và ví dụ lọc nhầm cụm tốt là lý do nên đọc ca outlier/cụm khó, không chỉ dựa vào nhãn nhiễu hoặc một ngưỡng similarity.
- **Event detection / OOD:** Có thể thử kết hợp cụm ngữ nghĩa với ngày xuất bản, kiểm tra sự kiện không có đỉnh rõ, nguồn mới và giai đoạn mới. Đây là **đề xuất cho đồ án**, chưa phải kết quả thực nghiệm của bài báo.
- **Khác biệt triển khai:** Nếu đồ án chưa có tóm tắt sự kiện bằng GPT-3.5, bước gộp cụm và bước gán lại bài qua tóm tắt, không mô tả HDBSCAN hiện tại là bản tái tạo đầy đủ của KeyEvents.

## 8. Vị trí bằng chứng

| Nhận định/số liệu | Vị trí trong [PDF gốc](https://aclanthology.org/2023.findings-emnlp.274.pdf) |
|---|---|
| Venue, DOI, trang | Trang đầu PDF; [ACL Anthology](https://aclanthology.org/2023.findings-emnlp.274/) |
| GPT-3.5/BM25 để truy xuất; NELA 1,8 triệu bài/367 nguồn; chọn 207 nguồn Mỹ | Appendix A, trang 4169 |
| Đỉnh thời gian; title + bốn dòng đầu; UMAP; HDBSCAN; GPT-3.5; lọc/gộp cụm | Section 2.1, trang 4163–4164; Appendices B–C |
| Gán bài bằng cosine similarity/ngưỡng và cửa sổ ±1 ngày | Section 2.2, trang 4164 |
| 335 nghìn bài truy xuất; 90 nghìn bài sau lọc | Section 3, trang 4164 |
| KeyEvents khoảng 40 nghìn bài, 611 sự kiện, 11 issue | Abstract, trang 4162 |
| Coverage, entity purity, entity coherence | Section 3, trang 4164; Appendix D, trang 4170 |
| Số liệu tự động và đánh đổi purity–coverage | Table 3; Section 4.1, trang 4165 |
| Human evaluation và phạm vi mẫu Climate Change | Table 4; Section 4.3, trang 4165–4166; Appendix E, trang 4170–4171 |
| Trường hợp lọc nhầm cụm tốt | Section 4.2, trang 4165; Table 7, trang 4171 |
| Timestamp, GPT-3.5, chi phí và hướng nghiên cứu | *Limitations*, trang 4167 |

## 9. Điều cần xác minh thêm khi áp dụng vào đồ án

- Kiểm tra chất lượng `published_at` và khả năng phát hiện sự kiện không có đỉnh rõ trên dữ liệu tiếng Việt.
- Chọn cách biểu diễn và ngưỡng ánh xạ bài → sự kiện trên validation; không áp dụng trực tiếp ngưỡng hoặc số liệu NELA cho đồ án.
- Nếu thử tóm tắt/gộp cụm, đo chất lượng sự kiện, coverage, chi phí và lỗi lọc nhầm cụm tốt trên cùng tập bài với HDBSCAN hiện tại.
- Nếu đánh giá nguồn/thời gian mới, tách riêng thành kiểm tra OOD và không dùng tập OOD để tinh chỉnh cấu hình.

## 10. Trạng thái và bàn giao

- Trạng thái: đã đối chiếu với PDF gốc; chờ TV3 review/nghiệm thu.
- File bàn giao: `paper/notes/tv4_nakshatri_2023_keyevents.md`.
- Nhận xét của TV3 sau review: Chưa có.
