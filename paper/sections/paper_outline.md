# Vietnamese News Topic Discovery — Khung báo cáo NCKH

**Người phụ trách tổng hợp:** TV3 — Tuấn
**Trạng thái:** Bản khung nội bộ, chưa phải bản nộp chính thức.

## 1. Title, Authors, Abstract & Keywords

**Phụ trách:** Tuấn

- Title: Đã có tên bài báo dự kiến trong `paper/paper_nckh.md`.
- Authors & Affiliations: Đã điền 5 tác giả và đơn vị trong `paper/paper_nckh.md`.
- Abstract: Viết sau khi tổng hợp kết quả năm phương pháp.
- Keywords: Chốt sau Abstract.

## 2. Introduction

**Phụ trách:** Tuấn

- Bối cảnh khám phá chủ đề trong tập tin tức tiếng Việt.
- Problem Statement.
- Research Questions.
- Mục tiêu nghiên cứu.
- Những đóng góp chính của đề tài.
- Giới thiệu cấu trúc bài báo.

## 3. Related Work

**Phụ trách tổng hợp:** Tuấn; nguồn từ TV1–TV5.

- Classical Clustering: TF-IDF, K-Means và tiền xử lý tiếng Việt.
- Topic Modeling: NMF, LDA và đánh giá topic coherence.
- Semantic Topic Discovery: Embeddings, HDBSCAN và BERTopic.
- Tổng hợp khoảng trống nghiên cứu và mối liên hệ với đề tài.

**Nguồn quản lý:** `paper/related_work_matrix.csv`, `paper/sections/related_work.md`.

**Trạng thái:** Matrix có 7 công trình; 2 nguồn TV2 chờ bổ sung.

## 4. Dataset & Preprocessing

**Phụ trách chuyên môn:** TV1 — Khôi.

- Nguồn và quy mô Dataset v1.0.
- Tiền xử lý dữ liệu.
- Các preprocessing variants.
- Chia Train/Validation/Test.
- Kiểm tra dữ liệu trùng lặp, leakage và reproducibility.

**Trạng thái:** Chờ TV1 bàn giao nội dung Paper.

## 5. Methodology

**Phần giới thiệu chung:** Tuấn.

**Các phần phương pháp chuyên môn:**

- TF-IDF + K-Means — TV1, Khôi.
- NMF — TV2, Tiến.
- Bag-of-Words + LDA — TV3, Tuấn.
- Embeddings + HDBSCAN — TV4, Vân.
- BERTopic — TV5, Ngân.

**Nguồn LDA:** `paper/sections/lda_method_results.md`.

**Nguồn NMF:** `paper/sections/nmf_method_results.md`.

## 6. Experimental Setup

**Phụ trách tổng hợp:** Tuấn, nhận thông tin từ các thành viên.

- Dataset và split protocol chung.
- Cấu hình thực nghiệm của từng phương pháp.
- Chiến lược chọn hyperparameters.
- Các metrics và cách tính.
- Random seeds và môi trường thực nghiệm.
- Quy tắc không sử dụng Test để tuning.

**Lưu ý:** Chỉ so sánh trực tiếp các chỉ số khi giao thức và cách tính tương thích.

## 7. Results

**Nguồn:** Kết quả thực nghiệm do từng thành viên cung cấp.

- K-Means — Khôi.
- NMF — Tiến.
- LDA — Tuấn.
- HDBSCAN — Vân.
- BERTopic — Ngân.

**Phần LDA cần có:** lựa chọn K, stability, Final Test, keywords và representative documents.

## 8. Main Model Comparison

**Phụ trách tổng hợp:** Tuấn.

Bảng so sánh năm phương pháp cần ghi rõ:

- Preprocessing và input representation.
- Cấu hình cuối cùng.
- Topic coherence và topic diversity.
- Chỉ số ngoài nếu có và phù hợp.
- Runtime nếu đã benchmark.
- Khả năng diễn giải và giới hạn của từng phương pháp.

Không điền kết quả chưa được xác nhận hoặc so sánh các giá trị tính bằng giao thức không tương đương như thể chúng hoàn toàn giống nhau.

## 9. Discussion & Limitations

**Phụ trách tổng hợp:** Tuấn.

- Nhận xét về các hướng biểu diễn văn bản.
- Chất lượng và khả năng diễn giải topic.
- Sự khác biệt giữa phân cụm truyền thống, topic modeling và semantic topic discovery.
- Đánh đổi giữa coherence, diversity, outlier và chi phí tính toán.
- Các giới hạn chung của dữ liệu và đánh giá.

Mỗi thành viên tự viết phần Discussion và Limitations cho phương pháp mình thực hiện.

## 10. System / Application

**Phụ trách:** TV5 — Ngân.

- Cấu trúc và chức năng ứng dụng.
- Quy trình đưa dữ liệu và mô hình vào hệ thống.
- Minh họa kết quả trên giao diện.
- Giới hạn triển khai.

**Trạng thái:** Chờ TV5 bàn giao.

## 11. Conclusion & Future Work

**Phụ trách:** Tuấn.

- Tóm tắt kết quả nghiên cứu.
- Trả lời các Research Questions.
- Những đóng góp đã được kiểm chứng.
- Hạn chế còn tồn tại.
- Hướng phát triển tiếp theo.

Viết sau khi chốt toàn bộ Results và Discussion.

## 12. References

**Phụ trách chuẩn hóa:** Tuấn.

- Thống nhất một chuẩn citation.
- Đảm bảo nguồn trong nội dung khớp References.
- Loại bỏ trùng lặp tài liệu.
- Kiểm tra metadata và DOI/URL.

## 13. Appendix

**Phụ trách hợp nhất:** Tuấn.

- Bảng tham số đầy đủ.
- Kết quả thực nghiệm bổ sung.
- Ví dụ bài báo đại diện.
- Hình ảnh, bảng biểu và tài liệu tái lập phù hợp.

## Checklist tổng biên tập

- [ ] Nhận đủ nội dung chuyên môn từ các thành viên.
- [ ] Viết đầy đủ Introduction và phần kết nối chung.
- [x] Hoàn thiện báo cáo LDA.
- [ ] Tổng hợp Experimental Setup và bảng so sánh năm mô hình.
- [ ] Viết Abstract, Keywords, Discussion chung và Conclusion.
- [ ] Kiểm tra citations và References.
- [ ] Kiểm tra tất cả số liệu với artifact gốc.
- [ ] Hợp nhất hình, bảng và phụ lục.
- [ ] Chuẩn hóa hình thức, bố cục và độ dài theo yêu cầu nhóm.
- [ ] Nhận xác nhận từ các thành viên trước khi chốt bản nộp.
