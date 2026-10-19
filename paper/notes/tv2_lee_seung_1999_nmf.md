# Paper notes — Lee & Seung (1999): Non-negative Matrix Factorization

## 1. Người cung cấp

- Thành viên: TV2 — Tiến.
- Phương pháp phụ trách: NMF/topic modeling.
- Ngày đọc: Chưa kiểm tra.

## 2. Citation info — Thông tin trích dẫn

- Citation key: `lee_seung_1999_nmf`.
- Tên đầy đủ của tài liệu: “Learning the parts of objects by non-negative
  matrix factorization.”
- Tác giả: Daniel D. Lee và H. Sebastian Seung.
- Năm xuất bản: 1999.
- Tạp chí/hội nghị hoặc nơi xuất bản: *Nature*, volume 401, trang 788–791.
- DOI hoặc URL nguồn gốc: <https://doi.org/10.1038/44565>.
- Loại tài liệu: `Scientific paper`.

## 3. Method — Phương pháp

- Bài toán và mục tiêu: phân tích biểu diễn cộng tính từ các nhân tử không âm.
- Phương pháp: Non-negative Matrix Factorization (NMF).
- Representation — cách biểu diễn dữ liệu: ma trận không âm; hướng và ý nghĩa
  các chiều phụ thuộc cách tổ chức dữ liệu đầu vào.
- Ý tưởng/cơ chế chính: xấp xỉ ma trận bằng tích của hai ma trận không âm.
  Bài báo thảo luận biểu diễn các bộ phận của đối tượng và có phần minh họa
  cho dữ liệu văn bản.

## 4. Dataset — Dữ liệu

- Tên/nguồn: bài báo có thí nghiệm nhận diện bộ phận trên ảnh khuôn mặt và
  minh họa xử lý văn bản; tên corpus văn bản: Chưa kiểm tra.
- Ngôn ngữ và miền nội dung: miền ảnh gồm khuôn mặt; ngôn ngữ và miền corpus
  văn bản: Chưa kiểm tra.
- Quy mô: Chưa kiểm tra.
- Cách chia dữ liệu/protocol — quy trình thí nghiệm — nếu có: Chưa kiểm tra.

## 5. Metric — Đánh giá

- Chỉ số và cách đánh giá: Chưa kiểm tra.
- Kết quả/phát hiện chính: bài báo trình bày cách phân rã không âm và minh họa
  các biểu diễn cộng tính. Kết quả định lượng và metric cụ thể: Chưa kiểm tra.
- Điều kiện đi kèm số liệu: dataset, split, cấu hình và baseline: Chưa kiểm tra.

Không dùng thí nghiệm ảnh làm bằng chứng NMF tốt hơn LDA trên tin tức tiếng
Việt hoặc thay cho đánh giá của Dataset V1.

## 6. Gap — Hạn chế hoặc khoảng trống

- Hạn chế/vấn đề chưa giải quyết: phạm vi áp dụng, độ nhạy cấu hình và so sánh
  NMF với mô hình chủ đề khác trong nghiên cứu này: Chưa kiểm tra.
- Do tác giả nêu hay người đọc nhận xét? Chưa kiểm tra.
- Bằng chứng và phạm vi của nhận định: chỉ có thể kết luận từ các thí nghiệm
  và dữ liệu được kiểm tra trong bài báo; vị trí cụ thể: Chưa kiểm tra.

## 7. Liên hệ với đồ án

- Tài liệu hỗ trợ phương pháp/thí nghiệm nào? M04 — NMF pipeline.
- Nhóm kế thừa hoặc khảo sát điều gì? Dùng NMF để phân rã ma trận TF-IDF không
  âm thành document-topic weights `W` và topic-word weights `H`.
- Khác biệt về dữ liệu/protocol cần lưu ý: NMF của đồ án chạy trên tin tức
  tiếng Việt sau preprocessing P3; cách sắp trục `W`/`H` theo code của nhóm
  cần được mô tả riêng. Bài báo không phải thực nghiệm trên Dataset V1.

Đây là liên hệ về phương pháp, không phải khẳng định hai triển khai dùng cùng
dữ liệu hoặc cùng protocol.

## 8. Vị trí bằng chứng

| Nhận định/số liệu | Section/trang/bảng/vị trí nguồn |
|---|---|
| Thông tin thư mục và mục tiêu bài báo | Trang *Nature* và PDF gốc; vị trí trong toàn văn: Chưa kiểm tra |
| Phân rã ma trận không âm | Phần mô tả thuật toán; section/trang: Chưa kiểm tra |
| Minh họa bộ phận khuôn mặt và dữ liệu văn bản | Kết quả/thí nghiệm; section, dataset và metric: Chưa kiểm tra |

Nguồn để đối chiếu: [trang Nature](https://www.nature.com/articles/44565) và
[PDF bài gốc](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/lee-nature-99.pdf).

## 9. Điều cần xác minh thêm

- Đọc toàn văn để xác nhận corpus, quy mô, protocol, metric và kết quả.
- Ghi section/trang cụ thể cho thuật toán và từng thí nghiệm.
- Xác nhận phần văn bản là thí nghiệm hay minh họa, và không suy diễn kết quả
  giữa miền ảnh với topic modeling tin tức.

## 10. Trạng thái và bàn giao

- Trạng thái: `Draft`.
- Link file hoặc PR bàn giao: Chưa tạo.
- Nhận xét của TV3 sau review: Chưa kiểm tra.
