# Paper notes — Mẫu nghiên cứu liên quan cho R06

## A. Cách sử dụng và đặt tên

1. Giữ file mẫu này trống. Sao chép thành một file riêng cho mỗi tài liệu.
2. Lưu notes trong `paper/notes/`.
3. Tên file: `tv<so_thanh_vien>_<citation_key>.md`.
4. Citation key: `<ho_tac_gia_dau>_<nam>_<tu_khoa_ngan>`.
   Dùng chữ thường, không dấu, không khoảng trắng;
   phân cách bằng `_`.
5. Ví dụ: `tv3_blei_2003_lda.md`.
   Citation key tương ứng: `blei_2003_lda`.
   Đây là ví dụ đặt tên, không phải notes đã đọc.
6. Thay các phần trong dấu `< >` bằng giá trị thật;
   không dùng dấu `< >` trong tên file.
7. Cùng một tài liệu dùng cùng citation key, dù nhiều người đọc.
   Hai tài liệu khác nhau trùng mã thì thêm từ khóa phân biệt;
   TV3 thống nhất mã cuối cùng.
8. Viết bằng lời người đọc; không chép nguyên abstract.
   Không điền số liệu hoặc hạn chế chưa kiểm tra.
9. Mục nguồn không báo cáo ghi `Không được báo cáo`;
   mục chưa đọc/xác minh ghi `Chưa kiểm tra`.
   Hai trạng thái này khác nhau.
10. Mọi số liệu và nhận định quan trọng cần
    section/trang/bảng hoặc vị trí bằng chứng tương đương.

## B. Phân công theo sheet

| Thành viên | Hướng tài liệu |
|---|---|
| TV1 — Khôi | TF-IDF, K-Means, preprocessing |
| TV2 — Tiến | NMF/topic modeling |
| TV3 — Tuấn | LDA/topic modeling; tổng hợp R06 |
| TV4 — Vân | Embeddings, HDBSCAN, event detection |
| TV5 — Ngân | BERTopic/topic representation |

TV1/TV2/TV4/TV5 có nhiệm vụ tìm 1–2 scientific papers
theo sheet của mình.

TV3 cung cấp nguồn liên quan LDA và tổng hợp notes TV1–TV5.

Yêu cầu ít nhất 8 tài liệu khác nhau, ít nhất 4 scientific papers
là của toàn nhóm; không phải mỗi người.

Nguồn trùng chỉ tính một lần.
TV3 đối chiếu thêm yêu cầu môn học nếu có.

## 1. Người cung cấp

- Thành viên:
- Phương pháp phụ trách:
- Ngày đọc:

## 2. Citation info — Thông tin trích dẫn

- Citation key:
- Tên đầy đủ của tài liệu:
- Tác giả:
- Năm xuất bản:
- Tạp chí/hội nghị hoặc nơi xuất bản:
- DOI hoặc URL nguồn gốc:
- Loại tài liệu: `Scientific paper` / `Official documentation` / `Other`.

Giải thích:

- Scientific paper: bài báo khoa học.
- Official documentation: tài liệu chính thức.
- Other: nguồn khác.

README/tài liệu thư viện không tự được tính là scientific paper.
TV3 kiểm tra loại nguồn trước khi thống kê.

## 3. Method — Phương pháp

- Bài toán và mục tiêu:
- Phương pháp:
- Representation — cách biểu diễn dữ liệu:
- Ý tưởng/cơ chế chính:

## 4. Dataset — Dữ liệu

- Tên/nguồn:
- Ngôn ngữ và miền nội dung:
- Quy mô:
- Cách chia dữ liệu/protocol — quy trình thí nghiệm — nếu có:

## 5. Metric — Đánh giá

- Chỉ số và cách đánh giá:
- Kết quả/phát hiện chính:
- Điều kiện đi kèm số liệu: dataset, split, cấu hình, baseline nếu cần:

Không so trực tiếp các số liệu thuộc dataset/protocol khác nhau
như cùng một thí nghiệm.

## 6. Gap — Hạn chế hoặc khoảng trống

- Hạn chế/vấn đề chưa giải quyết:
- Do tác giả nêu hay người đọc nhận xét?
- Bằng chứng và phạm vi của nhận định:

Phân biệt “paper này chưa khảo sát” với “chưa ai khảo sát”.
Không đưa khẳng định rộng khi chưa có đủ bằng chứng.

## 7. Liên hệ với đồ án

- Tài liệu hỗ trợ phương pháp/thí nghiệm nào?
- Nhóm kế thừa hoặc khảo sát điều gì?
- Khác biệt về dữ liệu/protocol cần lưu ý:

Đây là hướng liên hệ dự kiến;
không trình bày như kết quả nhóm đã thực nghiệm.

## 8. Vị trí bằng chứng

| Nhận định/số liệu | Section/trang/bảng/vị trí nguồn |
|---|---|
| | |

## 9. Điều cần xác minh thêm

- ...

## 10. Trạng thái và bàn giao

- Trạng thái: `Draft` / `Ready for TV3 review`.
- Link file hoặc PR bàn giao:
- Nhận xét của TV3 sau review:

Giải thích trạng thái:

- Draft: đang viết.
- Ready for TV3 review: sẵn sàng để TV3 kiểm tra.

Thành viên commit/push notes trên branch của mình
và cung cấp link file/PR cho TV3 theo workflow nhóm.

Không cùng sửa template; không push trực tiếp vào branch
của người khác.

Nếu template chưa merge vào develop, có thể sao chép
từ branch R06 trên GitHub.

TV3 kiểm tra nguồn, nội dung và trùng lặp trước khi đưa vào
`paper/related_work_matrix.csv`.

Template là tài liệu nội bộ hỗ trợ.
Hai output chính của R06 là matrix và section draft.