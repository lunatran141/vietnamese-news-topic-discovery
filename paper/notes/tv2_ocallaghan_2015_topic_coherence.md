# Paper notes — O’Callaghan et al. (2015): Topic Coherence

## 1. Người cung cấp

- Thành viên: TV2 — Tiến.
- Phương pháp phụ trách: NMF/topic modeling.
- Ngày đọc: Chưa kiểm tra.

## 2. Citation info — Thông tin trích dẫn

- Citation key: `ocallaghan_2015_topic_coherence`.
- Tên đầy đủ của tài liệu: “An analysis of the coherence of descriptors in
  topic modeling.”
- Tác giả: Derek O’Callaghan, Derek Greene, Joe Carthy và Pádraig Cunningham.
- Năm xuất bản: 2015.
- Tạp chí/hội nghị hoặc nơi xuất bản: *Expert Systems with Applications*,
  volume 42, trang 5645–5657.
- DOI hoặc URL nguồn gốc:
  <https://researchrepository.ucd.ie/rest/bitstreams/18055/retrieve>.
- Loại tài liệu: `Scientific paper`.

## 3. Method — Phương pháp

- Bài toán và mục tiêu: phân tích coherence của các mô tả topic.
- Phương pháp: so sánh các mô tả topic và các phép đo coherence; phạm vi cấu
  hình NMF/LDA cụ thể: Chưa kiểm tra toàn văn.
- Representation — cách biểu diễn dữ liệu: descriptor/top terms của topic;
  cách biểu diễn và cách cân trọng số: Chưa kiểm tra.
- Ý tưởng/cơ chế chính: đánh giá mức coherence của descriptor bằng nhiều
  metric; định nghĩa chính xác từng metric: Chưa kiểm tra.

## 4. Dataset — Dữ liệu

- Tên/nguồn: ghi chú hiện có mô tả nhiều corpus, trong đó có corpus tin tức
  tiếng Anh; tên từng corpus: Chưa kiểm tra.
- Ngôn ngữ và miền nội dung: được ghi nhận có tin tức tiếng Anh; ngôn ngữ và
  miền của các corpus còn lại: Chưa kiểm tra.
- Quy mô: Chưa kiểm tra.
- Cách chia dữ liệu/protocol — quy trình thí nghiệm — nếu có: Chưa kiểm tra.

## 5. Metric — Đánh giá

- Chỉ số và cách đánh giá: dùng ba phép đo coherence theo ghi chú nguồn hiện
  có; tên và công thức của từng phép đo: Chưa kiểm tra toàn văn.
- Kết quả/phát hiện chính: hai trong ba phép đo được ghi nhận cho thấy NMF có
  coherence cao hơn trong các corpus khảo sát. Metric nào, corpus nào và bảng
  kết quả tương ứng: Chưa kiểm tra.
- Điều kiện đi kèm số liệu: cách cân trọng số, số topic, top-N, corpus và
  baseline: Chưa kiểm tra.

Không diễn giải kết quả này thành kết luận NMF luôn tốt hơn LDA hoặc áp dụng
trực tiếp sang Dataset V1.

## 6. Gap — Hạn chế hoặc khoảng trống

- Hạn chế/vấn đề chưa giải quyết: một metric riêng lẻ không đủ đại diện cho
  chất lượng diễn giải topic; nội dung chính xác và phạm vi tác giả nêu:
  Chưa kiểm tra.
- Do tác giả nêu hay người đọc nhận xét? Chưa kiểm tra.
- Bằng chứng và phạm vi của nhận định: cần đối chiếu abstract, phần metric và
  kết quả theo từng corpus; vị trí cụ thể: Chưa kiểm tra.

## 7. Liên hệ với đồ án

- Tài liệu hỗ trợ phương pháp/thí nghiệm nào? M05 — chọn số topic và đánh giá
  coherence/diversity; hỗ trợ cân nhắc review định tính.
- Nhóm kế thừa hoặc khảo sát điều gì? Kết hợp metric với top keywords, bài
  đại diện và phiếu đánh giá của người chấm khi lựa chọn K.
- Khác biệt về dữ liệu/protocol cần lưu ý: NMF của đồ án dùng TF-IDF P3 trên
  tin tức tiếng Việt; chỉ so các số coherence sau khi xác nhận công thức,
  tokenization, top-N và corpus tham chiếu tương đương.

Đây là liên hệ về thiết kế đánh giá; không trình bày như kết quả đồ án đã
được xác nhận.

## 8. Vị trí bằng chứng

| Nhận định/số liệu | Section/trang/bảng/vị trí nguồn |
|---|---|
| Mục tiêu và tổng quan coherence | Abstract; câu chữ và vị trí trang: Chưa kiểm tra |
| Ba metric và cách tính | Section về metric; tên metric, section/trang: Chưa kiểm tra |
| So sánh coherence NMF/LDA | Bảng kết quả theo corpus; số bảng/trang và điều kiện: Chưa kiểm tra |
| Dataset và protocol | Phần thực nghiệm; section/trang: Chưa kiểm tra |

Nguồn để đối chiếu: [bản tại kho tác giả](https://researchrepository.ucd.ie/rest/bitstreams/18055/retrieve)
và [trang nhà xuất bản](https://www.sciencedirect.com/science/article/pii/S0957417415001633).

## 9. Điều cần xác minh thêm

- Đọc toàn văn và ghi tên, công thức, chiều tốt/xấu của cả ba metric.
- Xác minh từng corpus, quy mô, split/protocol, số topic, top-N và baseline.
- Đối chiếu nhận định NMF cao hơn trên đúng metric/corpus/bảng; kiểm tra
  nghiên cứu có đánh giá thủ công hay không.
- Điền section/trang/bảng cho mọi nhận định trước khi chuyển trạng thái review.

## 10. Trạng thái và bàn giao

- Trạng thái: `Draft`.
- Link file hoặc PR bàn giao: Chưa tạo.
- Nhận xét của TV3 sau review: Chưa kiểm tra.
