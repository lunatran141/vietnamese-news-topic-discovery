# Related Work — Nghiên cứu liên quan

## Trạng thái bản nháp R06

Đây là khung nội bộ; chưa phải phần tổng hợp nghiên cứu
hoàn chỉnh.

Chưa có nội dung paper notes được kiểm tra và đưa vào
bản nháp này.

Chưa thể xác nhận đủ ít nhất 8 tài liệu khác nhau,
trong đó ít nhất 4 scientific papers.

Đầu vào: notes từ TV1–TV5 trong `paper/notes/`.

Bảng tổng hợp: `paper/related_work_matrix.csv`.

## 2.1 Classical clustering — Phân cụm truyền thống

Nguồn: TV1 về TF-IDF, K-Means và preprocessing;
TV1 phối hợp TV3.

Nội dung cần viết sau khi kiểm tra nguồn:

- Tổng hợp cách biểu diễn từ vựng và hướng phân cụm truyền thống.
- So sánh method, dataset, metric và các phát hiện có căn cứ.
- Phân tích hạn chế, tác động preprocessing và liên hệ thí nghiệm của nhóm.
- Mỗi nhận định lấy từ tài liệu có trích dẫn phù hợp.

Phần văn bản tổng hợp: Chưa viết; chờ nguồn đã review.

## 2.2 Topic Modeling — Mô hình chủ đề

Nguồn: TV2 về NMF và TV3 về LDA; TV2 phối hợp TV3.

Nội dung cần viết:

- Tổng hợp hướng matrix factorization — phân rã ma trận —
  và probabilistic topic modeling — mô hình chủ đề xác suất.
- So sánh representation, document–topic/topic–word outputs
  và phạm vi nghiên cứu.
- Phân tích đánh giá, interpretability, stability hoặc
  các khía cạnh được nguồn thực sự khảo sát.
- Liên hệ với việc so NMF/LDA của nhóm;
  không suy ra kết quả trước khi chạy thí nghiệm.

Phần văn bản tổng hợp: Chưa viết; chờ nguồn đã review.

## 2.3 Semantic Topic Discovery — Khám phá chủ đề bằng ngữ nghĩa

Nguồn: TV4 về sentence embeddings/HDBSCAN và TV5 về BERTopic;
TV3 phối hợp TV4/TV5.

Nội dung cần viết:

- Tổng hợp hướng dùng biểu diễn ngữ nghĩa và cơ chế tạo topic.
- So sánh các phát hiện, dữ liệu và cách đánh giá của nguồn liên quan.
- Phân tích hạn chế và gap liên quan dữ liệu tiếng Việt
  trong phạm vi có bằng chứng.
- Liên hệ với các phương pháp/thí nghiệm của project;
  không khẳng định “chưa có ai làm” nếu chưa khảo sát đủ.

Phần văn bản tổng hợp: Chưa viết; chờ nguồn đã review.

## Hướng dẫn tổng hợp của TV3 — nội bộ

1. Nhận notes riêng cho từng tài liệu,
   có citation info và method–dataset–metric–gap.
2. Mở nguồn gốc, kiểm tra nhận định quan trọng;
   yêu cầu bổ sung mục thiếu.
3. Hợp nhất nguồn trùng bằng citation key;
   một tài liệu chỉ tính một lần.
4. Điền một hàng matrix cho mỗi nguồn đã kiểm tra.
   Không thêm hàng minh họa như kết quả thật.
5. Viết ba mục trên thành các đoạn liên kết:
   hướng nghiên cứu → nguồn hỗ trợ/so sánh → hạn chế → liên hệ project.
   Không dán nguyên notes hoặc chỉ liệt kê paper.
6. Thống nhất kiểu citation theo yêu cầu môn học/nhóm;
   citation key là mã quản lý, không tự là kiểu trích dẫn chính thức.
7. Đối chiếu References:
   nguồn trích trong phần viết có trong danh sách và ngược lại.
   Nếu chưa chốt định dạng, dùng placeholder rõ ràng trong nháp;
   không để lại trong bản nộp cuối.
8. Kiểm tra tính so sánh của dataset/metric;
   không xếp hạng phương pháp bằng số liệu
   từ điều kiện không tương đương.

Header của matrix:

```csv
citation_key,owner,title,authors,year,venue,doi_or_url,document_type,method,dataset,metric,key_findings,gap,relevance_to_project,evidence_location
```

Quy tắc điền matrix:

- Một hàng là một nguồn khác nhau.
- `owner` ghi người cung cấp notes, có thể nhiều người.
- `gap` phân biệt hạn chế tác giả nêu với nhận xét người đọc.
- `evidence_location` chỉ rõ nơi kiểm tra.
- Giữ CSV UTF-8.
- Ô chứa dấu phẩy, dấu nháy hoặc xuống dòng
  phải được xử lý đúng quy tắc CSV.
- Ưu tiên nội dung ngắn để so sánh.
- Matrix và notes dùng cùng citation key.
- Quy định tên file nằm trong
  `paper/notes/related_work_note_template.md`.

## Checklist hoàn thành R06 — nội bộ

- [ ] Có ít nhất 8 tài liệu khác nhau sau loại trùng.
- [ ] Có ít nhất 4 scientific papers, loại nguồn đã được kiểm tra.
- [ ] Đã đối chiếu yêu cầu môn học bổ sung nếu được cung cấp.
- [ ] Matrix có paper–method–dataset–metric–gap và citation info kiểm tra được.
- [ ] Ba mục 2.1–2.3 có văn bản tổng hợp, so sánh và liên hệ project.
- [ ] Citation thống nhất, khớp References; không còn placeholder trong bản nộp.
- [ ] Không có số liệu/claim bịa hoặc copy nguyên văn chưa xử lý.
- [ ] Có link PR/artifact bàn giao theo quy ước nhóm.

Trước khi đưa vào paper nộp chính thức, chuyển các hướng dẫn
và checklist nội bộ sang tài liệu quản lý,
giữ lại văn bản nghiên cứu đã viết.

Việc tạo bộ khung hoặc push lên GitHub
chưa có nghĩa R06 hoàn thành.