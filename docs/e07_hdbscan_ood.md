# E07 — Đánh giá HDBSCAN trên dữ liệu tháng 10/2026

## Dữ liệu và quy trình

Tập dữ liệu độc lập gồm 5.136 bài từ VnExpress (1.865), Tuổi Trẻ
(1.898) và VietnamNet (1.373), có ngày xuất bản từ 01/10 đến
15/10/2026 theo giờ Việt Nam. Thời điểm thu thập, số bài theo ngày và
SHA-256 được lưu trong `data/ood/oct2026/manifest.json`.

Bộ mới đã được kiểm tra trùng article_id, URL và tiêu đề chuẩn hóa với
bộ gốc. Dữ liệu E07 không được dùng để chọn tham số hoặc fit lại mô hình.
Pipeline suy luận nạp MiniLM, UMAP 5D và HDBSCAN eom_15_5 đã chốt.

## Kết quả

- Tổng số bài: 5.136.
- Được gán topic: 3.205.
- Outlier (`-1`): 1.931.
- Coverage: 62,40%.
- Outlier rate: 37,60%.
- Số topic xuất hiện trong tập OOD: 100/101.
- ARI: 0,1895; NMI: 0,4135.

ARI/NMI dùng bảy nhãn `category_gold` cùng hệ với dữ liệu gốc và tính
trên cả bài `-1`. Đây là nhãn chuyên mục rộng, không phải nhãn đúng/sai
cho từng topic chi tiết.

## Ví dụ đọc thủ công

Các ví dụ dưới đây lấy từ `results/hdbscan/ood/ood_reviewed_cases.csv`
với `review_status=approved`. Nhận xét dựa trên nội dung bài báo, không
chỉ dựa vào tiêu đề.

### Gán topic hợp lý

- **Article ID:** `VNE_00383be52a24`
- **Tiêu đề:** Chứng khoán mất mốc 1.700 điểm
- **Topic được gán:** `94` — Thị trường chứng khoán và VN-Index.
- **Bằng chứng:** Bài cho biết VN-Index giảm hơn 6,5 điểm, đóng cửa ở
  1.698 điểm; đồng thời phân tích áp lực bán ở các cổ phiếu trụ, thanh
  khoản và hoạt động bán ròng của khối ngoại.
- **Nhận xét:** Nội dung tập trung vào diễn biến thị trường chứng khoán,
  phù hợp với topic 94.

### Cụm có phạm vi rộng

- **Article ID:** `TTR_096a819aaba9`
- **Tiêu đề:** Bé trai 6 tuổi mắc dị vật góc cạnh trong cổ suốt 6 tháng
- **Topic được gán:** `1` — Y tế và chăm sóc sức khỏe.
- **Bằng chứng:** Bài thuật ca phẫu thuật lấy dị vật gần 2 cm khỏi thực
  quản của một bệnh nhi. Trong cùng topic 1, bài `VNE_0e645966f432`
  lại tập trung vào số ca cúm A, viêm hô hấp và biện pháp phòng bệnh.
- **Nhận xét:** Cả hai bài đều thuộc lĩnh vực y tế, nên việc gán vào
  topic 1 có cơ sở. Tuy nhiên, một bài là ca phẫu thuật riêng lẻ, bài
  còn lại là khuyến cáo phòng dịch; điều này cho thấy topic 1 bao phủ
  nhiều vấn đề y tế khác nhau.

### Bài bị xếp outlier nhưng có chủ đề rõ

- **Article ID:** `VNE_02d57a473b01`
- **Tiêu đề:** GDP quý III tăng 9,95%
- **Topic được gán:** `-1` — outlier.
- **Bằng chứng:** Bài dẫn số liệu của Cục Thống kê: GDP quý III tăng
  9,95% so với cùng kỳ, GDP chín tháng tăng 9,01%, rồi phân tích đóng
  góp của công nghiệp, xây dựng và dịch vụ. Bài
  `TTR_a73b15048ba0` cũng đưa tin về cùng số liệu GDP và được gán `-1`.
- **Nhận xét:** Đây là tin kinh tế có nội dung rõ ràng, không phải bài
  thiếu chủ đề. Nhãn `-1` cho biết HDBSCAN không đưa bài vào một cụm đã
  hình thành; riêng ví dụ này chưa đủ để kết luận việc xếp outlier là
  sai.
