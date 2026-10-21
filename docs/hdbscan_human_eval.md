# Đánh giá định tính HDBSCAN trên tập validation

## 1. Mục tiêu và phạm vi

Phần này đọc bài báo để đánh giá khả năng diễn giải các cụm HDBSCAN và mức phù hợp giữa từng bài với cụm được gán. Cấu hình được khảo sát là `minilm_title_content + eom_15_5`, sử dụng kết quả dự đoán trên 1.391 bài validation.

Đây là đánh giá định tính trên **5 cụm và 25 bài mẫu**, không phải điểm chất lượng của toàn bộ 1.391 bài. Mẫu không bao gồm các bài có nhãn nhiễu `-1`.

## 2. Cách tạo bộ mẫu

Script `experiments/human_eval/build_hdbscan_human_eval_samples.py` sử dụng:

- Nhãn validation trong `results/hdbscan/validation/minilm_title_content_eom_15_5_val_assignments.csv`.
- Từ khóa cụm trong `artifacts/common/hdbscan/minilm_title_content_eom_15_5/topic_summary.csv`.
- Tiêu đề, nội dung và URL bài báo trong `data/processed/p0_minimal.csv`.
- Danh sách ID validation trong `data/splits/split_manifest.csv`.

Script kiểm tra rằng assignments chứa đúng 1.391 `article_id` của validation. Sau đó, script loại các bài có `cluster_id = -1` và chỉ xét các cụm có ít nhất 5 bài. Năm cụm được chọn tại các vị trí trải từ cụm nhỏ đến cụm lớn sau khi sắp theo kích thước; cụm không được chọn dựa trên điểm đánh giá.

Với mỗi cụm, script chọn **3 bài có `membership_strength` cao nhất** và **2 bài ngẫu nhiên trong số các bài còn lại**. Việc lấy mẫu và xáo thứ tự hiển thị dùng seed `42`. Tổng cộng có **5 cụm × 5 bài = 25 cặp bài–cụm**.

Các cụm được mã hóa thành `T01`–`T05`; bài được mã hóa thành `HS001`–`HS025`. Phiếu đọc bài hiển thị từ khóa, tiêu đề, nội dung và URL. `topic_mapping.csv` cùng `evaluation_samples.csv` là các file quản lý bộ mẫu.

## 3. Phiếu và tiêu chí chấm

Mỗi bộ phiếu gồm:

- `reviewer_R1_topics.csv` hoặc `reviewer_R2_topics.csv`: chấm cấp cụm, đặt nhãn đề xuất, chọn một bài phù hợp và một bài khó, ghi lý do.
- `reviewer_R1_documents.csv` hoặc `reviewer_R2_documents.csv`: chấm mức phù hợp bài–cụm, ghi `error_tag` và lý do theo từng bài.

Năm tiêu chí cấp cụm được chấm bằng số nguyên từ 1 đến 5:

1. **Nhất quán:** các bài có cùng một chủ đề hay vụ việc rõ ràng không.
2. **Cụ thể:** chủ đề có đủ hẹp để phân biệt với chủ đề khác không.
3. **Đặt tên:** từ khóa và bài mẫu có hỗ trợ một tên chủ đề chính xác không.
4. **Hữu ích:** cụm có giúp người đọc khám phá tin tức không.
5. **Ranh giới:** cụm có tránh gộp bài thuộc chủ đề khác không.

Ở cấp bài, `document_relevance_1_5` đo mức phù hợp của từng bài với nhãn chủ đề được đề xuất.

**Nguồn phiếu:** R1 là phiếu do thành viên nhóm đọc và chấm. R2 là bản chấm của thành viên trong nhóm.

## 4. Kết quả điểm số

Script `experiments/human_eval/summarize_hdbscan_human_eval.py` đã kiểm tra và tổng hợp đủ **5 cụm** cùng **25 cặp bài–cụm**. Kết quả được lưu trong `results/hdbscan/human_eval/summary/`.

| Tiêu chí | Trung bình R1 | Trung bình R2 | Trùng điểm chính xác |
|---|---:|---:|---:|
| Nhất quán | 3,00 | 3,00 | 100% |
| Cụ thể | 3,40 | 3,20 | 80% |
| Đặt tên | 3,60 | 2,80 | 20% |
| Hữu ích | 3,60 | 3,20 | 60% |
| Ranh giới | 2,80 | 2,20 | 40% |
| Mức phù hợp bài–cụm | 3,92 | 3,88 | 88% |

Nếu lấy trung bình của hai bộ phiếu và cho mỗi cụm trọng số bằng nhau, `method_summary.csv` ghi: nhất quán **3,0**, cụ thể **3,3**, đặt tên **3,2**, hữu ích **3,4**, ranh giới **2,5**, và mức phù hợp bài–cụm **3,9** trên thang 5. Đây là các thống kê mô tả của **bộ mẫu đã chọn**, không phải ước lượng cho toàn bộ bài validation.

Hai bộ phiếu cho cùng điểm ở **22/25 bài**. Ba bài lệch 1 điểm là:

- **HS016 — `VNE_29086a9d7dfc`:** R1 chấm 5, R2 chấm 4. Bài có trọng tâm đáng kể về giá vàng nhưng cũng bàn về chứng khoán Mỹ sau quyết định lãi suất của Fed.
- **HS022 — `VNE_b5cf07f3bb2d`:** R1 chấm 5, R2 chấm 4. Bài liên quan điều trị ung thư và bảo hiểm y tế, đồng thời nhấn mạnh gánh nặng tài chính của gia đình.
- **HS023 — `TTR_02e6c53cad15`:** R1 chấm 2, R2 chấm 3. Đây là tin về lễ kỷ niệm bệnh viện và thành tựu y khoa; bài thuộc lĩnh vực y tế nhưng khác thể loại với bài tư vấn hoặc điều trị bệnh.

`disagreements.csv` không có dòng dữ liệu vì script chỉ đưa vào file này các lượt chấm lệch **từ 2 điểm trở lên**. Kết quả này không có lượt nào đạt ngưỡng đó. Vẫn có các lượt lệch 1 điểm: **10 lượt ở tiêu chí cấp cụm và 3 lượt ở cấp bài**.

## 5. Phân tích các cụm và ca lỗi

### T01 — iPhone Duo và các tin Apple khác

HS003 (`VNN_3b82dc68b42e`) và HS005 (`VNN_6033127b1aef`) nói trực tiếp về iPhone Duo màn hình gập. Ngược lại, HS001 (`VNE_4e84cfb652e9`) và HS004 (`VNN_f0d233a65f6a`) cùng đưa tin một vụ kiện sáng chế Taptic Engine; HS002 nói về tính năng iPhone Handoff trên iOS 27.

Cụm có trọng tâm dễ nhận ra nhưng gộp thêm bài do cùng nhắc Apple hoặc iPhone. HS001 và HS004 còn là hai bài từ các nguồn khác nhau về gần như cùng một sự kiện. Lỗi cần ghi nhận là **trùng thực thể chung nhưng khác chủ đề cụ thể**.

### T02 — Hai vụ án phúc thẩm độc lập

Năm bài đều liên quan đến xét xử phúc thẩm, nhưng chúng thuộc hai vụ việc: vụ cựu hiệu trưởng Nguyễn Thị Bình và vụ cựu bộ trưởng Nguyễn Thị Kim Tiến. Vì thế, từng bài vẫn phù hợp với nhãn rộng “các vụ án phúc thẩm”, trong khi tính nhất quán ở mức **một vụ việc** thấp.

Tag `cụm gộp nhiều vụ việc` mô tả lỗi **cấp cụm**; nó không có nghĩa cả năm bài đều bị gán sai khỏi chủ đề pháp lý. Đây là ví dụ về sự khác biệt giữa mức phù hợp từng bài và độ cụ thể của một cụm.

### T03 — Phát triển đô thị và hạ tầng

HS011 và HS015 liên quan đến Luật Phát triển đô thị; HS012 và HS014 liên quan đến metro hoặc đường sắt đô thị. HS013 (`TTR_42079ce07920`) là bài phỏng vấn rộng về phát triển và quản trị Cần Thơ, nên chỉ liên quan gián tiếp. Một số cụm từ khóa như “triển đô”, “luật phát”, “sắt đô” cũng khó đọc nếu tách khỏi ngữ cảnh.

### T04 — Biến động giá vàng

Cả năm bài đều có nội dung rõ về giá vàng trong nước hoặc thế giới. Đây là cụm dễ đặt tên và có mức phù hợp bài–cụm cao trong bộ mẫu. HS016 là ca cần đọc kỹ vì ngoài giá vàng, bài còn bàn đáng kể về chứng khoán Mỹ. Sự khác biệt một điểm ở HS016 phản ánh cách xác định trọng tâm của một bài có nhiều mảng thị trường.

### T05 — Nhóm y tế và sức khỏe quá rộng

Các bài mẫu trải từ tư vấn sức khỏe, chi phí điều trị ung thư đến sự kiện bệnh viện. HS024 (`TTR_07f28315c701`) chủ yếu là tin xã hội về trẻ sơ sinh bị bỏ rơi; chi tiết bệnh viện tiếp nhận không biến trọng tâm bài thành tư vấn hoặc điều trị y tế. Cả hai bộ phiếu đều chấm HS024 ở mức 1.

Phần cuối `content` của HS024 còn xuất hiện một mẩu tin khác về trẻ bị bỏ rơi ở Bình Phước. Đây là **nghi vấn nội dung nguồn bị nối lẫn**, cần kiểm tra riêng trước khi kết luận nguyên nhân dữ liệu. T05 cho thấy một nhãn rộng như “y tế và sức khỏe” có thể bao quát nhiều bài, nhưng kém hữu ích khi cần khám phá một chủ đề cụ thể.

## 6. Tổng hợp lỗi và giới hạn

Các ca mẫu minh họa những kiểu lỗi sau:

- **Trùng thực thể, khác trọng tâm:** HS001 và HS004 cùng nhắc Apple nhưng không nói về iPhone Duo.
- **Gộp nhiều vụ việc:** T02 chứa hai vụ án phúc thẩm độc lập.
- **Chủ đề rộng, bài chỉ liên quan gián tiếp:** HS013 và HS021.
- **Khác thể loại trong cùng lĩnh vực:** HS023 là tin sự kiện bệnh viện.
- **Lạc chủ đề và nghi vấn chất lượng nội dung:** HS024.

Cần đọc các số liệu với những giới hạn sau:

1. Chỉ khảo sát một cấu hình HDBSCAN trên tập validation và 5 cụm được chọn theo kích thước. Bộ mẫu không đại diện cho toàn bộ các cụm.
2. Mỗi cụm có 3 bài ưu tiên theo `membership_strength` và 2 bài lấy ngẫu nhiên. Mẫu ưu tiên bài có độ mạnh thành viên cao, nên có thể thể hiện cụm rõ hơn so với việc lấy toàn bộ bài ngẫu nhiên.
3. Không chấm các bài nhãn `-1` trong thí nghiệm này. Phân tích outlier riêng nằm ở M12.
4. Điểm cấp cụm và điểm cấp bài cần được đọc cùng lý do và `article_id`. Điểm trung bình không cho biết đầy đủ vì sao một cụm bị gộp nhiều vụ việc hoặc vì sao một bài nằm sát ranh giới chủ đề.

## 7. File kết quả

Các phiếu và kết quả được lưu tại `results/hdbscan/human_eval/`:

- `reviewer_R1_topics.csv`, `reviewer_R1_documents.csv`: phiếu R1.
- `reviewer_R2_topics.csv`, `reviewer_R2_documents.csv`: phiếu R2.
- `evaluation_samples.csv`, `topic_mapping.csv`: thông tin quản lý mẫu.
- `summary/topic_scores.csv`: điểm từng tiêu chí cấp cụm của R1 và R2.
- `summary/document_scores.csv`: điểm từng bài của R1 và R2.
- `summary/agreement_summary.csv`: mức trùng điểm và chênh lệch trung bình giữa hai phiếu.
- `summary/disagreements.csv`: chỉ liệt kê trường hợp lệch từ 2 điểm trở lên; file hiện không có dòng dữ liệu.
- `summary/method_summary.csv`: điểm trung bình mô tả trên năm cụm mẫu.

Lệnh tổng hợp đã sử dụng:

```powershell
python -B -m experiments.human_eval.summarize_hdbscan_human_eval
```
