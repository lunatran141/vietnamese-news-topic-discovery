# TV2 — Case study NMF trên tập test (M06)

Nguồn: `multi_topic_cases.csv` của run `nmf_p3_k10_seed42_v1`.
`primary_topic_share` và `secondary_topic_share` là trọng số W **chuẩn hóa
theo từng bài**, không phải xác suất nhãn đúng. Ngưỡng chủ đề phụ ≥0,20 chỉ
chọn ứng viên để đọc. Các nhận xét dưới đây đã đối chiếu **toàn văn cột
`content` của Dataset V1** cho đúng năm `article_id`; chúng vẫn cần một
reviewer độc lập xác nhận trước khi dùng như nhãn đánh giá chính thức.

| `article_id` | Hai topic và share | Nhận xét sơ bộ | Kiểm tra tiếp |
|---|---|---|---|
| `VNN_93c346ff1800` | NMF_02 0,4141; NMF_05 0,4096 | **Giao thoa có căn cứ**: toàn văn nói về DEA khám xét, điều tra cái chết và hồ sơ diễn xuất của nữ diễn viên. Hai mảng điều tra và giải trí đều hiện diện. | Reviewer xác nhận liệu cả hai là chủ đề chính hay NMF_05 chỉ là thuộc tính nhân vật. |
| `TTR_4a36d0c749d2` | NMF_04 0,3641; NMF_03 0,3636 | **Giao thoa nhưng lệch trọng tâm**: chương trình đào tạo bác sĩ được nhắc nhiều, song toàn văn tập trung vào năng lực y tế cơ sở và chăm sóc sức khỏe. | Reviewer đánh giá liệu NMF_04 có đáng là topic thứ hai hay chỉ là bối cảnh đào tạo. |
| `TTR_10e13df4da6f` | NMF_03 0,2092; NMF_02 0,2084 | **Nghi vấn gán sai hai nhãn**: toàn văn về vệ sinh thực phẩm ở chợ và xử lý hành chính một quầy. Không phải tin điều trị bệnh hay điều tra hình sự. | Đối chiếu top words để xác định nhiễu từ “an toàn”, “vi phạm”, “kiểm tra”. |
| `TTR_2a44c1c095ed` | NMF_00 0,3083; NMF_05 0,3083 | **Nghi vấn topic giải trí do tên người/chương trình**: toàn văn chủ yếu giới thiệu quy trình sản xuất, thương hiệu và xuất khẩu nước mắm; MasterChef là người trải nghiệm/quảng bá. | Reviewer quyết định đây là bài kinh doanh/ẩm thực hay thật sự có chủ đề giải trí độc lập. |
| `VNE_1e32ccbe78ad` | NMF_02 0,2250; NMF_00 0,2237 | **Gán sai rõ ở NMF_02**: toàn văn về tuyển tài xế Green SM, cọc và thu nhập; không có điều tra/pháp luật. | Kiểm tra đóng góp từ vựng khiến bài kéo về NMF_02. |

## Phân loại lỗi sơ bộ cho TV4/nhóm

- **Tin thật sự giao thoa**: một bài có hai mảng nội dung cùng đáng kể.
- **Từ khóa chung gây nhiễu**: một vài từ của topic xuất hiện trong ngữ cảnh
  khác, ví dụ “tài xế”.
- **Nhãn topic quá hẹp hoặc theo sự kiện**: NMF_06/NMF_07/NMF_08 có thể phản ánh
  Asiad, ASEAN Cup, iPhone hơn là nhóm chủ đề bền vững.
- **Nhận xét một người chưa đủ**: reviewer độc lập ghi bằng chứng, quyết định
  và tên người review vào bản cuối; bất đồng cần giữ lại để báo cáo.

Không dùng phân tích trên test để thay K=10 hoặc ngưỡng 0,20. Nếu cần thay đổi
mô hình, thiết kế thử nghiệm mới trên train/validation rồi đánh giá độc lập.
