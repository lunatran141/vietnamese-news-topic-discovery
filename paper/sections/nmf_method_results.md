# TV2 — NMF: Methodology, Results và Discussion (bản nháp R02)

## Phương pháp

Chúng tôi dùng Dataset V1 gồm 9.274 bài báo sau khử trùng lặp. Biến thể P3
được chọn theo manifest dữ liệu: chuẩn hóa, tách từ tiếng Việt bằng PyVi rồi
loại stopword. Đầu vào `text` gồm tiêu đề gốc và nội dung đã tiền xử lý.
Các bài được nối với split cố định bằng `article_id`: 6.493 train, 1.391
validation và 1.390 test. Chi tiết snapshot và SHA-256 nằm trong
`docs/nmf_data_review.md`.

Mỗi bài được biểu diễn bằng TF-IDF unigram/bigram (`min_df=3`, `max_df=0.95`,
tối đa 50.000 feature, sublinear TF). NMF xấp xỉ ma trận bài–từ không âm
`X ≈ W H`; `W[d,k]` là trọng số chủ đề k ở bài d, `H[k,v]` là trọng số từ v
ở chủ đề k. Khi hiển thị nhiều chủ đề của một bài, hàng W được chia cho tổng
trọng số của hàng. Các giá trị này là **trọng số tương đối**, không phải
xác suất nhãn đúng. Mô hình dùng `nndsvda`, solver `cd`, Frobenius loss,
`max_iter=500` và seed 42. NMF là hướng phân rã cộng tính theo
[Lee và Seung (1999)](https://www.nature.com/articles/44565).

Chúng tôi fit các ứng viên K∈{4,6,8,10,12} trên train, tính NPMI từ sự đồng
xuất hiện của 10 từ đứng đầu mỗi topic trên validation, cùng topic diversity
và lỗi tái tạo chuẩn hóa. NPMI ở đây là triển khai nội bộ ở mức document,
vì vậy khi so với phương pháp khác phải xác nhận cùng mã tính metric. Việc
đọc top words và bài đại diện là bước đánh giá bổ sung, vì coherence một
thước đo không thay thế hoàn toàn khả năng hiểu của người đọc
([O’Callaghan et al., 2015](https://researchrepository.ucd.ie/rest/bitstreams/18055/retrieve)).

## Kết quả chọn K trên validation

| K | NPMI ↑ | Diversity ↑ | Lỗi tái tạo chuẩn hóa ↓ |
|---:|---:|---:|---:|
| 4 | 0,3583 | 0,9750 | 0,9727 |
| 6 | 0,3738 | 1,0000 | 0,9675 |
| 8 | 0,4293 | 1,0000 | 0,9633 |
| **10** | **0,4493** | **1,0000** | 0,9595 |
| 12 | 0,4403 | 0,9917 | **0,9568** |

Nguồn: `results/nmf/nmf_p3_select_review_seed42_v1/topic_count_metrics.csv`. K=10
có NPMI cao nhất, trong khi K=12 có lỗi tái tạo thấp hơn. Đây là **đề xuất
định lượng**; run mới đã xuất 40 topic và 200 bài đại diện validation, nhưng
quyết định cuối cần ghi lại điểm review của ít nhất hai người. Run K=10 hiện
có được fit lại trên
train+validation rồi mới đánh giá trên test.

## Đánh giá test và diễn giải

Run `nmf_p3_k10_seed42_v1` đạt NPMI 0,4407, diversity 1,0000, lỗi tái tạo
chuẩn hóa 0,9590. ARI 0,5391 và NMI 0,6514 chỉ đối chiếu chủ đề chính với
`category_gold` (chuyên mục báo). Các chuyên mục không là ground truth topic;
không nên gọi ARI/NMI là độ chính xác phân loại. Từ khóa gợi ra các nhóm như
điều tra/pháp luật, sức khỏe, giáo dục, giải trí, thị trường tài chính. Một số
topic theo sự kiện hoặc thương hiệu (Asiad, ASEAN Cup, iPhone), nên cần kiểm
tra độ bền và mức độ chồng lấn trước khi đặt nhãn chính thức.

Ngưỡng chủ đề phụ ≥0,20 gắn cờ 523/1.390 bài test để sàng lọc; bảng 50 bài
ưu tiên trường hợp hai share gần nhau. Đây không phải 523 bài đa chủ đề đã
được người xác nhận. `results/nmf/nmf_p3_k10_seed42_v1/multi_topic_case_studies.md`
ghi ví dụ đúng, mơ hồ và nghi vấn gán sai. Phần đánh giá bằng người đang chờ
TV2 và reviewer độc lập đọc toàn văn.

## Giới hạn và so sánh cần hoàn thiện

- K=10 hiện được chọn theo NPMI; phiếu đánh giá khả năng hiểu cho từng K chưa
  được người chấm hoàn tất.
- NMF học trên TF-IDF P3; LDA có đầu vào đếm từ và thường dùng P2 theo manifest.
  Do đó khi so với LDA phải ghi cả biểu diễn đầu vào, split, cách tính metric,
  runtime và cùng tập `article_id`. Chưa có run LDA trong nhánh hiện tại nên
  chưa thể đưa bảng thắng/thua định lượng.
- Một chủ đề theo sự kiện hoặc sản phẩm có thể hữu ích cho demo nhưng kém bền
  khi tin mới đổi chủ đề. Cần đối chiếu stability/OOD của TV3 và outlier/event
  của TV4 trước khi kết luận chung.
- Kết quả test đã được dùng để mô tả case sau khi chốt mô hình. Không dùng
  các nhận xét hậu kiểm này để đổi K/ngưỡng rồi báo cáo lại cùng test như một
  phép đánh giá độc lập.
