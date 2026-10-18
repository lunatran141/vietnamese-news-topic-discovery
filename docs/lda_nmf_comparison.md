
# Task 3.7 — LDA vs NMF Comparison

## 1. Mục tiêu

So sánh hai pipeline Topic Modeling: Latent Dirichlet Allocation
(LDA) và Non-negative Matrix Factorization (NMF) trên Dataset v1.0.

Việc so sánh tập trung vào:

- Topic Coherence.
- Topic Diversity.
- ARI và NMI.
- Khả năng diễn giải từ khóa.
- Bài báo đại diện cho từng topic.
- Những điểm khác biệt giữa hai mô hình.

## 2. Dữ liệu và giao thức

Cả hai phương pháp sử dụng Dataset v1.0 và cùng data/splits/split_manifest.csv.

Train: 6.493 bài.
Validation: 1.391 bài.
Test: 1.390 bài.

### LDA

- Preprocessing: p2_segmented.
- Representation: Bag-of-Words.
- K đề xuất: 15.
- Seed: 42.
- Nguồn lựa chọn K: M08.
- Nguồn stability: M09.
- Final Test: Chưa thực hiện.

### NMF

- Preprocessing: p3_no_stopwords.
- Representation: TF-IDF.
- K: 10.
- Seed: 42.
- Nguồn kết quả: TV2 M04-M06.
- Final Test: Đã có kết quả.

## 3. Quy trình thực nghiệm

K được lựa chọn dựa trên Train/Validation.

Mô hình cuối được huấn luyện trên Train+Validation và chỉ đánh giá trên Test sau khi cấu hình đã chốt. Test không được dùng để lựa chọn K hoặc tuning.

## 4. So sánh định lượng

| Metric | LDA K=15 | NMF K=10 |
|---|---:|---:|
| NPMI (document-level) | Chưa tính | 0.440680 |
| Topic Diversity | Chưa tính | 1.000000 |
| ARI | Chưa tính | 0.539079 |
| NMI | Chưa tính | 0.651373 |

Kết quả NMF được lấy từ results/nmf/nmf_p3_k10_seed42_v1/test_metrics.json.
Kết quả LDA Test sẽ được bổ sung sau khi cấu hình K=15 được phê duyệt và finalization hoàn thành.

## 5. So sánh khả năng diễn giải


## 5. So sánh khả năng diễn giải

### 5.1. Khả năng diễn giải của LDA K=15

Dựa trên kết quả M08, LDA K=15 hình thành nhiều nhóm
chủ đề có từ khóa và bài báo đại diện tương đối phù hợp.

| Topic | Chủ đề diễn giải | Bài báo đại diện |
|---|---|---|
| 0 | Giá vàng, tài chính | Giá vàng SJC tăng |
| 1 | Doanh nghiệp và công nghệ | Công ty Việt mở nhà máy UAV tại Mỹ |
| 2 | Y tế và bệnh lý | Polyp mũi sau phẫu thuật |
| 3 | Dinh dưỡng và sức khỏe | Lợi khuẩn đường ruột |
| 4 | Apple và thiết bị di động | Ra mắt iPhone 18 |
| 5 | Chưa xác định rõ | CEO Anthropic và rủi ro AI |
| 6 | Giáo dục phổ thông | Vụ 127 học sinh chưa tới trường |
| 7 | Giáo dục đại học | Sinh viên tốt nghiệp GPA 4.0 |
| 8 | Giá hàng hóa | Giá rau củ tăng |
| 9 | Pháp luật và điều tra | Bắt giữ giang hồ mạng |
| 10 | Thể thao ASIAD | Vận động viên thi đấu ASIAD |
| 11 | Tranh chấp và tố tụng | Vụ thẩm định giá và kháng cáo |
| 12 | Bóng đá | U23 Việt Nam tại ASIAD |
| 13 | Giải trí và nhân vật | Phỏng vấn ca sĩ Akira Phan |
| 14 | Chính sách thuế và doanh nghiệp | Giảm thuế cho hộ kinh doanh |

LDA phân tách một số lĩnh vực thành nhiều topic,
chẳng hạn giáo dục phổ thông và đại học, y tế và
dinh dưỡng, tài chính và giá hàng hóa.

Tuy nhiên, Topic 5 chứa nhiều từ phổ biến như
"một", "không", "những", "người" và "có_thể",
khiến việc đặt tên chủ đề chưa rõ ràng.

Các nhận xét dựa trên top-10 keywords và một bài
đại diện cho mỗi topic trong M08. Chưa có đánh giá
thủ công độc lập từ nhiều người.

### 5.2. Khả năng diễn giải của NMF K=10

Dựa trên artifacts NMF M04-M06 của TV2, các nhóm
chủ đề được nhận diện sơ bộ như sau:

| Topic | Chủ đề diễn giải | Keywords tiêu biểu |
|---|---|---|
| NMF_00 | Doanh nghiệp, công nghệ | doanh_nghiệp, phát_triển, công_nghệ |
| NMF_01 | Bóng đá | bóng, bàn, trận, cầu_thủ |
| NMF_02 | Pháp luật và điều tra | công_an, điều_tra, vụ_án |
| NMF_03 | Y tế và bệnh lý | bác_sĩ, bệnh, điều_trị |
| NMF_04 | Giáo dục | trường, học_sinh, giáo_dục |
| NMF_05 | Giải trí và điện ảnh | phim, khán_giả, diễn_viên |
| NMF_06 | Thể thao ASIAD | asiad, giành, u23 |
| NMF_07 | Bóng đá quốc tế, giải đấu | asean cup, fifa |
| NMF_08 | Apple và iPhone | iphone, apple, pro |
| NMF_09 | Giá cả, thị trường | giá, tăng, đồng, lãi_suất |

Các nhóm NMF_01, NMF_02, NMF_03 và NMF_08 có từ khóa tương đối đặc trưng, giúp nhận diện nội dung chủ đề.

Những bài đại diện đã kiểm tra của NMF_01 tập trung vào bóng đá, còn các bài đầu của NMF_02 liên quan đến các vụ án và hoạt động điều tra.

Tuy nhiên, chưa hoàn tất đánh giá bài đại diện của toàn bộ 10 topic theo cùng một phiếu chấm định tính.

### 5.3. Đối chiếu LDA và NMF

Cả hai mô hình phát hiện được những lĩnh vực nội dung tương đồng như kinh tế, bóng đá, y tế, giáo dục, pháp luật và công nghệ.

LDA K=15 phân chia một số lĩnh vực thành nhiều topic cụ thể hơn, trong khi NMF K=10 có một số nhóm từ khóa dễ nhận diện.

LDA có hạn chế ở những topic chứa nhiều từ phổ biến, đặc biệt là Topic 5.

Đây là nhận xét định tính từ hai cấu hình có K và preprocessing khác nhau. Chưa thể kết luận rằng sự khác biệt xuất phát hoàn toàn từ thuật toán.

Các topic keywords của LDA lấy từ M08 trên Train/Validation; NMF lấy từ mô hình final được fit trên Train+Validation. Vì vậy, việc đối chiếu
hiện tại mang tính mô tả, chưa phải một đánh giá định tính được chuẩn hóa hoàn toàn.

## 6. Các giới hạn khi so sánh

LDA và NMF sử dụng hai biến thể preprocessing khác nhau.

Do đó, kết quả phản ánh hiệu quả của các pipeline đã lựa chọn, không tách riêng ảnh hưởng của thuật toán.

Mặc dù hai pipeline sử dụng cùng hàm tính NPMI ở bước đánh giá cuối, vocabulary và preprocessing khác nhau vẫn ảnh hưởng đến kết quả.

ARI/NMI phản ánh mức độ khớp với category_gold, không đại diện đầy đủ cho chất lượng topic discovery.

## 7. Kết luận

Chưa đưa ra kết luận mô hình nào tốt hơn khi chưa có đầy đủ LDA Final Test và phân tích định tính tương ứng.
