# Hướng dẫn phân tích outlier HDBSCAN — Task M12

## 1. Mục tiêu

Task M12 phân tích các bài được HDBSCAN gán `cluster_id = -1` ở M11. Nhãn `-1` nghĩa là mô hình chưa đưa bài vào cụm trong **run tương ứng**; nó không tự động có nghĩa bài báo bị lỗi.

M12 gồm hai phần:

- Tính số lượng và tỷ lệ outlier trên toàn bộ tập train cho 32 run M11.
- Chọn mẫu bài outlier để đọc và phân loại thủ công, nhằm tìm hiểu vì sao một số bài bị gán `-1`.

## 2. Dữ liệu và cấu hình

Phân tích sử dụng Dataset v1.0 ở tập `train`, gồm **6.493 bài**. Dữ liệu được ghép với kết quả M11 bằng `article_id` theo `data/splits/split_manifest.csv`.

Cấu hình nằm trong `configs/outlier_analysis.yaml`. Phần tính tỷ lệ sử dụng kết quả của **4 embedding × 8 cấu hình HDBSCAN = 32 run**. Phần đọc thủ công tập trung vào embedding `mpnet_title_content` với hai cấu hình:

- `eom_15_5`
- `leaf_15_5`

Mỗi cấu hình được lấy **30 bài outlier**, tổng cộng **60 dòng nhận xét**. Mẫu có **55 `article_id` khác nhau**, vì 5 bài xuất hiện ở cả hai cấu hình. Do đó, khóa của một ca là `embedding_run + experiment + article_id`.

Sau khi rà soát, 60 dòng trong mẫu gồm:

- 39 dòng `valid_outlier`
- 16 dòng `rare_valid`
- 3 dòng `data_noise`
- 2 dòng `multi_topic`

Hai dòng `multi_topic` thuộc **cùng một bài** được xem trong hai run. Các số lượng trên chỉ mô tả mẫu đọc thủ công, không phải tỷ lệ nhãn của toàn bộ tập train.

## 3. Cách chạy và kiểm tra

Chạy từ thư mục gốc của dự án:

```powershell
python -B -m experiments.hdbscan.run_m12_outlier_analysis
```

Chương trình xuất ba file:

- `results/hdbscan/outlier_analysis/m12_outlier_rates.csv`: số lượng và tỷ lệ outlier theo run, nguồn báo và nhóm độ dài.
- `results/hdbscan/outlier_analysis/m12_candidates.csv`: 60 dòng bài được chọn để đọc.
- `results/hdbscan/outlier_analysis/m12_candidate_neighbors.csv`: ba bài **đã vào cụm** gần mỗi bài mẫu nhất theo cosine similarity.

Nhận xét thủ công nằm riêng trong `results/hdbscan/outlier_analysis/m12_labeled_cases.csv`. Runner không tạo lại file nhãn này, nhờ đó việc chạy lại chương trình không ghi đè nhận xét.

Chạy test:

```powershell
python -B -m pytest -p no:cacheprovider tests/test_outlier_analysis.py -v
```

Ở lần kiểm tra hiện tại, **3 test passed**. Test kiểm tra tỷ lệ M12 khớp kết quả M11, các bài được gán nhãn thật sự là outlier trong đúng run, và ID ba hàng xóm khớp file kết quả.

## 4. Cách tính tỷ lệ và lấy mẫu

Tỷ lệ outlier được tính bằng:

```text
outlier_percent = 100 × n_outliers / n_articles
```

Khi tính theo nguồn báo hoặc nhóm độ dài, mẫu số `n_articles` là **toàn bộ bài thuộc nhóm đó trong cùng run**, không phải tổng số outlier.

Độ dài bài là số từ trong `processed_content`, được chia thành bốn nhóm: `0-99`, `100-299`, `300-599` và `600+`.

Để lấy mẫu thủ công, chương trình chỉ chọn các bài có `cluster_id = -1`. Trong mỗi run, các bài được trộn với `random_state = 42`, rồi chọn luân phiên giữa các nhóm `source × length_group`. Cách này giúp mẫu có bài từ nhiều nguồn và nhiều nhóm độ dài.

Cột `content_excerpt` chứa 500 ký tự đầu của nội dung để xem nhanh. Khi kết luận một ca khó, cần đọc nội dung đầy đủ và đối chiếu bài nguồn.

Ba hàng xóm trong `m12_candidate_neighbors.csv` chỉ được chọn từ **các bài đã vào cụm**. Bảng này không thể hiện những bài cùng chủ đề nhưng cũng bị gán `-1`. Cosine similarity hỗ trợ đối chiếu ngữ nghĩa; nó không phải lời giải thích chính xác cho quyết định gán nhãn của HDBSCAN.

## 5. Kết quả tỷ lệ outlier

Với `mpnet_title_content` trên 6.493 bài train:

- `eom_15_5`: **1.660 bài outlier**, tương ứng **25,57%**.
- `leaf_15_5`: **2.148 bài outlier**, tương ứng **33,08%**.

`leaf_15_5` gán nhiều bài là outlier hơn `eom_15_5`. Riêng chênh lệch tỷ lệ này chưa đủ để kết luận cấu hình nào tốt hơn; cần xét thêm chất lượng cụm và các trường hợp cụ thể.

Ví dụ phân rã theo nguồn ở `eom_15_5`:

- Tuổi Trẻ: 680/2.649 bài, **25,67%**.
- Vietnamnet: 434/1.738 bài, **24,97%**.
- VnExpress: 546/2.106 bài, **25,93%**.

Nhóm độ dài `0-99` chỉ có **23 bài** trong tập train. Tỷ lệ của nhóm nhỏ này dễ thay đổi khi thêm hoặc bớt vài bài.

## 6. Quy tắc phân loại thủ công

`m12_labeled_cases.csv` sử dụng các nhãn sau:

- `data_noise`: văn bản đầu vào có dấu hiệu thiếu, lẫn nội dung hoặc thiếu ngữ cảnh khiến việc phân tích chủ đề không đáng tin cậy. Cần đối chiếu bài nguồn trước khi kết luận nguyên nhân.
- `rare_valid`: bài có nội dung hợp lệ, thuộc một chủ đề hoặc nhóm chủ đề nhỏ chưa tạo thành cụm rõ ràng trong run đang xét. Việc có vài bài tương tự cũng bị gán `-1` vẫn có thể phù hợp với nhãn này.
- `multi_topic`: một bản ghi chứa nhiều mảng nội dung đáng kể, khiến việc gán vào một chủ đề hẹp trở nên khó khăn.
- `valid_outlier`: bài hợp lệ, có chủ đề tương đối rõ nhưng vẫn bị gán `-1` trong run đang xét.
- `uncertain`: chưa đủ căn cứ để chọn một trong bốn nhãn trên.

Cột `evidence` ghi những dấu hiệu quan sát được từ bài và các bài đối chiếu. Cột `review_reason` giải thích vì sao những dấu hiệu đó dẫn tới `review_label`.

Không nên chỉ dựa vào việc bài “khác ba hàng xóm gần nhất” để kết luận chủ đề hiếm, vì bảng hàng xóm không bao gồm những bài cũng bị gán `-1`.

## 7. Ví dụ đã nhận xét

- **`data_noise` — `VNN_14f9af2f4e8b`, `eom_15_5`:** trang nguồn có câu dẫn về sức mua ở chợ truyền thống, nhưng cột `content` trong dữ liệu thiếu câu này và bắt đầu bằng “Trong bối cảnh đó”. Phần văn bản đầu vào chỉ khoảng 70 từ. Nhãn `data_noise` ở đây mô tả đầu vào thiếu ngữ cảnh, không khẳng định bài nguồn hoặc crawler bị lỗi.
- **`multi_topic` — `VNN_a9503ec90cf8`, `eom_15_5`:** bản ghi ngắn ghép các mẩu riêng về Tuấn Vũ, Hoàng Oanh và các hoa hậu. Các mẩu không cùng một sự kiện, nên khó xác định một chủ đề hẹp chung.
- **`valid_outlier` — `TTR_607809186810`, `eom_15_5`:** bài về lịch thi đấu ASIAD ngày 28/9. Các bài gần được đối chiếu cũng nói về lịch thi đấu và huy chương ASIAD, nhưng bài này vẫn được gán `-1`.
- **`rare_valid` — `VNE_f4ce960e6fd7`, `eom_15_5`:** bài về lớp đờn ca tài tử miễn phí. Có các bài cùng chủ đề như `TTR_a877e46a5083` và `TTR_c50eb2616808`, nhưng chúng cũng bị gán `-1` ở run này. Nhãn `rare_valid` mô tả nhóm chủ đề nhỏ đã tìm thấy, không có nghĩa bài hoàn toàn không có bài tương tự.
- **`rare_valid` — `VNE_d8327fe6097b`, `leaf_15_5`:** bài về quy định visa du học Australia. Các bài gần cùng vấn đề như `TTR_341d2fcab909` cũng bị gán `-1` ở run này.

Nhãn thủ công có thể được điều chỉnh nếu việc đọc toàn văn hoặc khảo sát thêm các bài cùng chủ đề cung cấp bằng chứng mới.

## 8. Giới hạn và kết luận

**60 dòng được gán nhãn là mẫu phục vụ phân tích trường hợp.** Mẫu này không đại diện cho toàn bộ 1.660 outlier của `eom_15_5`, 2.148 outlier của `leaf_15_5`, hoặc toàn bộ 32 run. Không dùng số lượng các nhãn trong 60 dòng để suy ra tỷ lệ `data_noise`, `rare_valid`, `multi_topic` hoặc `valid_outlier` của toàn bộ tập train.

Đặc biệt, hai dòng `multi_topic` trong mẫu thuộc cùng một bài ở hai run. Vì vậy, mẫu hiện chỉ cung cấp **một bài minh họa riêng biệt** cho loại này.

Kết quả M12 cho thấy các bài bị gán `-1` có nhiều dạng: văn bản đầu vào có thể thiếu ngữ cảnh, bản ghi có thể chứa nhiều chủ đề, bài có thể thuộc nhóm chủ đề nhỏ, hoặc một bài hợp lệ vẫn đứng ngoài các cụm. Khi đánh giá tiếp, cần xem cả chất lượng cụm và các ca outlier cụ thể, thay vì chỉ dựa vào tỷ lệ `-1`.