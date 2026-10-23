# D05 - Train/Validation/Test Protocol Audit & Verification Report

**Người thực hiện:** TV1 (Phạm Đăng Khôi)  
**Ngày thực hiện:** 23/10/2026  
**Dự án:** Phân loại và khám phá chủ đề tin tức tiếng Việt không giám sát  
**Trạng thái kiểm tra:** ĐÃ XÁC MINH TOÀN DIỆN (7/7 Unit Tests PASSED)

---

## 1. Mục tiêu kiểm tra (Audit Objectives)

Theo phân công nhiệm vụ D05, TV1 thực hiện kiểm tra và chuẩn hóa giao thức phân chia dữ liệu cho toàn bộ nhóm nghiên cứu nhằm trả lời các câu hỏi then chốt:
1. Số lượng bài báo trong các tập Train, Validation và Test có thống nhất giữa cả 5 mô hình không?
2. Tập Test có bị sử dụng để chọn tham số (tuning) ở bất kỳ mô hình nào không?
3. Hạt giống ngẫu nhiên (seed) và danh sách mã định danh `article_id` của từng split đã được lưu trữ cố định chưa?
4. Có hiện tượng trùng lặp hoặc rò rỉ thông tin (data leakage / event leakage) giữa các tập không?
5. Mỗi mô hình trong nhóm hiện đang sử dụng tập dữ liệu nào cho từng giai đoạn?

---

## 2. Kết quả kiểm tra tính toàn vẹn phân chia (Split Integrity Audit)

Toàn bộ dữ liệu được quản lý tập trung qua file chỉ mục `data/splits/split_manifest.csv` và 3 file danh sách ID: `data/splits/train_ids.csv`, `data/splits/validation_ids.csv`, `data/splits/test_ids.csv`.

### 2.1. Bảng số lượng và phân bố dữ liệu theo Split

| Tập dữ liệu (Split) | Số lượng bài báo | Tỷ lệ (%) | Mục đích sử dụng | Trạng thái bảo vệ |
| :--- | :---: | :---: | :--- | :--- |
| **Train (Huấn luyện)** | 6.493 | 70,01% | Trích xuất từ vựng, học trọng số đặc trưng và huấn luyện mô hình | Mở |
| **Validation (Phát triển)** | 1.391 | 15,00% | Tìm kiếm siêu tham số, khảo sát số cụm $K$, đo độ ổn định đa seed và chạy các thực nghiệm cắt giảm (ablation) | Mở cho tuning |
| **Test (Kiểm tra cuối)** | 1.390 | 14,99% | Đánh giá tổng kết khả năng khái quát hóa một lần duy nhất trên cấu hình tối ưu đã chốt | **Khóa kín (Locked hold-out)** |
| **Tổng cộng (Dataset v1.0)** | **9.274** | **100,00%** | Toàn bộ ngữ liệu báo chí sạch sau khử trùng lặp | — |

### 2.2. Kiểm tra giao nhau và rò rỉ (Zero-Overlap & Leakage Verification)

- **Trùng lặp mã bài viết:**
  - $\text{Train} \cap \text{Validation} = \emptyset$ (0 bài)
  - $\text{Train} \cap \text{Test} = \emptyset$ (0 bài)
  - $\text{Validation} \cap \text{Test} = \emptyset$ (0 bài)
- **Rò rỉ nhóm sự kiện (Event Leakage):** Toàn bộ 9.114 nhóm sự kiện thời sự (`group_id`) được phân bổ độc lập; không có bất kỳ sự kiện nào bị phân mảnh hoặc xuất hiện đồng thời ở nhiều hơn 1 split.
- **Tính đại diện chuyên mục:** Tỷ lệ 7 chuyên mục biên tập (`category_gold`) giữa ba tập lệch tối đa không quá 0,09 điểm phần trăm, bảo đảm tính phân tầng chặt chẽ.
- **Lưu trữ seed:** Toàn bộ quá trình chia tách sử dụng `random_state = 42`, lưu tại `configs/split.yaml` và `artifacts/configs/split.yaml`.

---

## 3. Rà soát việc sử dụng Split trên 5 mô hình nghiên cứu

| Mô hình | Thành viên | Split trích xuất đặc trưng | Split huấn luyện (Fit) | Split chọn cấu hình (Tuning) | Split đánh giá cuối (Test) | Tuân thủ Test Lock? |
| :--- | :---: | :--- | :--- | :--- | :--- | :---: |
| **K-Means** | TV1 | Fit TF-IDF trên Train | Fit KMeans trên Train (6.493 bài) | Validation (tìm $K \in [3, 18]$) | Test (1.390 bài, chỉ transform & predict) | **ĐẠT** |
| **NMF** | TV2 | Fit TF-IDF trên Train | Giai đoạn 1: Fit Train; Giai đoạn 2: Fit Train+Val (7.884 bài) | Validation (chọn $K = 10$) | Test (1.390 bài, chỉ transform & predict) | **ĐẠT** |
| **LDA** | TV3 | Fit BoW trên Train | Giai đoạn 1: Fit Train; Giai đoạn 2: Fit Train+Val (7.884 bài) | Validation (chọn $K = 15$) | Test (1.390 bài, chỉ inference) | **ĐẠT** |
| **HDBSCAN** | TV4 | Encode Train+Val | Fit UMAP & HDBSCAN trên Train (6.493 bài) | Validation (chọn min_cluster_size) | Test (1.390 bài, encode & approx predict) | **ĐẠT** |
| **BERTopic** | TV5 | SBERT embedding | Fit BERTopic trên Train (6.493 bài) | Validation (chọn n_neighbors/cluster_size) | Test (1.390 bài, transform & predict) | **ĐẠT** |

### Nhận xét quan trọng về giao thức giữa các mô hình:
1. **Không có mô hình nào sử dụng tập Test để chọn tham số:** Toàn bộ việc lựa chọn số cụm $K$, số chủ đề, kích thước cụm tối thiểu hoặc tham số giảm chiều đều được quyết định độc lập trên tập Validation.
2. **Khác biệt về tập dữ liệu fit cuối cùng:**
   - K-Means, HDBSCAN và BERTopic giữ nguyên mô hình fit từ tập Train (6.493 bài) để suy luận trên tập Test.
   - NMF và LDA sau khi chốt cấu hình tối ưu trên Validation đã thực hiện bước tái huấn luyện (refit) trên gộp cả Train + Validation (7.884 bài) trước khi đo Test.
   - *Đánh giá tính hợp lệ:* Cả hai cách tiếp cận đều hợp lệ trong học máy thực nghiệm và không gây rò rỉ nhãn Test. Tuy nhiên, sự khác biệt này cần được ghi chú minh bạch khi đối chiếu thời gian chạy (runtime) và độ thuần khiết (purity) ở phần thảo luận chung.

---

## 4. Đoạn văn mô tả Protocol đưa vào Báo cáo NCKH

Dưới đây là đoạn văn chuẩn mực đã được đưa vào Mục 3.5 của báo cáo nghiên cứu:

> *"Để bảo đảm tính khách quan và khả năng tái lập, tập ngữ liệu gồm 9.274 bài báo sạch được phân chia cố định theo tỷ lệ 70% / 15% / 15% thành ba tập: Huấn luyện (Train: 6.493 bài), Phát triển (Validation: 1.391 bài) và Kiểm tra khóa kín (Test: 1.390 bài). Quá trình phân chia sử dụng phương pháp lấy mẫu phân tầng theo nhãn chuyên mục biên tập của tòa soạn kết hợp ràng buộc bảo toàn nhóm sự kiện thời sự (seed 42), bảo đảm không xảy ra hiện tượng rò rỉ dữ liệu hoặc chia cắt sự kiện giữa các tập. Quy tắc đánh giá được khóa cố định nghiêm ngặt: **Validation dùng để chọn cấu hình; Test chỉ mở sau khi chốt mô hình**."*

---

## 5. Kết luận

- Split dữ liệu hiện tại hoàn toàn chuẩn xác, nhất quán 100% giữa 5 mô hình, không cần chia lại dữ liệu.
- Hoàn thành đầy đủ các yêu cầu kiểm tra và bàn giao của Task D05.

