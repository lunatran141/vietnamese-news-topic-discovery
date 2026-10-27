# E04 — Tổng hợp stability từ kết quả có thể truy vết

Trạng thái: bản bổ sung chờ review. Phạm vi hiện tại có LDA, HDBSCAN và NMF; chưa hoàn tất E04 cho cả năm phương pháp.

## Giao thức và giới hạn

Các mô hình được học trên Train và đối chiếu phân nhóm trên cùng 1.391 article_id của Validation. Không dùng Test để chọn seed, K hoặc tham số khác. Bộ seed được giữ theo nguồn: LDA và NMF có 42, 52, 62, 72, 82; HDBSCAN có 42 đến 51. Đây chưa phải thí nghiệm thống nhất seed cho toàn nhóm. Kết quả mô tả từng pipeline; không dùng bảng này để xếp hạng thuật toán ổn định nhất.

Số topic của LDA được cố định ở 15 và NMF ở 10. Số topic HDBSCAN lấy từ các cụm được học trên Train, không phải số nhãn có mặt trong Validation; tỷ lệ outlier và ARI của bảng chung được tính trên Validation. Coherence HDBSCAN chưa có trong nguồn đang tổng hợp nên để thiếu, không gán bằng 0. Trường hợp có coherence, mỗi phương pháp phải ghi rõ định nghĩa và corpus đánh giá; không gộp trung bình các định nghĩa khác nhau.

ARI giữa các lần chạy so sánh hai vector nhãn của cùng article_id, không đối chiếu category_gold và không cần căn chỉnh số thứ tự topic. Với HDBSCAN, báo cả ARI trên mọi bài, bao gồm nhãn -1, và ARI trên các bài không phải outlier trong cả hai lần chạy. Kèm số bài và tỷ lệ giữ lại của từng cặp, vì điểm trên tập con không có cùng phạm vi với điểm trên mọi bài.

Jaccard từ khóa và Cosine phân phối của LDA/NMF dùng ghép Hungarian theo top-10 từ khóa. Với NMF, phân phối là topic share chuẩn hóa, không phải xác suất đã hiệu chuẩn. Không đưa dòng tự so sánh vào thống kê cặp. Các điểm Jaccard được tạo bằng quy tắc ghép khác nhau cũng không được coi là cùng thước đo chỉ vì cùng tên.

Thống kê gồm mean, sample standard deviation (ddof=1), min, max và range. Đơn vị thống kê được ghi là seed hoặc seed_pair. Các cặp seed có chung mô hình thành phần nên không độc lập; độ lệch chuẩn ở đây là mô tả, không phải sai số chuẩn hay khoảng tin cậy.

## Chạy và đầu ra

Tại thư mục gốc repo, dùng môi trường Python của dự án:

```powershell
python -B -u -m experiments.lda.run_e04_lda_handoff
python -B -u -m experiments.nmf.run_stability
python -B -m experiments.summarize_e04_stability
```

Hai lệnh chạy mô hình từ chối ghi đè thư mục đã tồn tại. Có thể dùng --output với thư mục mới cho lần chạy khác và cập nhật đường dẫn tương ứng trong configs/e04_stability.yaml. Lượt bổ sung LDA dùng n_jobs=1; NMF giới hạn 2 thread, được ghi trong metadata; các file M09 gốc và LDA Final không bị sửa. Hai lượt cũ/mới không được trộn vào cùng thống kê.

Thư mục results/lda/e04_handoff chứa per_seed.csv, pairwise.csv, validation_assignments.csv, topic_keywords.csv, validation_distributions.npz và run_metadata.json. Từ vựng được học một lần trên Train rồi dùng chung giữa các seed. Metadata lưu phiên bản Python/NumPy/scikit-learn, cấu hình và SHA256 của dữ liệu đầu vào.

Thư mục `results/stability/submissions/nmf` chứa submission NMF hợp lệ: năm seed, 10 cặp, K=10, Train/Validation và TF-IDF khớp một lần trên Train. Có thêm top-10 từ khóa, phép ghép từng topic, topic share Validation, hình và log. Jaccard/ARI của 10 cặp bằng 1; Cosine trung bình 0,9999999616. Xem `docs/nmf_stability.md` về kết quả và giới hạn của khởi tạo `nndsvda`.

Thư mục results/stability/e04 chứa:

- per_seed.csv: các kết quả theo seed, có source_directory và coherence_definition.
- pairwise.csv: ARI giữa các cặp seed và phạm vi bài được đánh giá.
- summary.csv: thống kê theo metric, phương pháp và đơn vị.
- coverage_status.csv: nguồn đã có và các submission còn thiếu.
- manifest.json: giao thức tổng hợp và SHA256 của nguồn đã đọc.

## Bổ sung kết quả từ các thành viên

Đường dẫn được cấu hình trong configs/e04_stability.yaml. Với mỗi nguồn chuẩn, cần ba file:

1. per_seed.csv, mỗi seed đúng một dòng, tối thiểu ba seed. Cột bắt buộc: method, seed, split, n_documents, topic_count, coherence, coherence_definition, topic_diversity, outlier_rate, coverage. Metric chưa tính để trống; các trường số bài, số topic, outlier và coverage phải có giá trị. method phải đúng tên trong cấu hình, split là val; outlier_rate và coverage dùng tỷ lệ 0–1.
2. validation_assignments.csv: seed, article_id, topic_id. Mỗi seed chứa đủ các article_id của Validation, không trùng; outlier có topic_id=-1.
3. run_metadata.json: dataset_version=v1.0, fit_split=train, evaluation_split=val, seeds, preprocessing, biểu diễn, cấu hình mô hình và phiên bản thư viện. Ghi rõ tham số nào nhận seed và những bước nào được giữ cố định.

Có thể bổ sung pairwise.csv theo schema LDA để tổng hợp keyword_jaccard và document_cosine; file này không thay thế assignments. Công cụ tính lại ARI và kiểm tra với điểm ARI đã lưu nếu có. Nguồn thiếu được đánh dấu missing_submission; nguồn không hợp lệ làm lệnh dừng với lỗi.

Kết quả chạy một seed, nhiều giá trị K hoặc nhiều preprocessing không tự động được coi là thí nghiệm nhiều seed với cấu hình cố định. NMF đã bổ sung bộ đầu vào phù hợp; K-Means và BERTopic vẫn được đánh dấu `missing_submission`. E04 hiện chưa bao phủ đủ năm phương pháp.
