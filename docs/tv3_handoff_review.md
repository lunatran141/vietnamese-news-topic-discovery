# TV2 tiếp quản TV3

Ngày rà soát: 23/10/2026. Phạm vi: LDA M07–M09, E04, báo cáo LDA và Related Work 2.2. 

## Kết quả theo nhiệm vụ

| Nhiệm vụ | Đã làm | Còn cần hoàn tất |
|---|---|---|
| Tiếp quản M07–M09 | Đọc cấu hình/code; audit model Final, vocabulary, split, mapping và bài đại diện; chạy toàn bộ test_lda_*.py | Đã đạt: toàn bộ test LDA pass trong venv Python 3.12 |
| E04 Stability | Chạy LDA bổ sung năm seed; lưu assignments, distributions, keywords; tính ARI giữa các lần chạy; tổng hợp với HDBSCAN hiện có | Bộ đầu vào nhiều seed phù hợp cho K-Means, NMF, BERTopic; chốt yêu cầu seed chung với nhóm |
| R03 LDA | Chuẩn hóa sáu mục, rút hai bảng riêng thành đoạn văn, làm rõ scope stability, bổ sung References APA, giữ nhãn proposed | Review nội bộ và phân trang trong Paper Master |
| Related Work 2.2 | Làm rõ trọng số NMF so với xác suất LDA, giới hạn so sánh pipeline và phạm vi nghiên cứu trên Dataset v1.0 | Nhóm review nội dung và thống nhất với Paper Master |

## Kiểm chứng LDA Final

File bằng chứng: results/lda/handoff_audit/final_artifact_audit.json. Cả 10 kiểm tra cấu trúc và tính nhất quán đều đạt: model thực có K=15, seed=42, ma trận thành phần 15 × 15.000, vocabulary 15.000; split đúng 6.493/1.391/1.390; common artifact đủ 9.274 ID duy nhất, đúng split; 150 bài đại diện gồm 10 bài/topic và chỉ lấy Train/Validation; nhãn của 15 topic còn proposed. Test metrics khớp common model summary. Đây là audit artifact hiện có, không phải chạy lại Final Test.

Số liệu Final đang được giữ theo nguồn: NPMI 0,147094; diversity 0,653333; ARI 0,406466; NMI 0,572713.

Có hai cách tổng hợp stability M09 đã tồn tại: paper dùng trung bình toàn bộ 10 cặp khác seed, Jaccard 0,455685; common UI dùng bốn cặp so với seed tham chiếu 42, Jaccard 0,443351. Khác biệt này do phạm vi tổng hợp, không tự chứng minh artifact sai. Không thay một giá trị bằng giá trị còn lại mà bỏ mô tả scope.

## Lượt LDA bổ sung cho E04

Nguồn: results/lda/e04_handoff. Cấu hình học trên Train, đánh giá trên 1.391 bài Validation, P2, BoW 15.000 đặc trưng, K=15, seeds 42/52/62/72/82. Dùng n_jobs=1; config M09 cũ dùng n_jobs=-1. Thư viện và SHA256 dữ liệu/cấu hình được lưu trong run_metadata.json. Bộ đọc dùng chung CSV dữ liệu rồi lấy các split cần thiết; Test không tham gia fit, lựa chọn mô hình hay tính metric của lượt bổ sung.

| Metric | Đơn vị | Mean | Sample SD |
|---|---|---:|---:|
| NPMI document_binary_common | 5 seed | 0,131470 | 0,010312 |
| Topic diversity | 5 seed | 0,609333 | 0,012111 |
| ARI giữa lần chạy | 10 cặp seed | 0,514909 | 0,059627 |
| Keyword Jaccard sau ghép | 10 cặp seed | 0,485447 | 0,035953 |
| Document Cosine sau căn chỉnh | 10 cặp seed | 0,704830 | 0,049520 |

Các điểm mới khác số liệu M09 gốc. Hai lượt có khác biệt về thực thi song song; chưa có thí nghiệm kiểm soát để quy nguyên nhân của chênh lệch cho một yếu tố cụ thể. Các kết quả mới được giữ như một thí nghiệm bổ sung chờ review, không ghép vào bảng M09 gốc và chưa thay số liệu stability trong bản paper.

## Bảng tổng hợp E04 hiện có

Nguồn: results/stability/e04. LDA có 5 seed; HDBSCAN có 10 seed 42–51. Đã tính lại 45 ARI HDBSCAN từ Validation assignments và đối chiếu khớp bảng pairwise gốc.

| Phương pháp | ARI giữa lần chạy, mean ± SD | Số topic của mô hình | Outlier rate Validation, mean ± SD |
|---|---|---|---|
| LDA bổ sung | 0,514909 ± 0,059627 | 15, cố định | 0 |
| HDBSCAN nguồn có sẵn | 0,477408 ± 0,061906 | 99–117; mean 109,6 | 0,391445 ± 0,030244 |

ARI toàn bộ bài của HDBSCAN bao gồm nhãn -1. ARI trên tập bài không phải outlier trong cả hai lần chạy có mean 0,795243; tập con này chỉ giữ trung bình khoảng 50,88% số bài. Không dùng điểm tập con để so trực tiếp với ARI toàn bộ bài. LDA và HDBSCAN dùng pipeline, số topic và bộ seed khác nhau nên bảng này không đủ để kết luận mô hình nào ổn định nhất. SD của các cặp là thống kê mô tả; các cặp không độc lập.

Sau khi bổ sung thí nghiệm NMF, coverage_status.csv đánh dấu LDA, HDBSCAN và NMF là available; KMeans và BERTopic vẫn là missing_submission theo schema E04. NMF dùng năm seed 42, 52, 62, 72, 82 với K=10, đạt ARI giữa các lần chạy 1,000000 ± 0,000000 trên Validation; chi tiết và giới hạn xem docs/nmf_stability.md. Bảng hai phương pháp phía trên giữ số liệu của lượt bàn giao LDA trước đó; bảng tổng hợp hiện tại nằm ở results/stability/e04/summary.csv. Điều này nói rằng công cụ chưa nhận đủ bộ đầu vào phù hợp, không có nghĩa những mô hình đó chưa được triển khai hoặc không có kết quả thực nghiệm khác trong repo. Xem docs/e04_stability_protocol.md để chuẩn bị đầu vào hoặc đổi đường dẫn nguồn trong configs/e04_stability.yaml.

## Kiểm thử local sau khi cài Gensim

Đã chạy lại toàn bộ tests/test_lda_*.py cùng tests/test_e04_stability_summary.py bằng venv/Scripts/python.exe: **73 pass, 0 fail, 0 error, 0 skip** trong 15,26 giây. Môi trường là Python 3.12.10 và Gensim 4.4.0. Báo cáo máy đọc được: results/lda/handoff_audit/pytest_report.xml; phiên bản các thư viện được lưu tại results/lda/handoff_audit/test_environment.json.

Có 26 cảnh báo pandas tại experiments/human_eval/summarize_lda_nmf.py, liên quan đến fillna với chuỗi trên cột số. Các cảnh báo này không làm thất bại kiểm thử; chúng thuộc phần tổng hợp human evaluation.

Kết quả cũ 69 pass, 4 fail do thiếu Gensim thuộc môi trường .venv dùng Python 3.14. Lần chạy mới đã kiểm chứng đủ bốn test coherence trước đây bị chặn, không skip hoặc mock. Điều kiện toàn bộ test LDA pass hiện đã đạt trong môi trường venv Python 3.12.

Việc dời import Gensim vào hai hàm thực sự sử dụng thư viện trong src/evaluation/lda_metrics.py vẫn được giữ. Lượt này chỉ chạy lại test và cập nhật bằng chứng; cấu hình và số liệu của thí nghiệm E04 đã lưu vẫn thuộc môi trường ghi trong run_metadata.json của lượt đó.

## Review nội dung paper

Đã chỉnh paper/sections/lda_method_results.md theo template sáu mục, bỏ bảng riêng để dành chỗ cho bảng so sánh chung, chuẩn hóa nguồn thực nghiệm và thêm ba mục References khớp các trích dẫn trong phần này. C_V/C_NPMI M08 vẫn được phân biệt với NPMI document-level của Final. Không có dấu/lệnh LaTeX trong phần LDA sau chỉnh sửa.

Đã chỉnh đoạn cuối mục 2.2 của paper/sections/related_work.md: research gap gắn với corpus và giao thức của đồ án, không tuyên bố toàn bộ lĩnh vực chưa có nghiên cứu tiếng Việt; phân biệt trọng số chuẩn hóa NMF với phân phối xác suất LDA và giới hạn của việc so sánh hai pipeline. Phần này vẫn dùng các nguồn học thuật hiện có trong repo; lượt sửa này không được mô tả là một lần đối chiếu lại toàn văn mọi paper gốc.

## Điều kiện còn thiếu để nghiệm thu

- Đã đạt: toàn bộ test LDA pass trong venv Python 3.12.10, Gensim 4.4.0.
- Có đủ submission stability cho năm phương pháp, hoặc nhóm thống nhất ghi rõ phạm vi chưa đầy đủ; chốt yêu cầu có cần cùng bộ seed hay giữ các bộ hiện có.
- Reviewer độc lập xác nhận nhãn/các nhận xét định tính nếu muốn nâng trạng thái proposed.
- Nhóm duyệt bộ thay đổi local và chọn cách đưa thí nghiệm E04 bổ sung vào Paper Master. Sau đó mới chuẩn bị commit/PR.
