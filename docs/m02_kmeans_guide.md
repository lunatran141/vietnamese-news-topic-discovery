# Hướng dẫn kỹ thuật và báo cáo Task M02: K-Means Clustering Baseline
## 1. Thiết lập

- Đầu vào: ma trận TF-IDF 20,000 chiều từ M01 (`artifacts/tfidf/`), fit trên Train (6,493 bài), Validation 1,391 bài. Test (1,390 bài) không dùng.
- K-Means: `k-means++`, `n_init=10`, `max_iter=300`, seed 42.
- Dải K khảo sát: 3, 5, 7, 9, 11, 14, 18. Corpus có 7 giá trị `category_gold`, nên K=7 là mốc đối chiếu.
- `category_gold` không tham gia huấn luyện, không tham gia độ ổn định và không nằm trong công thức `composite_score`.

## 2. Kết quả K-search trên Validation

Toàn bộ số liệu nằm ở `results/kmeans/kmeans_k_search_metrics.csv`.

| K | Silhouette | Davies-Bouldin | Ổn định (ARI giữa 5 seed, trung bình ± độ lệch) | Cụm nhỏ nhất (Val) | Giảm Train Inertia so với K trước | NMI | Purity | ARI |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 3 | 0.0147 | 7.149 | 0.849 ± 0.171 | 218 | - | 0.479 | 0.510 | 0.255 |
| 5 | 0.0189 | 7.194 | 0.917 ± 0.089 | 163 | 87.5 | 0.657 | 0.724 | 0.561 |
| 7 | 0.0205 | 6.618 | 0.828 ± 0.116 | 90 | 63.6 | 0.689 | 0.821 | 0.624 |
| **9** | 0.0190 | 6.157 | 0.822 ± 0.044 | 76 | 53.6 | 0.700 | 0.873 | 0.593 |
| 11 | 0.0205 | 5.826 | 0.806 ± 0.051 | 30 | 47.2 | 0.646 | 0.825 | 0.490 |
| 14 | 0.0194 | 5.629 | 0.699 ± 0.083 | 30 | 49.3 | 0.627 | 0.838 | 0.438 |
| 18 | 0.0191 | 5.603 | 0.713 ± 0.069 | 7 | 49.3 | 0.623 | 0.881 | 0.391 |

Cách đọc bảng:

- **Silhouette gần như không phân biệt được K.** Mọi giá trị nằm trong 0.0147 đến 0.0205, K=7 và K=11 bằng nhau. Với TF-IDF 20,000 chiều rất thưa, Silhouette luôn gần 0, nên chỉ dùng như chỉ báo phụ.
- **Davies-Bouldin giảm gần như đơn điệu theo K** (7.19 → 5.60), nên chỉ số này tự nó thiên về K lớn.
- **Không có điểm khuỷu rõ ràng.** Mức giảm Train Inertia giảm dần từ 87.5 xuống 47.2 rồi đứng ở khoảng 49 từ K=11 đến K=18. Đường cong không gãy ở một K cụ thể.
- **K=14 và K=18 kém ổn định** (ARI giữa các seed 0.70 và 0.71, so với 0.81 đến 0.83 ở K=7 đến 11). K=18 còn có cụm chỉ 7 bài trên Val.
- `composite_score` (trung bình min-max của Silhouette và Davies-Bouldin) chọn K=11, nhưng vì Davies-Bouldin thiên về K lớn nên công thức này chỉ mang tính tham khảo.

## 3. Đánh giá thủ công

Bảng chi tiết: `results/kmeans/kmeans_manual_evaluation.csv` (27 hàng: K=7, 9, 11). Mỗi cụm có 10 từ khóa, 5 bài gần tâm nhất (từ Train) và 5 bài ngẫu nhiên (từ Val, seed cố định).

> [NOTE]
> Cột `manual_label`, `coherence_1_5`, `is_mixed_topic`, `notes` của cấu hình tối ưu $K=9$ đã được **xác nhận chính thức** (`review_status = confirmed`). Chi tiết lưu tại `results/kmeans/kmeans_manual_evaluation.csv`. Nháp gốc đối chiếu cho $K=7, 9, 11$ nằm ở `results/kmeans/kmeans_manual_evaluation_drafts.json`.

Tóm tắt nháp:

| K | Điểm coherence trung bình (nháp) | Cụm trộn nhiều chủ đề | Cụm điểm ≤ 2 |
| :-: | :-: | :-: | :-: |
| 7 | 4.00 | 2/7 | 1 |
| 9 | 4.11 | 2/9 | 1 |
| 11 | 4.18 | 3/11 | 2 |

Điểm trung bình giữa K=7, 9, 11 chênh nhau rất ít và đến từ một người đánh giá, nên không đủ để xếp hạng. Điểm đáng chú ý là chất lượng cụm không đồng đều:

- Các cụm Sức khỏe, Giáo dục, Giá cả tài chính, Bóng đá đều thuần và nhất quán ở cả ba giá trị K.
- Có một cụm kiểu "thùng chứa" về doanh nghiệp/đầu tư/công nghệ ở mọi K (K=9: cụm 6, điểm 2).
- Cụm giải trí lẫn nội dung quảng cáo và đời sống (K=9: cụm 0, điểm 3).
- K=11 tách thêm cụm "Quy định, pháp lý và hành chính" (cụm 8) nhưng đây là cụm tạp (điểm 2), và tách cụm iPhone chỉ 30 bài Val.

## 4. K được chọn: K=9

Lý do (đây là một phán đoán, các chỉ số nội tại sát nhau):

1. Loại K=14 và K=18 vì độ ổn định thấp, K=18 còn có cụm 7 bài.
2. So với K=7: K=9 tách được mảng công nghệ (Apple và AI) khỏi cụm doanh nghiệp/đầu tư, và tách Asiad/đội tuyển đa môn khỏi bóng đá. Ở K=7, công nghệ bị trộn vào cụm "thùng chứa".
3. So với K=11: K=9 ổn định hơn một chút (0.822 ± 0.044 so với 0.806 ± 0.051), không có cụm quá nhỏ (nhỏ nhất 76 bài Val so với 30), và không có cụm tạp "quy định/pháp lý" như K=11.

Giới hạn cần nêu khi báo cáo:

- NMI, Purity và ARI cũng tốt nhất hoặc gần tốt nhất ở K=9. Các chỉ số này **không** được dùng làm tiêu chí, nhưng người chọn K đã nhìn thấy chúng, nên lựa chọn có thể bị ảnh hưởng. Nếu muốn loại ảnh hưởng này, đặt `selected_k` khác trong `configs/kmeans.yaml` rồi chạy lại.
- Điểm coherence nháp của K=11 (4.18) cao hơn K=9 (4.11) một chút. Chênh lệch này nằm trong sai số của một người đánh giá.

## 5. Kết quả K=9 trên Validation

| Chỉ số | Giá trị |
| :-- | :-: |
| Silhouette | 0.0190 |
| Davies-Bouldin | 6.157 |
| Calinski-Harabasz | 11.37 |
| ARI so với `category_gold` | 0.593 |
| NMI so với `category_gold` | 0.700 |
| Purity so với `category_gold` | 0.873 |

Các cụm (kích thước trên Train/Val, nhãn gold chiếm ưu thế trên Val, nhãn thủ công nháp):

| Cụm | Train / Val | Gold chiếm ưu thế (Val) | Nhãn thủ công (nháp) | Bài gần tâm nhất |
| :-: | :-: | :-: | :-- | :-- |
| 0 | 1135 / 251 | Giải trí 68.1% | Giải trí, nghệ sĩ, đời sống (lẫn quảng cáo) | Mỹ Anh: Không đợi thành công trong nước rồi mới ra quốc tế |
| 1 | 395 / 98 | Thể thao 99.0% | Thể thao: Asiad và đội tuyển đa môn | Xác định 8 đội vào tứ kết bóng đá nữ Asiad 20 |
| 2 | 377 / 76 | Khoa học công nghệ 92.1% | Công nghệ: Apple và AI | Toàn cảnh thông tin hot nhất về iPhone 18 Pro, 18 Pro Max, iPhone Duo gập trước giờ ra mắt |
| 3 | 722 / 159 | Sức khỏe 98.7% | Sức khỏe, bệnh và điều trị | 5 điều cần biết để giữ thận khỏe |
| 4 | 832 / 174 | Thể thao 100% | Bóng đá: đội tuyển, V-League, FIFA | HLV tuyển Philippines: 'Chúng tôi không đáng thua' |
| 5 | 882 / 188 | Pháp luật 87.2% | Pháp luật hình sự và điều tra | Khởi tố nhóm thuê người vay ngân hàng mua ô tô rồi bán lại |
| 6 | 1109 / 229 | Kinh doanh 74.7% | Doanh nghiệp, đầu tư, dịch vụ (cụm chung chung) | Việt Nam đang tạo ra những Doanh nghiệp Quốc gia thay đổi cả nền kinh tế thế nào? |
| 7 | 650 / 129 | Giáo dục 97.7% | Giáo dục, đào tạo (cả y khoa) | 5 kỳ vọng lớn của ngành giáo dục năm học mới |
| 8 | 391 / 87 | Kinh doanh 96.6% | Giá cả và thị trường tài chính | VN-Index mất mốc 1.800 điểm, ông Phạm Nhật Vượng vẫn thăng hạng tỷ phú USD |

Nhận xét về khoảng cách đến tâm: `distance_to_center` trên 7,884 bài có trung bình 0.948, độ lệch chuẩn 0.028, nhỏ nhất 0.795, lớn nhất 1.013. Giá trị gần 1 với vector chuẩn hóa L2 cho thấy các cụm khá lỏng, nhất quán với Silhouette gần 0. K-Means trên TF-IDF cho cụm dễ đọc về từ khóa, nhưng không cho cụm chặt về hình học.

## 6. Artifacts và kết quả

`artifacts/kmeans/` (K=9):

| Tệp | Nội dung |
| :-- | :-- |
| `kmeans_model.joblib` | Mô hình đã fit trên Train |
| `document_topics.csv` | 7,884 dòng (Train + Val): `article_id, title, cluster_id, distance_to_center, split, category_gold` |
| `topic_keywords.json` | 15 từ khóa theo trọng số tâm cụm cho mỗi cụm |
| `representative_docs.json` | 5 bài gần tâm nhất của mỗi cụm |
| `config.json`, `run_metadata.json` | Cấu hình và metadata |

`results/kmeans/`: `kmeans_k_search_metrics.csv`, `kmeans_manual_evaluation.csv`, `kmeans_manual_evaluation_drafts.json`, và 5 biểu đồ trong `figures/` (Elbow, Silhouette và Davies-Bouldin, Silhouette Analysis từng mẫu, kích thước cụm, chiếu 2D bằng TruncatedSVD).

## 7. Cách chạy lại

```bash
python -m experiments.kmeans.run_m02_kmeans
python -m experiments.kmeans.build_manual_evaluation
pytest tests/test_kmeans.py -v
```

Đổi K cuối: sửa `selected_k` trong `configs/kmeans.yaml` hoặc truyền `--k`. K-search luôn chạy đầy đủ, trừ khi dùng `--skip-search`.

## 8. Giới hạn

- K-Means gán mỗi bài vào đúng một cụm, nên bài liên ngành bị ép vào một cụm.
- TF-IDF không bắt được đồng nghĩa.
- Các cụm Thể thao ở K=9 và K=11 phản ánh các sự kiện trong khoảng thời gian thu thập (Asiad, FIFA ASEAN Cup), không nhất thiết là chủ đề bền vững.
- Đánh giá thủ công hiện là nháp chưa có người thứ hai kiểm tra chéo.
