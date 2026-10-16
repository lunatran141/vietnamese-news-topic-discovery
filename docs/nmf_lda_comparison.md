# TV2 → TV3 — Khung so sánh NMF và LDA (task 2.8)

| Khía cạnh | NMF đã chạy | LDA cần TV3 cung cấp |
|---|---|---|
| Cơ chế | Xấp xỉ TF-IDF không âm bằng W@H; topic share là W chuẩn hóa để đọc | Mô hình sinh xác suất chủ đề–từ và bài–chủ đề; ghi rõ biến thể triển khai |
| Dữ liệu | Dataset v1.0, P3, 9.274 ID | Cùng Dataset V1 và split; manifest khuyến nghị P2, ghi rõ nếu dùng biến thể khác |
| Chọn K | Train→validation; thử 4, 6, 8, 10, 12; K=10 có NPMI validation cao nhất | K candidates, split chọn K và luật chọn |
| Test | NPMI 0,4407; diversity 1,0000; ARI 0,5391; NMI 0,6514 | Cùng cách tính metric hoặc ghi N/A và lý do |
| Diễn giải | 10 top words/topic, 5 bài đại diện/topic, case bài đa chủ đề | Top words, đại diện và ví dụ bài nhiều topic |
| Chi phí/độ bền | Chưa có runtime chuẩn/stability/OOD cho NMF | TV3 cung cấp setup runtime, seed/stability và OOD |

**Quy tắc so sánh:** NMF dùng ma trận TF-IDF và LDA thường dùng count matrix;
topic share của NMF sau chuẩn hóa không là xác suất suy diễn giống LDA. Không
so “độ tin cậy” bằng share tuyệt đối giữa hai mô hình. Chỉ đặt hai NPMI cạnh
nhau nếu cùng top-N, cùng corpus tham chiếu, tokenization và công thức. NMI/ARI
so với `category_gold` chỉ phản ánh mức trùng với chuyên mục báo. Khi LDA
chưa có run/artifact trong repo, bảng này là khung bàn giao, **không phải kết
luận phương pháp nào tốt hơn**.
