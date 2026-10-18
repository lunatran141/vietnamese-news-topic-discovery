# Đánh giá HDBSCAN trên tập validation

## Mục tiêu

Đánh giá khả năng gán bài báo mới vào các cụm HDBSCAN đã học từ tập train. Thí nghiệm xét hai embedding dùng **tiêu đề + 60 từ đầu của nội dung**:

- `minilm_title_content`
- `mpnet_title_content`

Mỗi embedding được kiểm tra với hai cấu hình HDBSCAN: `eom_15_5` và `leaf_15_5`. Tổng cộng có bốn run validation. Cả bốn run dùng cùng 6.493 bài train và 1.391 bài validation theo split D05.

Đây là phép kiểm tra bốn cấu hình được chọn để đối chiếu. Kết quả không đại diện cho toàn bộ 32 run M11 trên train.

## Phương pháp

Với từng embedding:

1. Tải embedding train đã lưu từ M10.
2. Fit lại UMAP 5 chiều trên **train** theo cấu hình M11.
3. Fit HDBSCAN trên các vector train đã giảm chiều, với `prediction_data=True`.
4. Kiểm tra nhãn của toàn bộ 6.493 bài train khớp chính xác assignments M11 đã lưu. Cả bốn run đều vượt qua bước kiểm tra này.
5. Mã hóa 1.391 bài validation bằng đúng mô hình embedding, cột văn bản và quy tắc lấy tiêu đề + 60 từ nội dung tương ứng với train.
6. Dùng UMAP đã fit để `transform` validation; dùng HDBSCAN `approximate_predict` để gán bài vào các cụm train đã có.

Không fit UMAP hoặc HDBSCAN trên validation. `approximate_predict` là phép gán gần đúng vào cấu trúc cụm train và không tạo cụm mới.

## Kết quả

| Embedding | HDBSCAN | Số cụm train | Bài `-1` train | Tỷ lệ `-1` train | Bài `-1` validation | Tỷ lệ `-1` validation | Tỷ lệ được gán cụm validation |
|---|---|---:|---:|---:|---:|---:|---:|
| MiniLM, tiêu đề + 60 từ | `eom_15_5` | 101 | 1.608/6.493 | 24,77% | 484/1.391 | 34,80% | 65,20% |
| MiniLM, tiêu đề + 60 từ | `leaf_15_5` | 126 | 2.310/6.493 | 35,58% | 682/1.391 | 49,03% | 50,97% |
| MPNet, tiêu đề + 60 từ | `eom_15_5` | 110 | 1.660/6.493 | 25,57% | 528/1.391 | 37,96% | 62,04% |
| MPNet, tiêu đề + 60 từ | `leaf_15_5` | 124 | 2.148/6.493 | 33,08% | 677/1.391 | 48,67% | 51,33% |

Tỷ lệ bài `-1` trên validation cao hơn train ở cả bốn run. Điều này cần được xem cùng chất lượng cụm và các bài cụ thể; tỷ lệ nhiễu thấp hơn không tự chứng minh một cấu hình tốt hơn.

### ARI và NMI

ARI/NMI được tính trên **toàn bộ 1.391 bài validation** của mỗi run. Khi tính, `-1` được xem là một nhãn dự đoán. Nhãn tham chiếu là `category_gold` trong split manifest.

| Embedding | HDBSCAN | ARI | NMI |
|---|---|---:|---:|
| `minilm_title_content` | `eom_15_5` | 0,1303 | 0,4218 |
| `minilm_title_content` | `leaf_15_5` | 0,0093 | 0,3592 |
| `mpnet_title_content` | `eom_15_5` | 0,0464 | 0,4101 |
| `mpnet_title_content` | `leaf_15_5` | 0,0098 | 0,3562 |

Trong hai run EOM, MiniLM có tỷ lệ nhiễu thấp hơn và ARI/NMI cao hơn MPNet trên validation. Hai run leaf có kết quả gần nhau. `category_gold` là chuyên mục báo, không phải nhãn chuẩn cho mọi chủ đề nhỏ; vì vậy các điểm này là chỉ báo tham khảo, chưa đủ để kết luận chất lượng chủ đề tổng thể hoặc chọn mô hình cuối.

## File kết quả

Assignments validation, mỗi file có 1.391 `article_id` duy nhất:

- `results/hdbscan/validation/minilm_title_content_eom_15_5_val_assignments.csv`
- `results/hdbscan/validation/minilm_title_content_leaf_15_5_val_assignments.csv`
- `results/hdbscan/validation/mpnet_title_content_eom_15_5_val_assignments.csv`
- `results/hdbscan/validation/mpnet_title_content_leaf_15_5_val_assignments.csv`

Bảng metric của cả bốn run:

- `results/hdbscan/validation/m11_validation_metrics.csv`

Các mô hình đã fit để tái sử dụng cho phép gán bài mới:

- `artifacts/hdbscan/validation/minilm_title_content_umap.joblib`
- `artifacts/hdbscan/validation/minilm_title_content_eom_15_5_hdbscan.joblib`
- `artifacts/hdbscan/validation/minilm_title_content_leaf_15_5_hdbscan.joblib`
- `artifacts/hdbscan/validation/mpnet_title_content_umap.joblib`
- `artifacts/hdbscan/validation/mpnet_title_content_eom_15_5_hdbscan.joblib`
- `artifacts/hdbscan/validation/mpnet_title_content_leaf_15_5_hdbscan.joblib`

Các file `.joblib` chứa mô hình đã fit. Trước khi đưa lên GitHub, cần thống nhất với nhóm việc lưu mô hình hay chỉ lưu code và hướng dẫn tạo lại, vì tổng dung lượng các file này lớn.

## Cách chạy lại

Chạy từ thư mục gốc của dự án. Hai lệnh đầu fit lại trên train và mã hóa validation; lệnh cuối tính metric từ bốn file assignments đã lưu:

```powershell
python -B -m experiments.hdbscan.run_m11_validation --embedding-run minilm_title_content
python -B -m experiments.hdbscan.run_m11_validation --embedding-run mpnet_title_content
python -B -m experiments.hdbscan.evaluate_m11_validation
```

Nếu cả bốn file assignments đã có và chỉ muốn tính lại bảng metric, chỉ cần chạy lệnh `evaluate_m11_validation`.

## Kiểm tra

```powershell
python -B -m pytest -p no:cacheprovider tests/test_embeddings.py tests/test_hdbscan.py tests/test_hdbscan_validation.py tests/test_m11_outputs.py tests/test_outlier_analysis.py -v
```

Bộ test trên đã đạt **19 passed** ở phiên bản hiện tại. Ngoài ra, cả bốn file assignments validation đã được kiểm tra có 1.391 dòng, 1.391 `article_id` duy nhất; số bài `-1` và ARI/NMI tính lại khớp bảng metric.