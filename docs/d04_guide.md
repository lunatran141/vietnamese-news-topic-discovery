# Hướng dẫn Kiểm tra và Bàn giao Dataset v1 
## 1. Thông tin bàn giao chung

- **Phiên bản dữ liệu:** `Dataset v1.0`
- **Trạng thái phê duyệt:** `approved` (Cổng chất lượng Quality Gate: `PASSED`)
- **Số lượng bài viết chính thức:** `9,274` bài báo tiếng Việt (thuộc 7 chuyên mục, từ 3 nguồn báo uy tín)
- **Số cột mỗi biến thể:** `10` cột chuẩn hóa
- **Đường dẫn Manifest:** `data/processed/dataset_manifest.json` (hoặc `configs/dataset_manifest.json`)
- **Random Seed:** `42`

---

## 2. Các lệnh thao tác 

### 2.1. Xác thực tính toàn vẹn của Dataset v1
Bất kỳ thành viên nào sau khi kéo code hoặc dữ liệu về đều có thể chạy lệnh sau để kiểm tra xem dữ liệu trên máy mình có đúng chuẩn và khớp hash không:
```bash
python -m src.preprocessing.freeze_dataset --verify
```

### 2.2. Tái tạo toàn bộ 4 biến thể từ dữ liệu thô
```bash
python -m src.preprocessing.build_variants
```

### 2.3. Chạy toàn bộ kiểm thử tự động
```bash
pytest
```
---

## 3. Bảng tra cứu nhanh mã băm SHA-256

| Tên File | Đường dẫn | Dung lượng | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| **Raw Data** | `data/raw/news_articles.csv` | 42,506,167 bytes | `548b35953368857373e8f26639531b96ab83b742ed6432bff0b6462661b5077a` |
| **Dedup Interim** | `data/interim/news_articles_deduplicated.csv` | 42,361,876 bytes | `8d4ebfade830c4403d90eba8389407232437327f30e51125938b4d7d6e8ce9fa` |
| **P0 Minimal** | `data/processed/p0_minimal.csv` | 123,974,695 bytes | `8565456f294e2fdec1535070105989b43ecd15d42abcde8e80e67eafa1ccc31e` |
| **P1 Normalized** | `data/processed/p1_normalized.csv` | 123,762,909 bytes | `5fbacd7b0d23b21eeba5f1a4b4f9a4ae779b0a13b57b9bb0dadd23c29cba0f86` |
| **P2 Segmented** | `data/processed/p2_segmented.csv` | 125,460,905 bytes | `be99ead1f6608976c86b172ad885e2676a0d25f6f55e2d732a9f15cf62a2add5` |
| **P3 No Stopwords** | `data/processed/p3_no_stopwords.csv` | 105,285,313 bytes | `663a34f309ef1f6d8d2dc61a68fd041d5f7aac71aa930dccb74841847a7d6710` |
| **Preprocessing Config** | `configs/preprocessing.yaml` | 1,651 bytes | `6861f73c35506083b6a0dfb72a8673df40e1323da0374add8809a4bd4a8ca155` |
| **Stopwords List** | `configs/vietnamese_stopwords.txt` | 4,506 bytes | `bbee33410d40312c494171e413b10e2edfa13f5793a4cab75370980dfcfd3b51` |
| **Split Config** | `configs/split.yaml` | 333 bytes | `cf219a74009721d98ba7f2e36178edc69e22006f9255c363ccc0649e827ea8af` |

