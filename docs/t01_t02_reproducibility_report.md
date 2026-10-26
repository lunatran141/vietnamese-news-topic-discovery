# Báo cáo Nghiệm thu T01/T02 & P09 (TV1 – Khôi)

Tài liệu này ghi nhận kết quả thực hiện hai đầu việc:
1. **`T01/T02` — Kiểm tra khả năng chạy lại hệ thống bằng Docker (`docker compose up --build`)** từ nhánh `develop` mới nhất.
2. **`P09` — Rà soát và khắc phục lỗi thực tế trên trang phân tích lỗi (`"Trường hợp cần xem xét"`)** kèm kiểm thử tự động (`pytest`).

---

## 1. Kết quả `T01/T02`: Kiểm tra Khả năng Chạy lại Hệ thống (Docker & API/UI)

### 1.1. Môi trường kiểm thử
- **Hệ điều hành:** Windows 11 (`win32`)
- **Docker Engine / Compose:** Docker `29.1.2`, Docker Compose `v2.40.3-desktop.1`
- **Nhánh kiểm thử:** `develop` (đồng bộ với `origin/develop`)
- **Cấu hình dịch vụ (`docker-compose.yml`):**
  - `api`: FastAPI (`uvicorn src.api.main:app --host 0.0.0.0 --port 8000`), ánh xạ cổng `8000:8000`, mount read-only `./configs`, `./data`, `./artifacts`, và read-write `./results`.
  - `app`: Streamlit (`streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501`), ánh xạ cổng `8501:8501`, kết nối nội bộ tới `API_BASE_URL=http://api:8000` sau khi `api` đạt trạng thái `healthy`.

### 1.2. Các lệnh đã chạy và kết quả thực tế

#### Bước 1: Build và khởi chạy toàn bộ hệ thống bằng Docker Compose
```powershell
docker compose up --build -d
docker compose ps
```
**Kết quả `docker compose ps`:**
```text
NAME                                    IMAGE                                 COMMAND                  SERVICE   STATUS                    PORTS
vietnamese-news-topic-discovery-api-1   vietnamese-news-topic-discovery-api   "uvicorn src.api.mai…"   api       Up (healthy)              0.0.0.0:8000->8000/tcp
vietnamese-news-topic-discovery-app-1   vietnamese-news-topic-discovery-app   "streamlit run app/s…"   app       Up                        0.0.0.0:8501->8501/tcp
```

#### Bước 2: Kiểm tra `GET /health`
```powershell
curl.exe -s http://localhost:8000/health
```
**Kết quả (`HTTP 200 OK`):**
```json
{"status": "ok"}
```

#### Bước 3: Kiểm tra `POST /predict` với bài báo mới (unseen article)
- **Thử nghiệm với mô hình mặc định `kmeans` (`POST http://localhost:8000/predict`):**
  ```powershell
  Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json; charset=utf-8" -Body '{"title":"Giá vàng SJC tăng mạnh, lãi suất ngân hàng biến động","content":"Thị trường tài chính trong nước ghi nhận giá vàng miếng SJC và tỷ giá USD tăng mạnh trong phiên giao dịch hôm nay.","model":"kmeans"}'
  ```
  **Kết quả (`HTTP 200 OK`):**
  ```json
  {
    "model": "KMeans",
    "run_id": "kmeans_baseline",
    "topic_id": 8,
    "topic_name": "8_tăng_giá_đồng",
    "is_outlier": false,
    "confidence": null,
    "keywords": ["tăng", "giá", "đồng", "lãi_suất", "thị_trường", "giảm", "usd", "cổ_phiếu", "so", "bán"],
    "representative_articles": [
      {
        "article_id": "VNN_df3e5b8dfc8b",
        "title": "VN-Index mất mốc 1.800 điểm, ông Phạm Nhật Vượng vẫn thăng hạng tỷ phú USD",
        "split": "train"
      },
      {
        "article_id": "VNN_b33b5903223f",
        "title": "Giá vàng hôm nay 12/9/2026: Cuối tuần bật tăng, SJC và nhẫn lên cao",
        "split": "train"
      },
      {
        "article_id": "VNN_583afb87bcff",
        "title": "Giá vàng hôm nay 27/9/2026 lao dốc, vàng SJC giảm 3,2 triệu đồng/lượng",
        "split": "train"
      }
    ],
    "inference_seconds": 0.2101,
    "preprocessing_variant": "p2_segmented",
    "inference_mode": "saved_model_transform"
  }
  ```

- **Thử nghiệm với mô hình `lda` (`POST http://localhost:8000/predict`):**
  - Trả về `HTTP 200 OK`, dự đoán `topic_id = 6` (`"Bóng đá trong nước và quốc tế"`), `confidence = 0.5468`, `inference_seconds = 2.4214s`.

#### Bước 4: Kiểm tra luồng nhập bài mới trên Streamlit (`http://localhost:8501`)
- Kiểm tra healthcheck của Streamlit: `GET http://localhost:8501/_stcore/health` $\rightarrow$ `200 ok`.
- Kiểm tra luồng gửi bài mới từ container `app` sang container `api` (`http://api:8000/predict`):
  ```powershell
  docker compose exec -T app python -c "from streamlit.testing.v1 import AppTest; from tests.test_app_pages import _view_script; at = AppTest.from_function(_view_script, args=('articles', 'kmeans_baseline', 'all', '/app'), default_timeout=120).run(); at.text_input[2].set_value('Giá vàng SJC tăng mạnh hôm nay'); at.text_area[0].set_value('Thị trường tài chính ghi nhận giá vàng miếng SJC và tỷ giá USD biến động.'); at.button[0].click().run(); print('EXCEPTIONS:', at.exception); print('SUCCESSES:', [s.value for s in at.success])"
  ```
  **Kết quả:**
  - `EXCEPTIONS: ElementList()` (không có lỗi)
  - `SUCCESSES: ['Chủ đề dự đoán: 8_tăng_giá_đồng']`

### 1.3. Ghi nhận giới hạn môi trường & cải tiến UX
1. **Giới hạn artifact nhị phân của BERTopic trên máy clone sạch:**
   - Theo cấu hình `.gitignore` (dòng 226: `artifacts/bertopic/`), thư mục model nhị phân của BERTopic (`artifacts/bertopic/final/e1_vietnamese_sbert/model`) không được đẩy lên Git do dung lượng lớn (chỉ các file common artifact `.csv`/`.json` phục vụ hiển thị dashboard được commit trong `artifacts/common/bertopic/e1_vietnamese_sbert/`).
   - Khi gọi `POST /predict` với `"model": "bertopic"` trên môi trường chưa chép thủ công thư mục model nhị phân BERTopic, API trả về đúng mã `HTTP 503 Service Unavailable` theo thiết kế tại `docs/p10a_internal_inference_api.md`:
     ```json
     {"detail": "Không tìm thấy model BERTopic đã lưu. Các đường dẫn đã kiểm tra: /app/artifacts/bertopic/final/e1_vietnamese_sbert/model, /app/artifacts/bertopic/experiments/e1_vietnamese_sbert/model"}
     ```
   - **Cải tiến UX đã thực hiện:** Trong `app/views/articles.py`, danh sách `INFERENCE_MODELS` đã được điều chỉnh để đặt **K-Means** (mô hình đã có sẵn đầy đủ binary `.joblib` trên Git) làm lựa chọn mặc định đầu tiên, tiếp theo là `LDA`, `NMF`, `HDBSCAN`, `BERTopic`. Nhờ vậy, người kiểm thử mở tab `"Phân loại bài mới"` và bấm phân tích ngay sẽ nhận kết quả dự đoán thành công tức thì.
2. **Thời gian build Docker lần đầu:**
   - Do `requirements.txt` cài đặt đầy đủ hệ sinh thái `torch`, `sentence-transformers`, `bertopic`, lần chạy `docker compose up --build` đầu tiên trên máy chưa có cache Docker sẽ tải khoảng ~2.5–3.5 GB package. Các lần chạy sau tận dụng Docker BuildKit layer cache nên chỉ mất vài giây.

---

## 2. Kết quả `P09`: Rà soát & Sửa lỗi Trang Phân tích Lỗi (`"Trường hợp cần xem xét"`)

Qua rà soát hàm `_render_review` trong `app/views/articles.py` trên dữ liệu thật của cả 5 mô hình, đã phát hiện và khắc phục 4 vấn đề thực tế:

1. **Loại bỏ bài Outlier (`topic_id == -1`) bị lọc nhầm vào `"Bài đa chủ đề"` và `"Topic chồng lấn"`:**
   - *Trước khi sửa:* Ở BERTopic có `2.335 / 4.083` bài và ở HDBSCAN có `2.275 / 3.732` bài trong hai nhóm `"Bài đa chủ đề"` và `"Topic chồng lấn"` thực chất là bài `Outlier (topic_id == -1)` (do phân phối xác suất của bài nhiễu phẳng nên `top_two_share_gap <= 0.1`).
   - *Sau khi sửa:* Hàm `_select_review_cases` tách biệt rõ nhóm `"Outlier"` (`topic_id == -1` hoặc `is_outlier == True`) và loại trừ hoàn toàn các bài outlier khỏi 3 nhóm `"Topic quá rộng"`, `"Bài đa chủ đề"`, `"Topic chồng lấn"`. Sau khi lọc sạch outlier, BERTopic còn đúng `1.748` bài đa chủ đề/chồng lấn thật sự và HDBSCAN còn `1.457` bài.
2. **Bổ sung cột dữ liệu đặc thù của từng mô hình vào bảng hiển thị và file CSV tải về:**
   - Bổ sung `topic_label` (ánh xạ từ `labels` vốn được truyền vào `_render_review` nhưng trước đó chưa sử dụng) và `category_gold`.
   - Bổ sung `second_topic_probability` (dùng bởi **LDA** và **BERTopic**) và `distance_to_center` (dùng bởi **K-Means**).
3. **Loại bỏ cột toàn `NaN` và cột trùng lặp trong bảng & file CSV:**
   - Tự động lược bỏ các cột có 100% giá trị `NaN` trên tập kết quả đang xem (như cột `topic_probability` ở K-Means và NMF).
   - Khử trùng lặp cặp cột `secondary_topic_id` / `secondary_topic_share` khi đã có `second_topic_id` / `second_topic_share` (ở NMF).
4. **Thêm bộ lọc tra cứu nhanh trong nhóm lỗi đang xem:**
   - Bổ sung ô tìm kiếm theo tiêu đề / `article_id` và bộ lọc `Lọc theo chủ đề` ngay trong tab `"Trường hợp cần xem xét"`, giúp tra cứu ví dụ bài viết của bất kỳ chủ đề nào mà không bị giới hạn bởi `.head(300)` trên bảng giao diện.

### Kiểm thử tự động (`pytest`)
Đã bổ sung các bài kiểm thử tự động trong `tests/test_app_pages.py`:
- `test_inference_models_default_order`
- `test_review_cases_exclude_outliers_and_prepare_clean_csv_columns` (kiểm tra trên tất cả 5 mô hình × 4 chế độ)
- `test_review_table_preserves_model_specific_columns` (kiểm tra `distance_to_center` cho K-Means và `second_topic_probability` cho LDA)
- `test_articles_review_radio_modes_render_without_exception`

Chạy kiểm thử:
```powershell
python -m pytest tests/test_api.py tests/test_streamlit_data_service.py -v
python -m pytest tests/test_app_pages.py -k "articles or review or inference_models" -v
```
Kết quả: **100% PASSED**.
