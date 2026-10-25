# vietnamese-news-topic-discovery
Vietnamese news topic discovery using five NLP methods
## Chạy ứng dụng bằng Docker

### Yêu cầu

- Cài Docker Desktop và bật Docker Engine.
- Cài Git và Git LFS.
- Có mạng khi build image và khi tải embedding model lần đầu.
- Có đủ RAM và dung lượng đĩa để nạp các mô hình NLP.

Không cần cài Python hoặc các thư viện Python trực tiếp trên máy: Docker cài chúng từ `requirements.txt`.

### 1. Lấy mã nguồn và dữ liệu

Mở PowerShell:

```powershell
git clone https://github.com/lunatran141/vietnamese-news-topic-discovery.git
cd vietnamese-news-topic-discovery
git switch develop
git lfs install
git lfs pull
```

Nếu đã clone repo trước đó:

```powershell
git switch develop
git pull origin develop
git lfs pull
```

**Lưu ý về BERTopic:** model E1 cần có tại
`artifacts/bertopic/experiments/e1_vietnamese_sbert/model/`.
Thư mục model này hiện không có trong GitHub; cần nhận từ người phụ trách
BERTopic và đặt đúng đường dẫn trên trước khi dùng chức năng dự đoán BERTopic.
Có thể kiểm tra bằng:

```powershell
Test-Path .\artifacts\bertopic\experiments\e1_vietnamese_sbert\model\config.json
```

Kết quả phải là `True`. Không tự huấn luyện lại model để thay thế model E1
đã dùng trong báo cáo.

### 2. Khởi chạy API và Streamlit

Chạy tại thư mục gốc repository:

```powershell
docker compose up --build -d
docker compose ps
```

Docker tạo hai dịch vụ:

- Giao diện Streamlit: http://localhost:8501
- API và tài liệu endpoint: http://localhost:8000/docs
- Kiểm tra API: http://localhost:8000/health

Lần build đầu tiên có thể lâu vì Docker phải tải và cài các thư viện.
Lần dự đoán đầu tiên cũng có thể lâu vì model được nạp vào bộ nhớ.

### 3. Kiểm tra chương trình

```powershell
Invoke-RestMethod http://localhost:8000/health
docker compose ps
docker compose logs --tail=100 api
docker compose logs --tail=100 app
```

API hoạt động khi `/health` trả về `{"status":"ok"}`. Sau đó mở
http://localhost:8501, vào trang bài báo mới, nhập tiêu đề/nội dung và
chọn mô hình để dự đoán.

Có thể thử trực tiếp endpoint `/predict`:

```powershell
$body = @{
    title = "Giá vàng trong nước tăng"
    content = "Giá vàng miếng và giá vàng nhẫn tăng theo diễn biến thị trường."
    model = "kmeans"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://localhost:8000/predict" `
    -Method Post `
    -ContentType "application/json; charset=utf-8" `
    -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

Giá trị `model` có thể là `kmeans`, `nmf`, `lda`, `hdbscan` hoặc `bertopic`.
Mỗi mô hình cần các artifact/model đã lưu tương ứng. HDBSCAN có thể cần tải
embedding model từ Hugging Face ở lần chạy đầu. Nếu thiếu model hoặc không có
mạng để tải, API sẽ báo lỗi thay vì huấn luyện lại.

### 4. Dừng hoặc khởi động lại

```powershell
docker compose down
docker compose up -d
```

Khi cập nhật code hoặc `requirements.txt`:

```powershell
git pull origin develop
git lfs pull
docker compose up --build -d
```

Không dùng `docker compose down -v` nếu chưa kiểm tra các volume/dữ liệu cần giữ.

### 5. Nếu cổng 8000 hoặc 8501 đang được sử dụng

Trong PowerShell, đặt cổng khác trước khi chạy:

```powershell
$env:API_PORT = "8001"
$env:STREAMLIT_PORT = "8502"
docker compose up --build -d
```

Khi đó mở giao diện tại http://localhost:8502 và API tại
http://localhost:8001. Streamlit trong Docker vẫn kết nối API qua mạng nội bộ
Docker, nên không cần đổi `API_BASE_URL`.

### 6. Kết quả E08

Kết quả benchmark đã có trong `results/runtime_benchmark/`. **Không cần chạy
lại benchmark để mở giao diện.** Nếu muốn tái lập E08 trên máy khác, xem
`docs/e08_runtime_benchmark_guide.md`. Kết quả đo trên máy khác phải ghi
riêng cấu hình CPU/GPU và không thay trực tiếp số liệu của đợt đo gốc.
