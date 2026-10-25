# vietnamese-news-topic-discovery
Vietnamese news topic discovery using five NLP methods

Ứng dụng khám phá chủ đề tin tức tiếng Việt với K-Means, NMF, LDA, HDBSCAN và BERTopic. Giao diện Streamlit gọi API FastAPI để phân loại bài báo mới bằng các mô hình đã lưu.

## Chuẩn bị mã nguồn

```powershell
git clone https://github.com/lunatran141/vietnamese-news-topic-discovery.git
cd vietnamese-news-topic-discovery
git switch develop
git lfs install
git lfs pull
```

Nếu đã clone trước đó, chạy `git pull origin develop` và `git lfs pull` trong thư mục repository. Git LFS cần thiết cho một số dữ liệu và model. Chạy các lệnh dưới đây từ **thư mục gốc repository**.

> **Artifact BERTopic:** Model E1 tại `artifacts/bertopic/experiments/e1_vietnamese_sbert/model/` chưa được đưa lên GitHub. Để dùng chức năng dự đoán BERTopic trên máy mới, hãy nhận thư mục model này từ người phụ trách và đặt đúng đường dẫn. Trên PowerShell, dùng `Test-Path .\artifacts\bertopic\experiments\e1_vietnamese_sbert\model\config.json` để kiểm tra; kết quả phải là `True`. HDBSCAN cũng có thể cần kết nối mạng để tải embedding model lần đầu. Giao diện và các chức năng đọc artifact vẫn có thể mở khi chưa có model BERTopic.

## Cách 1: Chạy bằng Docker

Cài và mở Docker Desktop. Không cần cài Python hay từng thư viện Python trên máy; Docker image cài dependencies từ `requirements.txt`.

```powershell
docker compose up --build -d
docker compose ps
```

Mở giao diện tại <http://localhost:8501>. API ở <http://localhost:8000>; tài liệu endpoint ở <http://localhost:8000/docs>. Kiểm tra API:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Lần sau, nếu không đổi code hoặc dependencies, có thể chạy `docker compose up -d`. Xem lỗi và dừng chương trình:

```powershell
docker compose logs --tail=100 api
docker compose logs --tail=100 app
docker compose down
```

Nếu cổng 8000 hoặc 8501 đã được sử dụng, đặt cổng host khác trước khi chạy:

```powershell
$env:API_PORT = "8001"
$env:STREAMLIT_PORT = "8502"
docker compose up --build -d
```

Khi đó mở Streamlit tại <http://localhost:8502> và API tại <http://localhost:8001>.

## Cách 2: Chạy không dùng Docker (Windows PowerShell)

Cài Python 3.11 và Git LFS. Tạo môi trường riêng rồi cài thư viện một lần:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Mở **hai cửa sổ PowerShell**, đều ở thư mục gốc repository. Giữ cả hai cửa sổ chạy khi sử dụng ứng dụng.

Cửa sổ 1 — API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8001
```

Cửa sổ 2 — Streamlit:

```powershell
$env:API_BASE_URL = "http://127.0.0.1:8001"
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Mở <http://localhost:8501>. Kiểm tra API bằng `Invoke-RestMethod http://127.0.0.1:8001/health`. Nhấn `Ctrl+C` ở mỗi cửa sổ để dừng. Nếu cổng 8001 đã được sử dụng, đổi `--port` ở cửa sổ API và đổi cùng số cổng trong `API_BASE_URL`.

## Kiểm thử và kết quả E08

Khi chạy không dùng Docker, có thể kiểm tra API mà không cần tải model nặng:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_api.py -q
```

Kết quả benchmark E08 đã được lưu trong `results/runtime_benchmark/`; không cần chạy lại benchmark để mở giao diện. Cách tái lập phép đo và giới hạn so sánh được ghi tại [hướng dẫn E08](docs/e08_runtime_benchmark_guide.md).
