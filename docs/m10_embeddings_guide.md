# Hướng dẫn tạo Sentence Embeddings — Task M10

## 1. Mục tiêu

M10 tạo vector ngữ nghĩa (*sentence embedding*) cho các bài báo thuộc tập train của Dataset v1.0. Mỗi bài được biểu diễn bằng một vector số để bước phân cụm có thể so sánh mức độ gần nhau về nội dung.

M10 thực hiện:

1. Đọc đúng tập train theo split chính thức của nhóm.
2. Tạo văn bản đầu vào từ tiêu đề và nội dung bài báo.
3. Tạo embedding bằng hai mô hình MiniLM và MPNet.
4. Với mỗi mô hình, thử hai kiểu đầu vào: chỉ tiêu đề và tiêu đề cộng 60 từ đầu nội dung.
5. Lưu vector cùng `article_id`, tên mô hình, phiên bản dữ liệu và metadata cấu hình.
6. Tải lại artifact để kiểm tra dữ liệu được lưu đúng.

M10 tạo embedding. Việc giảm chiều bằng UMAP và phân cụm bằng HDBSCAN được thực hiện ở M11.

## 2. Dữ liệu đầu vào

Các file chính:

```text
data/processed/p0_minimal.csv
data/splits/split_manifest.csv
configs/embeddings.yaml
```

Ý nghĩa:

- `p0_minimal.csv`: dữ liệu bài báo đã được xử lý ở mức P0.
- `split_manifest.csv`: xác định mỗi `article_id` thuộc tập train, validation hay test.
- `embeddings.yaml`: cấu hình dữ liệu, mô hình, kiểu đầu vào và vị trí lưu artifact.

Hai cột văn bản được sử dụng:

```text
processed_title
processed_content
```

Runner dùng `load_split_dataset(..., "train")` để lấy đúng **6.493 bài thuộc tập train**. Mẫu lấy ngẫu nhiên từ toàn bộ dataset không được dùng làm kết quả M10 chính thức vì có thể chứa bài thuộc validation hoặc test.

## 3. Vì sao sử dụng dữ liệu P0?

P0 đã xử lý các vấn đề văn bản cơ bản như Unicode, thẻ HTML, ký tự điều khiển và khoảng trắng thừa, đồng thời vẫn giữ cấu trúc câu tương đối tự nhiên.

MiniLM và MPNet tạo embedding theo ngữ cảnh văn bản. Vì vậy, P0 là đầu vào tham chiếu phù hợp để bắt đầu thí nghiệm.

Các biến thể khác có mục đích xử lý riêng:

- **P1:** chuẩn hóa thêm văn bản. Có thể thử sau, nhưng chưa thể kết luận tốt hơn P0 nếu chưa đánh giá.
- **P2:** phân đoạn từ, có thể biểu diễn từ ghép bằng dấu gạch dưới. MiniLM và MPNet đã có tokenizer riêng nên không mặc nhiên cần kiểu biểu diễn này.
- **P3:** loại từ dừng. Việc bỏ từ có thể làm thay đổi ngữ cảnh của câu.

Chọn P0 **không có nghĩa P0 chắc chắn tốt nhất**. Nếu thử P1, P2 hoặc P3, cần giữ nguyên tập bài, mô hình và các tham số khác, rồi so sánh theo cùng quy trình đánh giá.

## 4. Hai mô hình được chọn

### MiniLM

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

MiniLM tạo vector **384 chiều** cho mỗi bài. Mô hình được dùng làm cấu hình tham chiếu với chi phí tính toán và lưu trữ thấp hơn MPNet trong lần chạy của nhóm.

### MPNet

```text
sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

MPNet tạo vector **768 chiều** cho mỗi bài. Mô hình cung cấp một cách biểu diễn khác để đối chiếu với MiniLM.

Vector nhiều chiều hơn **không tự động** có nghĩa chất lượng phân cụm tốt hơn. Việc chọn mô hình cần dựa trên nội dung các cụm, kết quả đánh giá, thời gian chạy và chi phí lưu trữ.

## 5. Bốn cấu hình embedding

M10 có **hai mô hình × hai kiểu đầu vào = bốn cấu hình**:

| Cấu hình | Mô hình | Văn bản đầu vào | Số chiều |
| --- | --- | --- | ---: |
| `minilm_title_content` | MiniLM | Tiêu đề + 60 từ đầu nội dung | 384 |
| `mpnet_title_content` | MPNet | Tiêu đề + 60 từ đầu nội dung | 768 |
| `minilm_title_only` | MiniLM | Chỉ tiêu đề | 384 |
| `mpnet_title_only` | MPNet | Chỉ tiêu đề | 768 |

Với cấu hình `title_content`, văn bản đầu vào được tạo từ:

```text
processed_title + ". " + 60 từ đầu của processed_content
```

Với cấu hình `title_only`, đầu vào chỉ gồm:

```text
processed_title
```

Trong code hiện tại, `content_words: 60` nghĩa là lấy tối đa 60 phần tử đầu sau khi tách nội dung theo khoảng trắng. Đây là **tham số thí nghiệm của nhóm**, không phải giá trị mặc định của mô hình và không phải toàn bộ bài báo.

`content_words: 0` tạo đầu vào chỉ gồm tiêu đề.

## 6. Số bài và số chiều vector

Tập train có **6.493 bài**, nên mỗi cấu hình tạo **6.493 vector**:

```text
MiniLM, title + content: (6493, 384)
MPNet, title + content:  (6493, 768)
MiniLM, title-only:      (6493, 384)
MPNet, title-only:       (6493, 768)
```

- `6.493` là số bài được mã hóa.
- `384` và `768` là số giá trị trong mỗi vector, do mô hình tương ứng quyết định.

Hai cấu hình dùng cùng một mô hình có cùng **số chiều**, nhưng giá trị trong vector khác nhau vì văn bản đầu vào khác nhau.

`expected_dimensions` trong YAML chỉ dùng để **kiểm tra** đầu ra. Thay số này trong YAML không làm mô hình đổi số chiều vector.

## 7. Quy trình so sánh có kiểm soát

### So sánh hai mô hình

Có thể so sánh:

```text
MiniLM, tiêu đề + 60 từ nội dung
với
MPNet, tiêu đề + 60 từ nội dung
```

Hai mô hình sử dụng cùng Dataset v1.0, cùng tập train, thứ tự `article_id`, biến thể P0 và cách ghép văn bản. Yếu tố thay đổi là **mô hình embedding**.

Cũng có thể so sánh MiniLM title-only với MPNet title-only theo cùng nguyên tắc.

### So sánh hai kiểu đầu vào (*input ablation*)

Trong từng mô hình, so sánh:

```text
MiniLM title-only ↔ MiniLM title + 60 từ nội dung
MPNet title-only  ↔ MPNet title + 60 từ nội dung
```

Trong mỗi cặp, giữ nguyên mô hình, tập train và tham số mã hóa. Yếu tố thay đổi là **lượng văn bản đầu vào**.

M10 tạo các bộ vector để phục vụ so sánh. Kết luận cấu hình nào tốt hơn cần dựa trên cùng quy trình phân cụm và đánh giá. Không dùng tập test để chọn cấu hình.

## 8. Cấu hình dữ liệu và mã hóa

Các thiết lập chính trong `configs/embeddings.yaml`:

```yaml
dataset:
  path: "data/processed/p0_minimal.csv"
  split_manifest_path: "data/splits/split_manifest.csv"
  split: "train"
  data_version: "v1.0"
  sample_size: null
  sample_random_state: 42
  title_col: "processed_title"
  content_col: "processed_content"

encoding:
  batch_size: 16
  normalize_embeddings: true
```

Ý nghĩa:

- `split: "train"`: chỉ mã hóa bài thuộc tập train chính thức.
- `sample_size: null`: dùng toàn bộ 6.493 bài train.
- `sample_random_state: 42`: cố định cách lấy mẫu nếu dùng `sample_size` để chạy thử.
- `batch_size: 16`: đưa văn bản vào mô hình theo từng nhóm 16 bài.
- `normalize_embeddings: true`: chuẩn hóa độ dài vector về gần 1.
- `data_version: "v1.0"`: ghi nhận phiên bản dữ liệu trong artifact.

Runner hiện yêu cầu `normalize_embeddings: true` vì hàm `create_embeddings()` luôn gọi bộ mã hóa với thiết lập chuẩn hóa này.

Mẫu 300 bài trước đây chỉ dùng để kiểm tra kỹ thuật. Bốn artifact M10 chính thức được tạo trên **toàn bộ 6.493 bài train**.

## 9. Code và luồng xử lý

Các file liên quan:

```text
src/data/dataset_loader.py
src/features/embeddings.py
experiments/hdbscan/run_m10_embeddings.py
configs/embeddings.yaml
tests/test_embeddings.py
```

Vai trò:

- `dataset_loader.py`: đọc Dataset v1.0 và chọn bài theo split manifest.
- `embeddings.py`: tạo văn bản, gọi mô hình, kiểm tra vector, lưu và tải artifact.
- `run_m10_embeddings.py`: đọc YAML và chạy các cấu hình có tên trong `RUN_NAMES`.
- `test_embeddings.py`: kiểm tra logic bằng encoder giả, không cần tải mô hình thật.

Luồng xử lý:

```text
P0 + split_manifest
    → chọn 6.493 bài train
    → tạo văn bản theo từng cấu hình
    → mã hóa bằng MiniLM hoặc MPNet
    → giữ vector khớp với article_id
    → lưu vector và metadata vào .npz
    → tải lại để kiểm tra
```

Dù runner nằm trong thư mục `experiments/hdbscan/`, M10 ở đây chỉ **tạo embedding**, chưa chạy HDBSCAN.

## 10. Chạy test

Từ thư mục gốc repository, chạy:

```powershell
python -B -m pytest -p no:cacheprovider tests/test_embeddings.py -v
```

Kết quả đã kiểm tra: **5 passed**.

Năm test kiểm tra:

1. Số vector, số chiều và thứ tự `article_id`.
2. Từ chối vector chứa `NaN`.
3. Từ chối `article_id` trùng.
4. Dùng đúng các cột P0 `processed_title` và `processed_content`.
5. Lưu rồi tải lại artifact mà vẫn giữ đúng **ID, vector và metadata**.

Các test dùng encoder giả nên chạy nhanh. Chúng kiểm tra logic code, chưa đánh giá chất lượng ngữ nghĩa của MiniLM hoặc MPNet.

Khi chạy chung test M10 và M11, kết quả đã ghi nhận là **14 passed**.

## 11. Chạy tạo embedding

Trước khi chạy trên toàn bộ tập train, kiểm tra YAML có:

```yaml
sample_size: null
```

Chạy từ thư mục gốc repository:

```powershell
python -B -m experiments.hdbscan.run_m10_embeddings
```

Runner chỉ thực thi các cấu hình có tên trong `RUN_NAMES`. Hiện danh sách gồm:

```python
RUN_NAMES = (
    "minilm_title_content",
    "mpnet_title_content",
    "minilm_title_only",
    "mpnet_title_only",
)
```

**Lưu ý:** chạy lại lệnh với danh sách trên sẽ mã hóa lại cả bốn cấu hình và ghi lại bốn artifact. Vì artifact đã được tạo, không cần chạy lại chỉ để xem vector.

Nếu chỉ muốn runner chạy một cấu hình, tuple một phần tử trong Python cần dấu phẩy:

```python
RUN_NAMES = ("mpnet_title_only",)
```

Viết `("mpnet_title_only")` khiến Python hiểu đây là chuỗi và vòng lặp đọc từng ký tự.

## 12. Bốn artifact đầu ra

Các file chính:

```text
artifacts/embeddings/m10_v1_train_minilm.npz
artifacts/embeddings/m10_v1_train_mpnet.npz
artifacts/embeddings/m10_v1_train_title_only_minilm.npz
artifacts/embeddings/m10_v1_train_title_only_mpnet.npz
```

Mỗi file `.npz` chứa:

- `article_ids`: ID bài báo theo thứ tự.
- `vectors`: ma trận embedding kiểu `float32`.
- `model_name`: mô hình tạo vector.
- `data_version`: phiên bản dữ liệu.
- `embedding_metadata_json`: thông tin cấu hình đã dùng để tạo artifact, được lưu dưới dạng chuỗi JSON.

Metadata ghi nhận tên cấu hình embedding, cấu hình dataset, cấu hình mã hóa, tên mô hình, `content_words` và số chiều mong đợi. Khi tải artifact bằng `load_embedding_artifact()`, metadata được đặt vào `DataFrame.attrs["embedding_metadata"]`.

Kích thước mong đợi:

```text
m10_v1_train_minilm.npz:            (6493, 384)
m10_v1_train_mpnet.npz:             (6493, 768)
m10_v1_train_title_only_minilm.npz: (6493, 384)
m10_v1_train_title_only_mpnet.npz:  (6493, 768)
```

Trong một artifact, `article_ids[i]` phải tương ứng với `vectors[i]`. File `.npz` là dữ liệu nhị phân, nên không mở như file văn bản thông thường.

## 13. Xem kích thước và các chỉ số của vector

Các lệnh trong mục này **chỉ đọc artifact**. Chúng không tải mô hình, không tạo lại embedding và không tạo file mới.

### Kiểm tra cả bốn artifact

Chạy nguyên khối lệnh sau trong PowerShell tại thư mục gốc repository:

```powershell
@'
import json
from pathlib import Path

import numpy as np

files = {
    "MiniLM title + content": "m10_v1_train_minilm.npz",
    "MPNet title + content": "m10_v1_train_mpnet.npz",
    "MiniLM title-only": "m10_v1_train_title_only_minilm.npz",
    "MPNet title-only": "m10_v1_train_title_only_mpnet.npz",
}

all_ids = []

for label, filename in files.items():
    path = Path("artifacts/embeddings") / filename

    with np.load(path, allow_pickle=False) as saved:
        ids = saved["article_ids"].copy()
        vectors = saved["vectors"].copy()
        model_name = str(saved["model_name"][0])
        data_version = str(saved["data_version"][0])
        metadata = json.loads(
            str(saved["embedding_metadata_json"][0])
        )

    norms = np.linalg.norm(vectors, axis=1)
    all_ids.append(ids)

    print("\n" + label)
    print("File:", path)
    print("Số bài, số chiều:", vectors.shape)
    print("ID không trùng:", len(np.unique(ids)) == len(ids))
    print("Có NaN hoặc vô cực:", not np.isfinite(vectors).all())
    print("Độ dài vector nhỏ nhất:", norms.min())
    print("Độ dài vector lớn nhất:", norms.max())
    print("Mô hình:", model_name)
    print("Phiên bản dữ liệu:", data_version)
    print("Cấu hình:", metadata["embedding_run"])
    print("Số từ nội dung:", metadata["content_words"])

print(
    "\nCả bốn file có cùng article_id và thứ tự:",
    all(
        np.array_equal(all_ids[0], ids)
        for ids in all_ids[1:]
    ),
)
'@ | python -B -
```

Kết quả cần kiểm tra:

- MiniLM có kích thước `(6493, 384)`.
- MPNet có kích thước `(6493, 768)`.
- `ID không trùng` là `True`.
- `Có NaN hoặc vô cực` là `False`.
- Độ dài vector gần `1` vì embedding đã được chuẩn hóa.
- `Phiên bản dữ liệu` là `v1.0`.
- Cả bốn file có cùng `article_id` theo cùng thứ tự.
- `Số từ nội dung` là `60` cho `title_content` và `0` cho `title_only`.

### Xem đầy đủ vector của một bài

Thay `article_id` nếu muốn xem bài khác:

```powershell
@'
import numpy as np

path = "artifacts/embeddings/m10_v1_train_minilm.npz"
article_id = "TTR_00278e9af821"

with np.load(path, allow_pickle=False) as saved:
    ids = saved["article_ids"]
    vectors = saved["vectors"]

positions = np.where(ids == article_id)[0]
if len(positions) != 1:
    raise ValueError("Không tìm thấy đúng một article_id")

vector = vectors[int(positions[0])]

print("article_id:", article_id)
print("Số chiều:", len(vector))
for index, value in enumerate(vector):
    print(f"{index:03d}: {value:.8f}")
'@ | python -B -
```

Với MiniLM, lệnh in các vị trí từ `000` đến `383`. Muốn xem MPNet, đổi `path` sang artifact MPNet; khi đó lệnh in từ `000` đến `767`.

Từng số trong vector là một tọa độ của cách biểu diễn văn bản, không có nhãn chủ đề riêng như “thể thao” hay “sức khỏe”.

## 14. Kiểm tra thủ công các bài gần nhau

Có thể chọn một bài thuộc tập train rồi tìm năm bài có vector gần nhất trong từng cấu hình. Việc này giúp phát hiện sớm những bài gần nhau về câu chữ nhưng không thật sự cùng chủ đề.

Một bài gốc từng được xem là:

```text
article_id: TTR_00278e9af821
Tiêu đề: Đề xuất ‘quét’ cửa hàng trực tuyến, ngăn tận gốc thuốc lá điện tử
```

Trong lần xem mẫu trước, cả bốn cấu hình đều tìm được một số bài có liên quan. Một số bài khác gần về cách diễn đạt, chẳng hạn cùng dùng từ “đề xuất” hoặc “cấm”, nhưng không cùng chủ đề thuốc lá.

Đây là **kiểm tra định tính trên một bài**, chưa đủ để kết luận MiniLM hay MPNet tốt hơn. Không so trực tiếp điểm cosine của hai mô hình như một thước đo chất lượng chung. Các artifact đã được đưa vào cùng quy trình UMAP/HDBSCAN của M11 để quan sát cụm và bài mang nhãn `-1`.

## 15. Một số tình huống thường gặp

### MPNet chạy lâu hơn MiniLM

Dòng `Loading weights: 100%` chỉ cho biết trọng số đã được tải. Chương trình vẫn phải mã hóa các bài báo. Đợi đến khi runner in `Đã lưu và tải lại` để xác nhận artifact đã hoàn tất.

### Cảnh báo chưa có `HF_TOKEN`

Nếu chương trình vẫn tải được mô hình và tiếp tục tạo vector, cảnh báo này không cho thấy artifact bị hỏng.

### `KeyError: 'mpnet_title_only'`

Runner yêu cầu cấu hình `mpnet_title_only`, nhưng YAML chưa có mục cùng tên trong `runs:`. Kiểm tra tên, mức thụt đầu dòng và lưu YAML trước khi chạy lại.

### `KeyError: 'm'`

Nếu chỉ chạy một cấu hình, tuple cần dấu phẩy:

```python
RUN_NAMES = ("mpnet_title_only",)
```

Viết `("mpnet_title_only")` khiến Python hiểu là chuỗi và vòng lặp đọc từng ký tự.

### Artifact hiển thị ký tự lạ trong VS Code

`.npz` là file nhị phân. Không chỉnh nội dung bằng trình soạn thảo văn bản; dùng NumPy như ví dụ ở mục 13 để xem ID và vector.

### Terminal hiển thị sai tiếng Việt

Trong PowerShell, có thể đặt:

```powershell
$env:PYTHONUTF8 = "1"
```

Sau đó chạy lại lệnh cần xem. Cách hiển thị chữ trong terminal không tự làm thay đổi vector đã lưu.

## 16. Trạng thái bàn giao và bước tiếp theo

Các file code, cấu hình, test và tài liệu M10:

```text
configs/embeddings.yaml
src/features/embeddings.py
experiments/hdbscan/run_m10_embeddings.py
tests/test_embeddings.py
docs/m10_embeddings_guide.md
```

M10 đã tạo embedding bằng **hai mô hình trên hai kiểu đầu vào** cho cùng tập train, kiểm tra ID, vector và metadata. M11 đã sử dụng bốn artifact này để chạy các cấu hình UMAP/HDBSCAN. Việc chọn cấu hình phù hợp cần dựa thêm trên nội dung cụm, phân tích bài nhãn `-1` và quy trình đánh giá chung của nhóm.