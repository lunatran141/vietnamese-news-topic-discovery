# Hướng dẫn tạo Sentence Embeddings - Task M10

## 1. Mục tiêu

Task M10 tạo vector ngữ nghĩa (*sentence embedding*) cho các bài báo thuộc tập train của Dataset v1.0. Mỗi bài báo được biểu diễn bằng một vector số để các bước phân cụm tiếp theo có thể so sánh mức độ gần nhau về nội dung.

M10 thực hiện:

1. Đọc đúng tập train theo split chính thức của nhóm.
2. Tạo văn bản đầu vào từ tiêu đề và nội dung bài báo.
3. Tạo embedding bằng hai mô hình MiniLM và MPNet.
4. Với mỗi mô hình, thử hai kiểu đầu vào: chỉ tiêu đề và tiêu đề cộng 60 từ đầu nội dung.
5. Lưu vector cùng `article_id`, tên mô hình và phiên bản dữ liệu.
6. Tải lại artifact để kiểm tra dữ liệu được lưu đúng.

M10 tạo embedding. Việc chạy HDBSCAN và đánh giá chất lượng cụm là bước tiếp theo.

## 2. Dữ liệu đầu vào

Các file chính:

```text
data/processed/p0_minimal.csv
data/splits/split_manifest.csv
configs/embeddings.yaml
```

Ý nghĩa:

- `p0_minimal.csv`: dữ liệu bài báo đã được xử lý ở mức P0.
- `split_manifest.csv`: xác định mỗi `article_id` thuộc train, val hay test.
- `embeddings.yaml`: cấu hình mô hình, văn bản đầu vào và vị trí lưu artifact.

Hai cột văn bản được sử dụng:

```text
processed_title
processed_content
```

Runner dùng `load_split_dataset(..., "train")` để lấy đúng **6.493 bài thuộc tập train**. Không lấy ngẫu nhiên bài từ toàn bộ Dataset v1.0 cho kết quả chính thức, vì cách đó có thể trộn bài thuộc val hoặc test.

## 3. Vì sao sử dụng dữ liệu P0?

P0 đã chuẩn hóa Unicode, loại thẻ HTML, ký tự điều khiển và khoảng trắng thừa. P0 vẫn giữ cấu trúc câu tự nhiên, chữ hoa/thường, dấu câu và các từ nối.

MiniLM và MPNet là mô hình tạo embedding theo ngữ cảnh văn bản. Vì vậy, P0 là đầu vào tham chiếu phù hợp để bắt đầu thử nghiệm.

Các biến thể còn lại có mục đích khác:

- **P1:** chuyển chữ thường và loại một số nội dung nhiễu. P1 có thể được thử thêm, nhưng chưa thể kết luận tốt hơn P0 nếu chưa đánh giá kết quả.
- **P2:** nối từ ghép như `công nghệ` thành `công_nghệ`. Cách biểu diễn này chủ yếu phục vụ các phương pháp dựa trên từ hoặc tần suất từ. MiniLM và MPNet đang dùng tokenizer riêng.
- **P3:** loại nhiều từ dừng. Dù pipeline giữ các từ phủ định như `không` và `chưa`, việc bỏ từ vẫn có thể thay đổi ngữ cảnh câu.

Chọn P0 **không có nghĩa P0 chắc chắn tốt nhất**. Nếu sau này thử P1, P2 hoặc P3, cần giữ nguyên tập train, mô hình và các tham số còn lại; chỉ thay đổi biến thể tiền xử lý rồi so sánh theo cùng quy trình đánh giá.

## 4. Hai mô hình được chọn

### MiniLM

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

MiniLM tạo vector **384 chiều** cho mỗi bài báo. Mô hình này được chọn làm cấu hình tham chiếu vì chi phí tính toán và lưu trữ thấp hơn MPNet.

### MPNet

```text
sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

MPNet tạo vector **768 chiều** cho mỗi bài báo. Mô hình này cung cấp một cách biểu diễn khác để đối chiếu với MiniLM.

Vector nhiều chiều hơn **không tự động** có nghĩa là phân cụm tốt hơn. Việc chọn mô hình cần dựa trên chất lượng cụm, thời gian chạy và chi phí lưu trữ ở các bước đánh giá tiếp theo.

Thông tin mô hình:

- [MiniLM trên Hugging Face](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- [MPNet trên Hugging Face](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)

## 5. Bốn cấu hình embedding

M10 có **hai mô hình × hai kiểu đầu vào = bốn cấu hình**:

- `minilm_title_content`: MiniLM, tiêu đề + 60 từ đầu nội dung, vector 384 chiều.
- `mpnet_title_content`: MPNet, tiêu đề + 60 từ đầu nội dung, vector 768 chiều.
- `minilm_title_only`: MiniLM, chỉ tiêu đề, vector 384 chiều.
- `mpnet_title_only`: MPNet, chỉ tiêu đề, vector 768 chiều.

Với cấu hình `title_content`, văn bản đầu vào của mỗi bài có dạng:

```text
processed_title + ". " + 60 từ đầu của processed_content
```

Với cấu hình `title_only`, văn bản đầu vào chỉ là:

```text
processed_title
```

Trong code hiện tại, `content_words: 60` nghĩa là lấy 60 phần tử đầu sau khi tách nội dung theo khoảng trắng. Đây là tham số thử nghiệm của nhóm, không phải giá trị mặc định của mô hình và cũng không phải toàn bộ nội dung bài báo.

`content_words: 0` tạo đầu vào chỉ gồm tiêu đề.

## 6. Phân biệt số bài và số chiều vector

Tập train có **6.493 bài**, nên mỗi cấu hình tạo **6.493 vector**:

```text
MiniLM, title + content: (6493, 384)
MPNet, title + content:  (6493, 768)
MiniLM, title-only:      (6493, 384)
MPNet, title-only:       (6493, 768)
```

- `6.493` là số bài thuộc tập train.
- `384` và `768` là số giá trị trong mỗi vector, do mô hình quyết định.

Hai cấu hình của cùng một mô hình có cùng kích thước vector, nhưng **giá trị bên trong vector có thể khác nhau** vì văn bản đầu vào khác nhau.

`expected_dimensions` trong YAML chỉ dùng để kiểm tra đầu ra. Thay số này trong YAML không làm mô hình tạo ra vector có số chiều khác.

## 7. Quy trình so sánh có kiểm soát

M10 chuẩn bị hai kiểu so sánh riêng.

### 7.1. So sánh mô hình

So sánh:

```text
MiniLM, title + 60 từ nội dung
với
MPNet, title + 60 từ nội dung
```

Giữ nguyên Dataset v1.0, tập train, thứ tự `article_id`, dữ liệu P0, cách tạo văn bản và thiết lập chuẩn hóa vector. Yếu tố thay đổi là **mô hình embedding**.

Cũng có thể so sánh MiniLM title-only với MPNet title-only theo cùng nguyên tắc: hai mô hình phải nhận đúng cùng các tiêu đề.

### 7.2. So sánh văn bản đầu vào (*input ablation*)

So sánh trong từng mô hình:

```text
MiniLM title-only ↔ MiniLM title + 60 từ nội dung
MPNet title-only  ↔ MPNet title + 60 từ nội dung
```

Trong mỗi cặp, giữ nguyên mô hình, tập train, thứ tự `article_id` và các tham số mã hóa. Yếu tố thay đổi là **lượng văn bản đầu vào**.

M10 chỉ tạo các bộ vector để phục vụ so sánh. Kết luận cấu hình nào tốt hơn cần dựa trên **cùng quy trình phân cụm và đánh giá** ở bước sau. Không dùng tập test để chọn cấu hình.

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

- `split: "train"`: chỉ mã hóa tập train theo split chính thức.
- `sample_size: null`: sử dụng toàn bộ 6.493 bài train.
- `sample_random_state: 42`: cố định cách lấy mẫu nếu đặt `sample_size` là một số để chạy thử.
- `batch_size: 16`: mã hóa văn bản theo từng nhóm 16 bài.
- `normalize_embeddings: true`: chuẩn hóa vector.
- `data_version: "v1.0"`: ghi nhận phiên bản dữ liệu trong artifact.

Mẫu 300 bài chỉ dùng để kiểm tra kỹ thuật. Các artifact M10 chính thức được tạo trên toàn bộ 6.493 bài train.

## 9. Các file code và luồng xử lý

Các file liên quan:

```text
src/data/dataset_loader.py
src/features/embeddings.py
experiments/hdbscan/run_m10_embeddings.py
tests/test_embeddings.py
```

Vai trò:

- `dataset_loader.py`: đọc Dataset v1.0 và chọn bài theo split manifest.
- `embeddings.py`: tạo văn bản, gọi mô hình, kiểm tra vector, lưu và tải artifact.
- `run_m10_embeddings.py`: đọc YAML và chạy các cấu hình được liệt kê trong `RUN_NAMES`.
- `test_embeddings.py`: kiểm tra logic bằng encoder giả, không cần tải mô hình thật.

Luồng xử lý:

```text
P0 + split_manifest
    -> chọn 6.493 bài train
    -> tạo văn bản theo từng cấu hình
    -> mã hóa bằng MiniLM hoặc MPNet
    -> giữ vector khớp với article_id
    -> lưu artifact .npz
    -> tải lại và kiểm tra
```

Dù runner nằm trong thư mục `experiments/hdbscan/`, M10 ở đây chỉ **tạo embedding**, chưa chạy HDBSCAN.

## 10. Chạy unit test

Từ thư mục gốc repository, chạy:

```powershell
python -B -m pytest -p no:cacheprovider tests/test_embeddings.py -v
```

Trong lần kiểm tra đã thực hiện, kết quả là:

```text
5 passed
```

Các test kiểm tra:

1. Số vector, số chiều và thứ tự `article_id`.
2. Từ chối vector chứa `NaN`.
3. Từ chối `article_id` trùng.
4. Dùng đúng các cột P0 `processed_title` và `processed_content`.
5. Lưu rồi tải lại artifact mà vẫn giữ đúng ID và vector.

Các unit test dùng encoder giả nên chạy nhanh. Chúng kiểm tra logic code, chưa đánh giá chất lượng ngữ nghĩa của MiniLM hoặc MPNet.

## 11. Chạy tạo embedding

Trước khi chạy trên toàn bộ tập train, kiểm tra YAML có:

```yaml
sample_size: null
```

Từ thư mục gốc repository, chạy:

```powershell
python -B -m experiments.hdbscan.run_m10_embeddings
```

Runner chỉ thực thi các cấu hình có tên trong `RUN_NAMES`. Nếu chạy cả bốn cấu hình trong một lần, danh sách là:

```python
RUN_NAMES = (
    "minilm_title_content",
    "mpnet_title_content",
    "minilm_title_only",
    "mpnet_title_only",
)
```

**Lưu ý:** chạy lại lệnh với danh sách trên sẽ tính lại cả bốn bộ vector và ghi lại artifact. Vì bốn artifact đã được tạo, không cần chạy lại chỉ để xem các giá trị vector.

Khi `RUN_NAMES` chỉ có một mục, cú pháp Python phải có dấu phẩy:

```python
RUN_NAMES = ("mpnet_title_only",)
```

Nếu bỏ dấu phẩy, Python coi giá trị là chuỗi và vòng lặp sẽ đọc từng ký tự.

## 12. Các artifact đầu ra

Bốn file kết quả:

```text
artifacts/embeddings/m10_v1_train_minilm.npz
artifacts/embeddings/m10_v1_train_mpnet.npz
artifacts/embeddings/m10_v1_train_title_only_minilm.npz
artifacts/embeddings/m10_v1_train_title_only_mpnet.npz
```

Mỗi file `.npz` chứa:

- `article_ids`: ID các bài báo theo thứ tự;
- `vectors`: ma trận embedding;
- `model_name`: mô hình đã tạo vector;
- `data_version`: phiên bản dữ liệu.

Kích thước mong đợi:

```text
m10_v1_train_minilm.npz:            (6493, 384)
m10_v1_train_mpnet.npz:             (6493, 768)
m10_v1_train_title_only_minilm.npz: (6493, 384)
m10_v1_train_title_only_mpnet.npz:  (6493, 768)
```

Trong một artifact, `article_ids[i]` phải tương ứng với `vectors[i]`. File `.npz` là dữ liệu nhị phân, nên không mở như file văn bản thông thường.

## 13. Xem kích thước và các chỉ số của vector

Các lệnh trong mục này **chỉ đọc artifact**, không tải mô hình, không chạy lại embedding và không tạo file mới.

### 13.1. Kiểm tra cả bốn artifact

Chạy nguyên khối lệnh sau trong PowerShell tại thư mục gốc repository:

```powershell
@'
import numpy as np
from pathlib import Path

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

print("\nCả bốn file có cùng article_id và thứ tự:",
      all(np.array_equal(all_ids[0], ids) for ids in all_ids[1:]))
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

### 13.2. Xem đầy đủ vector của một bài

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

Với MiniLM, lệnh in đầy đủ các vị trí từ `000` đến `383`. Muốn xem MPNet, đổi `path` thành:

```python
path = "artifacts/embeddings/m10_v1_train_mpnet.npz"
```

MPNet sẽ in các vị trí từ `000` đến `767`. Muốn xem bản title-only, thay `path` bằng tên artifact title-only tương ứng.

Các số được in là **tọa độ của vector**. Từng số riêng lẻ không có nhãn chủ đề như “thể thao” hay “sức khỏe”. Cần xem cả vector và quan hệ giữa nhiều bài để đánh giá ý nghĩa của embedding.

## 14. Kiểm tra thủ công các bài gần nhau

Để xem embedding có phản ánh nội dung bài báo hay không, có thể chọn một bài thuộc tập train rồi tìm 5 bài có vector gần nhất trong từng cấu hình.

Ví dụ đã kiểm tra:

```text
article_id: TTR_00278e9af821
Tiêu đề: Đề xuất ‘quét’ cửa hàng trực tuyến, ngăn tận gốc thuốc lá điện tử
```

Khi chỉ xét các tiêu đề **rõ ràng liên quan đến thuốc lá**, kết quả quan sát được:

- **MiniLM, title + content:** khoảng 2/5 bài gần nhất liên quan.
- **MPNet, title + content:** khoảng 3/5 bài gần nhất liên quan.
- **MiniLM, title-only:** khoảng 2/5 bài gần nhất liên quan.
- **MPNet, title-only:** khoảng 2/5 bài gần nhất liên quan.

Cả bốn cấu hình đều tìm được một số bài liên quan. Trong ví dụ này, MPNet với đầu vào tiêu đề và nội dung đưa thêm bài về “khách sạn không thuốc lá” vào nhóm gần nhất. Một số bài khác chỉ giống về cách diễn đạt như “đề xuất”, “cấm” hoặc “trực tuyến”, nhưng không cùng chủ đề thuốc lá.

Đây là **kiểm tra định tính trên một bài**, chưa đủ để kết luận MPNet tốt hơn MiniLM. Không so trực tiếp điểm cosine của hai mô hình như một thước đo chất lượng. Các artifact sẽ được đưa vào cùng quy trình UMAP/HDBSCAN ở M11 để kiểm tra chất lượng cụm và bài bị gán nhãn nhiễu.

## 15. Xử lý tình huống thường gặp

### MPNet chạy lâu hơn MiniLM

MPNet có mô hình và vector đầu ra lớn hơn. Dòng `Loading weights: 100%` chỉ cho biết trọng số đã được tải; chương trình còn phải mã hóa bài báo. Chờ đến khi thấy dòng `Đã lưu và tải lại` để xác nhận hoàn tất.

### Cảnh báo chưa có `HF_TOKEN`

Với mô hình công khai, đây thường là cảnh báo về giới hạn tải từ Hugging Face. Nếu chương trình vẫn tải mô hình và tiếp tục tạo vector, cảnh báo này không làm hỏng kết quả.

### `KeyError: 'mpnet_title_only'`

Runner đã yêu cầu chạy `mpnet_title_only`, nhưng YAML chưa có mục cùng tên trong phần `runs:`. Kiểm tra tên, mức thụt đầu dòng và nhớ lưu YAML trước khi chạy lại.

### `KeyError: 'm'`

Nếu chỉ chạy một cấu hình, cần đặt dấu phẩy trong tuple:

```python
RUN_NAMES = ("mpnet_title_only",)
```

Viết `("mpnet_title_only")` khiến Python hiểu là chuỗi và vòng lặp đọc từng ký tự.

### Artifact hiển thị ký tự lạ trong VS Code

`.npz` là file nhị phân. Không chỉnh nội dung bằng trình soạn thảo văn bản; dùng các lệnh NumPy ở mục 13 để xem ID và vector.

### Terminal hiển thị sai tiếng Việt

Trong PowerShell, chạy:

```powershell
$env:PYTHONUTF8 = "1"
```

Sau đó chạy lại lệnh cần xem kết quả. Việc hiển thị sai chữ ở terminal không tự làm thay đổi vector đã lưu.

## 16. Bàn giao M10 và bước tiếp theo

Các file code, cấu hình, test và tài liệu:

```text
configs/embeddings.yaml
src/features/embeddings.py
experiments/hdbscan/run_m10_embeddings.py
tests/test_embeddings.py
docs/m10_embeddings_guide.md
```

Bốn file `.npz` là artifact sinh ra khi chạy. Trước khi commit, cần kiểm tra các file Git sẽ đưa lên và thống nhất với nhóm cách chia sẻ artifact. `.gitignore` hiện có quy tắc cho `artifacts/bertopic/`, nhưng chưa thấy quy tắc tương ứng cho `artifacts/embeddings/`.

M10 đã tạo embedding bằng **hai mô hình** trên **hai kiểu đầu vào**, cùng với kiểm tra `article_id` và tính hợp lệ của vector. Bước tiếp theo là sử dụng các artifact trong cùng quy trình HDBSCAN và đánh giá để chọn cấu hình phù hợp với dữ liệu nhóm.