from pathlib import Path

import numpy as np
import pandas as pd
import json

def create_embeddings(
    articles: pd.DataFrame,
    encoder,
    model_name: str = "unknown",
    data_version: str = "unspecified",
    content_words: int = 60,
    title_col: str = "title",
    content_col: str = "content",
    batch_size: int = 16
) -> pd.DataFrame:
    """
    Tạo một vector cho mỗi bài báo.
    Giữ nguyên thứ tự article_id. Có thể chọn title/content gốc hoặc
    processed_title/processed_content của P0, P1.
    """
    required = {"article_id", title_col, content_col}
    missing = required - set(articles.columns)
    if missing:
        raise ValueError(f"Thiếu cột: {sorted(missing)}")

    if articles.empty:
        raise ValueError("Bảng bài báo rỗng")

    if content_words < 0:
        raise ValueError("content_words phải >= 0")

    if batch_size < 1:
        raise ValueError("batch_size phải >= 1")

    article_ids = articles["article_id"]
    if article_ids.isna().any() or article_ids.astype(str).str.strip().eq("").any():
        raise ValueError("Có article_id bị thiếu")

    if article_ids.duplicated().any():
        raise ValueError("Có article_id trùng")

    texts = []
    for article_id, title_value, content_value in zip(
        article_ids,
        articles[title_col],
        articles[content_col]):
        title = "" if pd.isna(title_value) else str(title_value).strip()
        content = "" if pd.isna(content_value) else str(content_value).strip()

        short_content = " ".join(content.split()[:content_words])
        text = f"{title}. {short_content}".strip(" .")

        if not text:
            raise ValueError(f"Bài {article_id} không có văn bản")

        texts.append(text)

    vectors = np.asarray(
        encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True),
            dtype=np.float32)

    if vectors.ndim != 2 or vectors.shape[0] != len(articles):
        raise ValueError("Số vector không khớp số bài báo")

    if vectors.shape[1] == 0:
        raise ValueError("Vector không có chiều")

    if not np.isfinite(vectors).all():
        raise ValueError("Vector chứa NaN hoặc Infinity")

    return pd.DataFrame({
        "article_id": article_ids.tolist(),
        "embedding": list(vectors),
        "model_name": model_name,
        "data_version": data_version})

def save_embedding_artifact(
    table: pd.DataFrame,
    path: str | Path,
    *,
    metadata: dict | None = None,
) -> None:
    """Lưu ID, vector và thông tin cấu hình embedding."""
    required = {"article_id", "embedding", "model_name", "data_version"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Thiếu cột artifact: {sorted(missing)}")
    if table.empty:
        raise ValueError("Bảng embedding rỗng")


    ids = table["article_id"]
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
        raise ValueError("Có article_id bị thiếu")

    article_ids = ids.astype(str).tolist()
    if len(set(article_ids)) != len(article_ids):
        raise ValueError("Có article_id trùng")

    if table["model_name"].isna().any() or table["model_name"].nunique() != 1:
        raise ValueError("Artifact phải dùng cùng một model_name")
    if table["data_version"].isna().any() or table["data_version"].nunique() != 1:
        raise ValueError("Artifact phải dùng cùng một data_version")

    try:
        vectors = np.stack([
            np.asarray(vector, dtype=np.float32)
            for vector in table["embedding"]
        ])
    except (ValueError, TypeError) as exc:
        raise ValueError("Các vector không hợp lệ hoặc không cùng số chiều") from exc

    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise ValueError("Vector không hợp lệ")
    if not np.isfinite(vectors).all():
        raise ValueError("Vector chứa NaN hoặc Infinity")

    if metadata is None:
        metadata = table.attrs.get("embedding_metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Metadata phải là dictionary")

    metadata_json = json.dumps(
        metadata, ensure_ascii=False, sort_keys=True
    )

    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Đường dẫn artifact phải kết thúc bằng .npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        article_ids=np.asarray(article_ids, dtype=np.str_),
        vectors=vectors,
        model_name=np.asarray([table["model_name"].iloc[0]]),
        data_version=np.asarray([table["data_version"].iloc[0]]),
        embedding_metadata_json=np.asarray([metadata_json], dtype=np.str_),
    )


def load_embedding_artifact(path: str | Path) -> pd.DataFrame:
    """Đọc artifact và đưa metadata vào thuộc tính của DataFrame."""
    with np.load(path, allow_pickle=False) as saved:
        article_ids = saved["article_ids"].copy()
        vectors = saved["vectors"].copy()
        model_name = str(saved["model_name"][0])
        data_version = str(saved["data_version"][0])

        # Cho phép đọc artifact cũ để bổ sung metadata.
        metadata = {}
        if "embedding_metadata_json" in saved.files:
            metadata = json.loads(
                str(saved["embedding_metadata_json"][0])
            )

    if not isinstance(metadata, dict):
        raise ValueError("Metadata artifact không hợp lệ")
    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise ValueError("Artifact không chứa ma trận vector hợp lệ")
    if len(article_ids) != len(vectors):
        raise ValueError("Số article_id không khớp số vector")
    if len(set(article_ids.tolist())) != len(article_ids):
        raise ValueError("Artifact có article_id trùng")
    if not np.isfinite(vectors).all():
        raise ValueError("Artifact chứa NaN hoặc Infinity")

    table = pd.DataFrame({
        "article_id": article_ids.tolist(),
        "embedding": list(vectors),
        "model_name": model_name,
        "data_version": data_version,
    })
    table.attrs["embedding_metadata"] = metadata
    return table


def build_embedding_metadata(config: dict, run_name: str) -> dict:
    """Mô tả cấu hình đã dùng để tạo một artifact embedding."""
    run = config["runs"][run_name]
    return {
        "embedding_run": run_name,
        "dataset": dict(config["dataset"]),
        "encoding": dict(config["encoding"]),
        "model_name": run["model_name"],
        "content_words": run["content_words"],
        "expected_dimensions": run["expected_dimensions"],
    }