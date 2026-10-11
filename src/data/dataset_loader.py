"""Đọc và kiểm tra dữ liệu đầu vào cho BERTopic."""

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "article_id",
    "title",
    "content",
    "published_at",
    "source",
    "url",
}


def load_dataset(dataset_path: str | Path) -> pd.DataFrame:
    """Đọc dataset và kiểm tra các điều kiện đầu vào."""
    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {path}")

    dataset = pd.read_csv(path)
    validate_dataset(dataset)

    # Giữ nguyên thứ tự dòng để khớp article_id với kết quả topic.
    return dataset


def validate_dataset(dataset: pd.DataFrame) -> None:
    """Kiểm tra schema, article_id và bài viết rỗng."""
    missing_columns = REQUIRED_COLUMNS - set(dataset.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset thiếu các cột: {sorted(missing_columns)}"
        )

    if dataset.empty:
        raise ValueError("Dataset không có dữ liệu")

    if dataset["article_id"].isna().any():
        raise ValueError("Dataset có article_id bị thiếu")

    if dataset["article_id"].duplicated().any():
        duplicate_count = int(dataset["article_id"].duplicated().sum())
        raise ValueError(
            f"Dataset có {duplicate_count} article_id bị trùng"
        )

    # fillna tránh giá trị thiếu bị chuyển thành chuỗi "nan".
    title = dataset["title"].fillna("").astype(str).str.strip()
    content = dataset["content"].fillna("").astype(str).str.strip()

    empty_articles = title.eq("") & content.eq("")

    if empty_articles.any():
        raise ValueError(
            f"Dataset có {int(empty_articles.sum())} bài bị rỗng"
        )

def load_split_dataset(
    dataset_path: str | Path,
    split_manifest_path: str | Path,
    split_name: str,
) -> pd.DataFrame:
    """Đọc một split và ánh xạ về Dataset v1."""
    dataset = load_dataset(dataset_path)
    split_path = Path(split_manifest_path)

    if not split_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy split manifest: {split_path}"
        )

    split_manifest = pd.read_csv(
        split_path,
        encoding="utf-8-sig",
    )

    required_columns = {"article_id", "split"}
    missing_columns = required_columns - set(split_manifest.columns)

    if missing_columns:
        raise ValueError(
            f"Split manifest thiếu các cột: {sorted(missing_columns)}"
        )

    valid_splits = {"train", "val", "test"}

    if split_name not in valid_splits:
        raise ValueError(
            f"Split không hợp lệ: {split_name}"
        )

    selected_ids = split_manifest.loc[
        split_manifest["split"] == split_name,
        ["article_id", "split"],
    ]

    split_dataset = selected_ids.merge(
        dataset,
        on="article_id",
        how="left",
        validate="one_to_one",
    )

    # Mọi ID trong split phải tồn tại trong Dataset v1.
    if split_dataset["title"].isna().any():
        raise ValueError(
            f"Split {split_name} có article_id không tồn tại"
        )

    return split_dataset