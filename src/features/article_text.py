"""Tạo các biến thể văn bản dùng cho mô hình."""

import pandas as pd


SUPPORTED_TEXT_VARIANTS = {
    "title",
    "content",
    "title_content",
    "processed_title_content",
    "text",
}


def build_article_text(
    dataset: pd.DataFrame,
    text_variant: str,
) -> pd.Series:
    """Tạo text đầu vào và giữ nguyên thứ tự bài viết."""
    if text_variant not in SUPPORTED_TEXT_VARIANTS:
        raise ValueError(
            f"Text variant không hợp lệ: {text_variant}"
        )

    title = dataset["title"].fillna("").astype(str).str.strip()
    content = dataset["content"].fillna("").astype(str).str.strip()

    if text_variant == "title":
        text = title
    elif text_variant == "content":
        text = content
    elif text_variant == "title_content":
        text = (title + ". " + content).str.strip(" .")
    elif text_variant == "text":
        if "text" not in dataset.columns:
            raise ValueError("Dataset không có cột text")

        text = dataset["text"].fillna("").astype(str).str.strip()
    else:
        required_columns = {
            "processed_title",
            "processed_content",
        }

        if not required_columns.issubset(dataset.columns):
            raise ValueError(
                "Dataset thiếu processed_title hoặc processed_content"
            )

        processed_title = (
            dataset["processed_title"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        processed_content = (
            dataset["processed_content"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        text = (
            processed_title + ". " + processed_content
        ).str.strip(" .")

    if text.eq("").any():
        raise ValueError("Text đầu vào có bài bị rỗng")

    # Không reset index vì embedding phải khớp với article_id.
    return text