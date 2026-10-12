"""Tạo các bảng dữ liệu dùng để giải thích kết quả BERTopic."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def prepare_topic_probabilities(
    probabilities: Any | None,
    expected_length: int,
) -> np.ndarray:
    """Chuẩn hóa xác suất gán topic thành một giá trị cho mỗi bài."""
    if probabilities is None:
        return np.full(expected_length, np.nan)

    probability_array = np.asarray(probabilities)

    if probability_array.ndim == 2:
        probability_array = probability_array.max(axis=1)

    probability_array = probability_array.reshape(-1)

    if len(probability_array) != expected_length:
        raise ValueError(
            "Số xác suất phải bằng số bài trong dataset."
        )

    return probability_array.astype(float)


def build_document_topics(
    dataset: pd.DataFrame,
    topics: Sequence[int],
    probabilities: Any | None,
    split: str,
) -> pd.DataFrame:
    """Tạo bảng cho biết từng bài được gán vào topic nào."""
    if "article_id" not in dataset.columns:
        raise ValueError(
            "Dataset phải có cột article_id."
        )

    if len(dataset) != len(topics):
        raise ValueError(
            "Số topic phải bằng số bài trong dataset."
        )

    topic_values = np.asarray(topics, dtype=int)
    probability_values = prepare_topic_probabilities(
        probabilities,
        len(dataset),
    )

    result = pd.DataFrame(
        {
            "article_id": dataset[
                "article_id"
            ].astype(str).to_numpy(),
            "split": split,
            "topic": topic_values,
            "topic_probability": probability_values,
            "is_outlier": topic_values == -1,
        }
    )

    for column in ("title", "category_gold"):
        if column in dataset.columns:
            result[column] = (
                dataset[column]
                .fillna("")
                .astype(str)
                .to_numpy()
            )

    return result


def build_topic_keywords(
    topic_model: Any,
) -> pd.DataFrame:
    """Tạo bảng từ khóa và trọng số c-TF-IDF của từng topic."""
    rows: list[dict[str, Any]] = []

    for topic_id, words in topic_model.get_topics().items():
        if int(topic_id) == -1:
            continue

        for rank, (word, score) in enumerate(
            words,
            start=1,
        ):
            rows.append(
                {
                    "topic": int(topic_id),
                    "rank": rank,
                    "word": str(word),
                    "ctfidf_score": float(score),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "topic",
            "rank",
            "word",
            "ctfidf_score",
        ],
    )


def build_topic_summary(
    document_topics: pd.DataFrame,
    topic_keywords: pd.DataFrame,
    representative_titles: int = 3,
) -> pd.DataFrame:
    """Tổng hợp kích thước, từ khóa và chuyên mục của từng topic."""
    assigned_documents = document_topics.loc[
        ~document_topics["is_outlier"]
    ].copy()

    rows: list[dict[str, Any]] = []
    total_documents = len(document_topics)

    for topic_id, group in assigned_documents.groupby(
        "topic",
        sort=True,
    ):
        topic_words = (
            topic_keywords.loc[
                topic_keywords["topic"] == topic_id
            ]
            .sort_values("rank")
            .head(5)["word"]
            .tolist()
        )

        row: dict[str, Any] = {
            "topic": int(topic_id),
            "topic_name": (
                f"{int(topic_id)}_"
                + "_".join(topic_words[:3])
            ),
            "document_count": len(group),
            "percentage": len(group) / total_documents,
            "top_keywords": ", ".join(topic_words),
        }

        if "title" in group.columns:
            titles = (
                group["title"]
                .loc[group["title"] != ""]
                .head(representative_titles)
                .tolist()
            )
            row["representative_titles"] = " | ".join(
                titles
            )

        if "category_gold" in group.columns:
            categories = group["category_gold"].loc[
                group["category_gold"] != ""
            ]

            if not categories.empty:
                category_counts = categories.value_counts()
                row["dominant_category"] = (
                    category_counts.index[0]
                )
                row["category_purity"] = (
                    category_counts.iloc[0]
                    / category_counts.sum()
                )
            else:
                row["dominant_category"] = ""
                row["category_purity"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        "document_count",
        ascending=False,
        ignore_index=True,
    )