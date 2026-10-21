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


def prepare_topic_diagnostics(
    probabilities: Any | None,
    expected_length: int,
    *,
    multi_topic_gap_threshold: float = 0.10,
) -> dict[str, np.ndarray]:
    """Trích top-1/top-2 và entropy khi BERTopic trả ma trận membership.

    BERTopic có thể chỉ trả một confidence cho mỗi bài. Trong trường hợp
    đó các trường chẩn đoán đa chủ đề được để thiếu thay vì suy diễn.
    """
    missing = np.full(expected_length, np.nan)
    result = {
        "second_topic_probability": missing.copy(),
        "top_two_share_gap": missing.copy(),
        "topic_entropy": missing.copy(),
        "is_multi_topic": np.full(expected_length, False, dtype=bool),
    }
    if probabilities is None:
        return result

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2:
        return result
    if len(values) != expected_length:
        raise ValueError("Số xác suất phải bằng số bài trong dataset.")
    if values.shape[1] < 2:
        return result

    sorted_values = np.sort(values, axis=1)
    top_one = sorted_values[:, -1]
    top_two = sorted_values[:, -2]
    gap = top_one - top_two
    safe_values = np.clip(values, 1e-12, 1.0)

    result["second_topic_probability"] = top_two
    result["top_two_share_gap"] = gap
    result["topic_entropy"] = -np.sum(
        safe_values * np.log(safe_values), axis=1
    )
    result["is_multi_topic"] = gap <= multi_topic_gap_threshold
    return result


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
    diagnostics = prepare_topic_diagnostics(
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
            **diagnostics,
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

def build_embedding_projection(
    article_ids: pd.Series,
    split: str,
    topics: list[int] | np.ndarray,
    embeddings_2d: Any,
) -> pd.DataFrame:
    """Tạo bảng tọa độ 2D dùng để trực quan hóa cụm topic.

    Mỗi dòng tương ứng với một bài báo:
    - x, y: tọa độ sau khi giảm chiều embedding;
    - topic: cụm được HDBSCAN gán;
    - is_outlier: bài không thuộc cụm rõ ràng.
    """
    coordinates = np.asarray(embeddings_2d)
    topic_values = np.asarray(topics)

    if coordinates.ndim != 2:
        raise ValueError(
            "embeddings_2d phải là ma trận hai chiều."
        )

    if coordinates.shape[1] != 2:
        raise ValueError(
            "embeddings_2d phải có đúng hai cột x và y."
        )

    expected_length = len(article_ids)

    if len(coordinates) != expected_length:
        raise ValueError(
            "Số tọa độ embedding không khớp số bài báo."
        )

    if len(topic_values) != expected_length:
        raise ValueError(
            "Số topic không khớp số bài báo."
        )

    projection = pd.DataFrame(
        {
            "article_id": article_ids.astype(str).to_numpy(),
            "split": split,
            "topic": topic_values.astype(int),
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
        }
    )

    projection["is_outlier"] = (
        projection["topic"] == -1
    )

    return projection
