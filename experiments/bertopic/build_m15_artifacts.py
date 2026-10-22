"""Tạo artifact Topic Representation cho BERTopic task M15."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


RUN_ID = "e1_vietnamese_sbert"

RESULT_DIR = (
    Path("results")
    / "bertopic_experiments"
    / RUN_ID
)

MODEL_DIR = (
    Path("artifacts")
    / "bertopic"
    / "experiments"
    / RUN_ID
    / "model"
)

OUTPUT_DIR = (
    Path("artifacts")
    / "bertopic"
    / "best_run"
)

KEYWORDS_PER_TOPIC = 10
REPRESENTATIVES_PER_TOPIC = 5

LABEL_REVIEW_PATH = (
    Path("configs")
    / "bertopic_topic_labels.yaml"
)


def write_json(
    path: Path,
    data: Any,
) -> None:
    """Ghi dữ liệu JSON với tiếng Việt được giữ nguyên."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def build_topic_keywords(
    keyword_table: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    """Chuyển bảng keyword thành dictionary theo topic."""
    result: dict[str, list[dict[str, Any]]] = {}

    valid_keywords = keyword_table.loc[
        keyword_table["topic"] != -1
    ].copy()

    valid_keywords = valid_keywords.sort_values(
        ["topic", "rank"]
    )

    for topic_id, group in valid_keywords.groupby("topic"):
        top_words = group.head(KEYWORDS_PER_TOPIC)

        result[str(int(topic_id))] = [
            {
                "word": str(row.word),
                "weight": float(row.ctfidf_score),
                "rank": int(row.rank),
            }
            for row in top_words.itertuples()
        ]

    return result


def build_representative_documents(
    document_topics: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    """Chọn các bài có xác suất cao nhất của mỗi topic."""
    valid_documents = document_topics.loc[
        ~document_topics["is_outlier"].astype(bool)
    ].copy()

    valid_documents["topic_probability"] = (
        pd.to_numeric(
            valid_documents["topic_probability"],
            errors="coerce",
        )
        .fillna(0.0)
    )

    valid_documents = valid_documents.sort_values(
        ["topic", "topic_probability", "article_id"],
        ascending=[True, False, True],
        kind="stable",
    )

    result: dict[str, list[dict[str, Any]]] = {}

    for topic_id, group in valid_documents.groupby("topic"):
        representatives = group.head(
            REPRESENTATIVES_PER_TOPIC
        )

        topic_documents: list[dict[str, Any]] = []

        for row in representatives.itertuples():
            document = {
                "article_id": str(row.article_id),
                "title": str(getattr(row, "title", "")),
                "split": str(row.split),
                "topic_probability": float(
                    row.topic_probability
                ),
            }

            if hasattr(row, "category_gold"):
                document["category_gold"] = str(
                    row.category_gold
                )

            topic_documents.append(document)

        result[str(int(topic_id))] = topic_documents

    return result


def make_suggested_label(
    keyword_items: list[dict[str, Any]],
) -> str:
    """Tạo nhãn tạm từ ba keyword đầu tiên."""
    words = [
        item["word"].replace("_", " ").strip()
        for item in keyword_items[:3]
    ]

    return " – ".join(words)


def build_display_mapping(
    topic_keywords: dict[str, list[dict[str, Any]]],
    label_review: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Tạo mapping nhãn hiển thị mà không đổi raw topic ID."""
    label_review = label_review or {}
    reviewed_labels = {
        str(topic_id): str(label).strip()
        for topic_id, label in (
            label_review.get("labels", {}) or {}
        ).items()
    }
    review_status = str(
        label_review.get("review_status", "provisional")
    )
    reviewed_by = [
        str(reviewer)
        for reviewer in (
            label_review.get("reviewed_by", []) or []
        )
    ]
    review_note = str(
        label_review.get("review_note", "")
    ).strip()

    mapping: dict[str, dict[str, Any]] = {
        "-1": {
            "display_label": "Outlier / Chưa xác định",
            "suggested_label": "Outlier / Chưa xác định",
            "top_keywords": [],
            "status": "system",
            "reviewed_by": [],
        }
    }

    for topic_id, keyword_items in topic_keywords.items():
        suggested_label = make_suggested_label(
            keyword_items
        )
        final_label = reviewed_labels.get(topic_id, "")

        mapping[topic_id] = {
            "display_label": final_label or suggested_label,
            "suggested_label": suggested_label,
            "final_label": final_label,
            "top_keywords": [
                item["word"]
                for item in keyword_items[:5]
            ],
            "status": (
                review_status
                if final_label
                else "provisional"
            ),
            "reviewed_by": (
                reviewed_by
                if final_label
                else []
            ),
            "review_note": (
                review_note
                if final_label
                else ""
            ),
        }

    return mapping


def main() -> None:
    """Tạo toàn bộ artifact cho task M15."""
    keyword_path = RESULT_DIR / "topic_keywords.csv"
    document_path = RESULT_DIR / "document_topics.csv"
    config_path = RESULT_DIR / "config_used.yaml"
    metrics_path = RESULT_DIR / "metrics.json"

    required_paths = [
        keyword_path,
        document_path,
        config_path,
        metrics_path,
        MODEL_DIR,
    ]

    missing_paths = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Thiếu input M14:\n"
            + "\n".join(missing_paths)
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    keyword_table = pd.read_csv(keyword_path)
    document_topics = pd.read_csv(document_path)

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    with metrics_path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        metrics = json.load(file)

    with LABEL_REVIEW_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        label_review = yaml.safe_load(file) or {}

    if label_review.get("run_id") != RUN_ID:
        raise ValueError(
            "Run ID trong mapping nhãn không khớp "
            f"{RUN_ID}: {label_review.get('run_id')}"
        )

    topic_keywords = build_topic_keywords(
        keyword_table
    )

    representative_documents = (
        build_representative_documents(
            document_topics
        )
    )

    display_mapping = build_display_mapping(
        topic_keywords,
        label_review,
    )

    expected_topics = set(topic_keywords)
    reviewed_topics = {
        str(topic_id)
        for topic_id in (
            label_review.get("labels", {}) or {}
        )
    }
    if reviewed_topics != expected_topics:
        missing = sorted(
            expected_topics - reviewed_topics,
            key=int,
        )
        unexpected = sorted(
            reviewed_topics - expected_topics,
            key=int,
        )
        raise ValueError(
            "Mapping nhãn phải bao phủ đúng toàn bộ topic. "
            f"Thiếu={missing}; dư={unexpected}"
        )

    broad_topics = {
        str(topic_id)
        for topic_id in (
            label_review.get("broad_topics", []) or []
        )
    }
    merge_candidates = {
        str(topic_id)
        for candidate in (
            label_review.get("merge_candidates", []) or []
        )
        for topic_id in candidate.get("topics", [])
    }

    write_json(
        OUTPUT_DIR / "topic_keywords.json",
        topic_keywords,
    )

    write_json(
        OUTPUT_DIR / "representative_docs.json",
        representative_documents,
    )

    write_json(
        OUTPUT_DIR / "topic_display_mapping.json",
        display_mapping,
    )

    write_json(
        OUTPUT_DIR / "config.json",
        config,
    )

    review_rows: list[dict[str, Any]] = []

    for topic_id, mapping_item in display_mapping.items():
        if topic_id == "-1":
            continue

        keyword_text = ", ".join(
            mapping_item["top_keywords"]
        )

        representative_titles = " | ".join(
            document["title"]
            for document in representative_documents.get(
                topic_id,
                [],
            )
        )

        review_rows.append(
            {
                "topic_id": int(topic_id),
                "suggested_label": mapping_item[
                    "suggested_label"
                ],
                "final_label": mapping_item.get(
                    "final_label",
                    "",
                ),
                "top_keywords": keyword_text,
                "representative_titles": (
                    representative_titles
                ),
                "keyword_quality": "",
                "representative_quality": "",
                "is_broad": topic_id in broad_topics,
                "is_duplicate": (
                    topic_id in merge_candidates
                ),
                "review_note": mapping_item.get(
                    "review_note",
                    "",
                ),
                "review_status": mapping_item[
                    "status"
                ],
            }
        )

    review_table = pd.DataFrame(review_rows).sort_values(
        "topic_id"
    )

    review_output_dir = (
        Path("results")
        / "bertopic_m15"
    )

    review_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    review_table.to_csv(
        review_output_dir / "topic_review.csv",
        index=False,
        encoding="utf-8-sig",
    )


    metadata = {
        "task_id": "M15",
        "run_id": RUN_ID,
        "source_result_dir": str(RESULT_DIR),
        "model_path": str(MODEL_DIR),
        "dataset_version": config["dataset"]["version"],
        "fit_split": config["dataset"]["fit_split"],
        "validation_split": config["dataset"][
            "validation_split"
        ],
        "topic_count": int(metrics["topic_count"]),
        "train_outlier_rate": float(
            metrics["train_outlier_rate"]
        ),
        "validation_outlier_rate": float(
            metrics["validation_outlier_rate"]
        ),
        "topic_diversity": float(
            metrics["topic_diversity"]
        ),
        "keywords_per_topic": KEYWORDS_PER_TOPIC,
        "representative_docs_per_topic": (
            REPRESENTATIVES_PER_TOPIC
        ),
        "label_status": label_review.get(
            "review_status",
            "provisional",
        ),
        "label_review_path": str(LABEL_REVIEW_PATH),
        "selection_reason": (
            "Run có đầy đủ artifact và tỷ lệ outlier "
            "validation thấp nhất trong các run M14."
        ),
    }



    write_json(
        OUTPUT_DIR / "run_metadata.json",
        metadata,
    )

    document_topics.to_parquet(
        OUTPUT_DIR / "document_topics.parquet",
        index=False,
    )

    print(f"Created M15 artifacts at: {OUTPUT_DIR}")
    print(f"Topic count: {len(topic_keywords)}")
    print(
        "Topics with representative documents: "
        f"{len(representative_documents)}"
    )


if __name__ == "__main__":
    main()
