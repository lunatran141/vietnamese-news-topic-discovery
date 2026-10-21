"""Tạo phiếu human evaluation cho outlier HDBSCAN đại diện."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RUN_ID = "minilm_title_content_eom_15_5"

DOCUMENT_TOPICS_PATH = (
    ROOT
    / "artifacts/common/hdbscan"
    / RUN_ID
    / "document_topics.csv"
)
ARTICLES_PATH = (
    ROOT / "data/processed/p0_minimal.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results/hdbscan/human_eval/normalized"
)
OUTPUT_PATH = (
    OUTPUT_DIR / "outlier_review_template.csv"
)

REVIEWERS = ("R1", "R2")
SAMPLES_PER_SPLIT = 5
RANDOM_STATE = 42


def load_inputs() -> pd.DataFrame:
    documents = pd.read_csv(
        DOCUMENT_TOPICS_PATH,
        encoding="utf-8-sig",
        dtype={"article_id": str},
    )

    required_document_columns = {
        "article_id",
        "split",
        "topic_id",
        "membership_strength",
        "soft_primary_topic_id",
        "soft_primary_membership_strength",
        "second_topic_id",
        "second_membership_strength",
        "top_two_share_gap",
        "is_multi_topic",
        "title",
    }
    missing = (
        required_document_columns
        - set(documents.columns)
    )
    if missing:
        raise ValueError(
            "document_topics thiếu cột: "
            f"{sorted(missing)}"
        )

    articles = pd.read_csv(
        ARTICLES_PATH,
        usecols=[
            "article_id",
            "content",
            "source",
            "published_at",
            "url",
        ],
        dtype={"article_id": str},
    )

    if articles["article_id"].duplicated().any():
        raise ValueError(
            "Dữ liệu bài báo có article_id trùng"
        )

    outliers = documents.loc[
        documents["topic_id"].eq(-1)
    ].copy()

    if outliers.empty:
        raise ValueError(
            "Run đại diện không có outlier"
        )

    outliers = outliers.merge(
        articles,
        on="article_id",
        how="left",
        validate="one_to_one",
    )

    required_values = [
        "title",
        "content",
        "source",
        "published_at",
        "url",
    ]
    if outliers[required_values].isna().any().any():
        raise ValueError(
            "Có outlier thiếu nội dung hoặc metadata"
        )

    return outliers


def sample_outliers(
    outliers: pd.DataFrame,
) -> pd.DataFrame:
    samples = []

    for index, split in enumerate(
        ("train", "val", "test")
    ):
        candidates = outliers.loc[
            outliers["split"].eq(split)
        ].copy()

        if len(candidates) < SAMPLES_PER_SPLIT:
            raise ValueError(
                f"Split {split} không đủ outlier để lấy mẫu"
            )

        selected = candidates.sample(
            n=SAMPLES_PER_SPLIT,
            random_state=RANDOM_STATE + index,
        ).sort_values(
            "article_id",
            kind="stable",
        )

        selected = selected.reset_index(drop=True)
        selected["sample_id"] = [
            (
                f"OUT-{split.upper()}-"
                f"{position:02d}"
            )
            for position in range(
                1,
                len(selected) + 1,
            )
        ]

        samples.append(selected)

    result = pd.concat(
        samples,
        ignore_index=True,
    )

    if len(result) != (
        3 * SAMPLES_PER_SPLIT
    ):
        raise ValueError(
            "Sai số lượng outlier được lấy mẫu"
        )
    if result["article_id"].duplicated().any():
        raise ValueError(
            "Mẫu outlier có article_id trùng"
        )

    return result


def build_review_template(
    samples: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for reviewer in REVIEWERS:
        for row in samples.itertuples(index=False):
            rows.append(
                {
                    "run_id": RUN_ID,
                    "reviewer": reviewer,
                    "review_status": "pending",
                    "sample_id": row.sample_id,
                    "article_id": row.article_id,
                    "split": row.split,
                    "topic_id": int(row.topic_id),
                    "title": row.title,
                    "content": row.content,
                    "source": row.source,
                    "published_at": row.published_at,
                    "url": row.url,
                    "membership_strength": (
                        float(row.membership_strength)
                    ),
                    "soft_primary_topic_id": int(
                        row.soft_primary_topic_id
                    ),
                    "soft_primary_membership_strength": (
                        float(
                            row.soft_primary_membership_strength
                        )
                    ),
                    "second_topic_id": int(
                        row.second_topic_id
                    ),
                    "second_membership_strength": (
                        float(
                            row.second_membership_strength
                        )
                    ),
                    "top_two_share_gap": float(
                        row.top_two_share_gap
                    ),
                    "model_is_multi_topic": bool(
                        row.is_multi_topic
                    ),

                    # Reviewer phải điền.
                    "outlier_correct": pd.NA,
                    "is_boundary_or_mixed": pd.NA,
                    "notes": "",
                }
            )

    template = pd.DataFrame(rows)

    expected_rows = (
        len(samples) * len(REVIEWERS)
    )
    if len(template) != expected_rows:
        raise ValueError(
            "Sai số dòng trong phiếu review"
        )

    if template.duplicated(
        ["reviewer", "sample_id"]
    ).any():
        raise ValueError(
            "Reviewer có sample_id trùng"
        )

    return template


def main() -> None:
    outliers = load_inputs()
    samples = sample_outliers(outliers)
    template = build_review_template(samples)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    template.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Đã lưu: {OUTPUT_PATH}")
    print(
        "Số bài duy nhất:",
        template["article_id"].nunique(),
    )
    print(
        "Số dòng review:",
        len(template),
    )
    print(
        "Theo reviewer:",
        template["reviewer"]
        .value_counts()
        .to_dict(),
    )
    print(
        "Theo split:",
        template.drop_duplicates(
            "sample_id"
        )["split"].value_counts().to_dict(),
    )


if __name__ == "__main__":
    main()