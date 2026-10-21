"""Kiểm tra và tổng hợp phiếu chấm HDBSCAN của R1, R2."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/hdbscan/human_eval"
SUMMARY = BASE / "summary"

TOPIC_SCORES = [
    "consistency_1_5",
    "specificity_1_5",
    "naming_1_5",
    "usefulness_1_5",
    "boundary_1_5",
]
DOCUMENT_SCORE = "document_relevance_1_5"


def read_csv(name):
    return pd.read_csv(
        BASE / name,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )


def validate_form(
    form,
    reference,
    key,
    fixed_columns,
    score_columns,
    reviewer,
):
    if not form["reviewer_id"].eq(reviewer).all():
        raise ValueError(
            f"Phiếu {reviewer} có reviewer_id sai"
        )
    if form[key].duplicated().any():
        raise ValueError(
            f"Phiếu {reviewer} có {key} trùng"
        )
    if set(form[key]) != set(reference[key]):
        raise ValueError(
            f"Phiếu {reviewer} thiếu hoặc thừa {key}"
        )

    actual = form.set_index(key).sort_index()
    expected = reference.set_index(key).sort_index()

    for column in fixed_columns:
        if not actual[column].equals(expected[column]):
            raise ValueError(
                f"Phiếu {reviewer} đã thay đổi cột "
                f"{column}"
            )

    for column in score_columns:
        values = pd.to_numeric(
            form[column].str.strip(),
            errors="coerce",
        )
        if (
            values.isna().any()
            or not values.between(1, 5).all()
            or not values.mod(1).eq(0).all()
        ):
            raise ValueError(
                f"{reviewer}: {column} phải là "
                "số nguyên từ 1 đến 5"
            )
        form[column] = values.astype(int)

    return form


def main():
    admin = read_csv("evaluation_samples.csv")
    mapping = read_csv("topic_mapping.csv")

    if admin["sample_id"].duplicated().any():
        raise ValueError("sample_id quản lý bị trùng")
    if mapping["blind_topic_id"].duplicated().any():
        raise ValueError("blind_topic_id mapping bị trùng")

    topic_reference = (
        admin[["blind_topic_id", "keywords"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    document_reference = admin[
        [
            "sample_id",
            "blind_topic_id",
            "article_id",
            "keywords",
            "title",
            "content",
            "url",
        ]
    ].copy()

    topic_forms = {}
    document_forms = {}

    for reviewer in ("R1", "R2"):
        topic_forms[reviewer] = validate_form(
            read_csv(
                f"reviewer_{reviewer}_topics.csv"
            ),
            topic_reference,
            key="blind_topic_id",
            fixed_columns=["keywords"],
            score_columns=TOPIC_SCORES,
            reviewer=reviewer,
        )

        document_forms[reviewer] = validate_form(
            read_csv(
                f"reviewer_{reviewer}_documents.csv"
            ),
            document_reference,
            key="sample_id",
            fixed_columns=[
                "blind_topic_id",
                "article_id",
                "keywords",
                "title",
                "content",
                "url",
            ],
            score_columns=[DOCUMENT_SCORE],
            reviewer=reviewer,
        )

    topic_pairs = topic_forms["R1"][
        ["blind_topic_id"] + TOPIC_SCORES
    ].merge(
        topic_forms["R2"][
            ["blind_topic_id"] + TOPIC_SCORES
        ],
        on="blind_topic_id",
        suffixes=("_R1", "_R2"),
        validate="one_to_one",
    )
    topic_pairs = topic_pairs.merge(
        mapping[
            ["blind_topic_id", "cluster_id"]
        ],
        on="blind_topic_id",
        validate="one_to_one",
    )

    document_pairs = document_forms["R1"][
        [
            "sample_id",
            "blind_topic_id",
            "article_id",
            DOCUMENT_SCORE,
        ]
    ].merge(
        document_forms["R2"][
            ["sample_id", DOCUMENT_SCORE]
        ],
        on="sample_id",
        suffixes=("_R1", "_R2"),
        validate="one_to_one",
    )

    agreement_rows = []
    disagreement_rows = []

    for level, pairs, key, criteria in (
        (
            "topic",
            topic_pairs,
            "blind_topic_id",
            TOPIC_SCORES,
        ),
        (
            "document",
            document_pairs,
            "sample_id",
            [DOCUMENT_SCORE],
        ),
    ):
        for criterion in criteria:
            r1 = pairs[f"{criterion}_R1"]
            r2 = pairs[f"{criterion}_R2"]
            difference = (r1 - r2).abs()

            agreement_rows.append({
                "level": level,
                "criterion": criterion,
                "n_items": len(pairs),
                "mean_R1": r1.mean(),
                "mean_R2": r2.mean(),
                "exact_agreement_percent": (
                    100 * difference.eq(0).mean()
                ),
                "mean_absolute_difference": (
                    difference.mean()
                ),
            })

            for index in pairs.index[
                difference.ge(2)
            ]:
                disagreement_rows.append({
                    "level": level,
                    "item_id": pairs.loc[index, key],
                    "criterion": criterion,
                    "score_R1": int(r1.loc[index]),
                    "score_R2": int(r2.loc[index]),
                    "absolute_difference": int(
                        difference.loc[index]
                    ),
                })

    # Mỗi cụm có trọng số bằng nhau trong điểm tổng hợp.
    method_rows = []
    for criterion in TOPIC_SCORES:
        r1 = topic_pairs[f"{criterion}_R1"]
        r2 = topic_pairs[f"{criterion}_R2"]
        method_rows.append({
            "criterion": criterion,
            "n_topics": len(topic_pairs),
            "mean_score": ((r1 + r2) / 2).mean(),
        })

    document_pairs["mean_relevance"] = (
        document_pairs[
            f"{DOCUMENT_SCORE}_R1"
        ]
        + document_pairs[
            f"{DOCUMENT_SCORE}_R2"
        ]
    ) / 2
    relevance_by_topic = (
        document_pairs.groupby(
            "blind_topic_id"
        )["mean_relevance"]
        .mean()
    )
    method_rows.append({
        "criterion": DOCUMENT_SCORE,
        "n_topics": len(relevance_by_topic),
        "mean_score": relevance_by_topic.mean(),
    })

    outputs = {
        "topic_scores.csv": topic_pairs,
        "document_scores.csv": document_pairs,
        "agreement_summary.csv": pd.DataFrame(
            agreement_rows
        ),
        "disagreements.csv": pd.DataFrame(
            disagreement_rows,
            columns=[
                "level",
                "item_id",
                "criterion",
                "score_R1",
                "score_R2",
                "absolute_difference",
            ],
        ),
        "method_summary.csv": pd.DataFrame(
            method_rows
        ),
    }

    existing = [
        name
        for name in outputs
        if (SUMMARY / name).exists()
    ]
    if existing:
        raise FileExistsError(
            "Không ghi đè kết quả đã tổng hợp: "
            + ", ".join(existing)
        )

    SUMMARY.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(
            SUMMARY / name,
            index=False,
            encoding="utf-8-sig",
        )

    print("Số cụm đã chấm:", len(topic_pairs))
    print(
        "Số cặp bài–cụm đã chấm:",
        len(document_pairs),
    )
    print(
        "Số trường hợp lệch ít nhất 2 điểm:",
        len(disagreement_rows),
    )
    print("Đã lưu trong:", SUMMARY)


if __name__ == "__main__":
    main()