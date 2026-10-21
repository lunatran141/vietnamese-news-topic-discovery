"""Tạo mẫu và phiếu chấm HDBSCAN cho hai reviewer."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

ASSIGNMENTS_PATH = (
    ROOT
    / "results/hdbscan/validation/"
    "minilm_title_content_eom_15_5_val_assignments.csv"
)
TOPICS_PATH = (
    ROOT
    / "artifacts/common/hdbscan/minilm_title_content_eom_15_5/"
    "topic_summary.csv"
)
DATASET_PATH = ROOT / "data/processed/p0_minimal.csv"
SPLIT_PATH = ROOT / "data/splits/split_manifest.csv"
OUTPUT_DIR = ROOT / "results/hdbscan/human_eval"

SEED = 42
N_CLUSTERS = 5
N_TOP = 3
N_RANDOM = 2

TOPIC_SCORE_COLUMNS = [
    "consistency_1_5",
    "specificity_1_5",
    "naming_1_5",
    "usefulness_1_5",
    "boundary_1_5",
]


def load_inputs():
    assignments = pd.read_csv(
        ASSIGNMENTS_PATH,
        dtype={"article_id": str},
        encoding="utf-8-sig",
    )
    topics = pd.read_csv(TOPICS_PATH, encoding="utf-8-sig")
    dataset = pd.read_csv(
        DATASET_PATH,
        usecols=["article_id", "title", "content", "url"],
        dtype={"article_id": str},
        encoding="utf-8-sig",
    )
    splits = pd.read_csv(
        SPLIT_PATH,
        dtype={"article_id": str},
        encoding="utf-8-sig",
    )

    if assignments["article_id"].duplicated().any():
        raise ValueError("Assignments có article_id trùng")
    if dataset["article_id"].duplicated().any():
        raise ValueError("Dataset có article_id trùng")
    if topics["topic_id"].duplicated().any():
        raise ValueError("Topic summary có topic_id trùng")

    if not assignments["split"].eq("val").all():
        raise ValueError("Assignments phải chỉ chứa split val")
    if not assignments["embedding_run"].eq(
        "minilm_title_content"
    ).all():
        raise ValueError("Sai embedding_run")
    if not assignments["experiment"].eq("eom_15_5").all():
        raise ValueError("Sai experiment")

    validation_ids = set(
        splits.loc[splits["split"].eq("val"), "article_id"]
    )
    if len(validation_ids) != 1391:
        raise ValueError(
            f"Manifest có {len(validation_ids)} ID validation"
        )
    if set(assignments["article_id"]) != validation_ids:
        raise ValueError(
            "Article ID trong assignments không khớp validation"
        )

    cluster_ids = pd.to_numeric(
        assignments["cluster_id"], errors="raise"
    )
    if (
        cluster_ids.isna().any()
        or not np.isfinite(cluster_ids).all()
        or not np.equal(cluster_ids, np.floor(cluster_ids)).all()
    ):
        raise ValueError("cluster_id không hợp lệ")
    assignments["cluster_id"] = cluster_ids.astype(int)

    assignments["membership_strength"] = pd.to_numeric(
        assignments["membership_strength"], errors="raise"
    )
    if (
        assignments["membership_strength"].isna().any()
        or not assignments["membership_strength"]
        .between(0, 1)
        .all()
    ):
        raise ValueError("membership_strength không hợp lệ")

    return assignments, topics, dataset


def build_samples(assignments, topics, dataset):
    clustered = assignments.loc[
        assignments["cluster_id"].ne(-1)
    ].copy()

    sizes = (
        clustered.groupby("cluster_id")
        .size()
        .rename("n_val")
        .reset_index()
    )
    eligible = (
        sizes.loc[sizes["n_val"].ge(N_TOP + N_RANDOM)]
        .sort_values(["n_val", "cluster_id"])
        .reset_index(drop=True)
    )
    if len(eligible) < N_CLUSTERS:
        raise ValueError(
            "Không đủ cụm có ít nhất 5 bài validation"
        )

    # Chọn 5 vị trí trải từ cụm nhỏ đến cụm lớn trong các
    # cụm đủ điều kiện. Không chọn theo điểm đánh giá.
    positions = np.linspace(
        0, len(eligible) - 1, N_CLUSTERS
    ).round().astype(int)
    selected_clusters = eligible.iloc[positions]

    topic_lookup = topics.set_index("topic_id")
    rng = np.random.default_rng(SEED)

    sample_groups = []
    mapping_rows = []

    for number, row in enumerate(
        selected_clusters.itertuples(index=False), start=1
    ):
        cluster_id = int(row.cluster_id)
        blind_topic_id = f"T{number:02d}"

        if cluster_id not in topic_lookup.index:
            raise ValueError(
                f"Thiếu topic keywords cho cụm {cluster_id}"
            )

        keywords = topic_lookup.loc[
            cluster_id, "top_keywords"
        ]
        if pd.isna(keywords) or not str(keywords).strip():
            raise ValueError(
                f"Cụm {cluster_id} không có keywords"
            )

        candidates = clustered.loc[
            clustered["cluster_id"].eq(cluster_id)
        ].copy()

        top = (
            candidates.sort_values(
                ["membership_strength", "article_id"],
                ascending=[False, True],
            )
            .head(N_TOP)
            .copy()
        )
        top["sample_type"] = "top_membership"

        remaining = (
            candidates.loc[
                ~candidates["article_id"].isin(top["article_id"])
            ]
            .sort_values("article_id")
            .reset_index(drop=True)
        )
        random_positions = rng.choice(
            len(remaining),
            size=N_RANDOM,
            replace=False,
        )
        random_docs = remaining.iloc[
            random_positions
        ].copy()
        random_docs["sample_type"] = "random"

        group = pd.concat(
            [top, random_docs],
            ignore_index=True,
        )
        group["blind_topic_id"] = blind_topic_id
        group["keywords"] = str(keywords)

        # Xáo thứ tự bài hiển thị; hai reviewer vẫn nhận
        # chính xác cùng một thứ tự.
        group = group.iloc[
            rng.permutation(len(group))
        ].reset_index(drop=True)
        sample_groups.append(group)

        mapping_rows.append({
            "blind_topic_id": blind_topic_id,
            "embedding_run": "minilm_title_content",
            "experiment": "eom_15_5",
            "split": "val",
            "cluster_id": cluster_id,
            "n_val_in_cluster": int(row.n_val),
            "keywords": str(keywords),
        })

    samples = pd.concat(
        sample_groups, ignore_index=True
    )
    samples["sample_id"] = [
        f"HS{i:03d}"
        for i in range(1, len(samples) + 1)
    ]

    samples = samples.merge(
        dataset,
        on="article_id",
        how="left",
        validate="many_to_one",
    )

    if samples["article_id"].duplicated().any():
        raise ValueError("Bộ mẫu có article_id trùng")
    if samples[["title", "content"]].isna().any().any():
        raise ValueError("Bộ mẫu thiếu title hoặc content")
    if samples["title"].astype(str).str.strip().eq("").any():
        raise ValueError("Bộ mẫu có title rỗng")

    return samples, pd.DataFrame(mapping_rows), len(eligible)


def export_files(samples, mapping):
    # Hai file quản lý, không gửi cho reviewer.
    admin_samples = samples[
        [
            "sample_id",
            "blind_topic_id",
            "article_id",
            "split",
            "cluster_id",
            "sample_type",
            "membership_strength",
            "keywords",
            "title",
            "content",
            "url",
        ]
    ].copy()

    topic_form = (
        samples[["blind_topic_id", "keywords"]]
        .drop_duplicates()
        .sort_values("blind_topic_id")
        .reset_index(drop=True)
    )
    for column in TOPIC_SCORE_COLUMNS:
        topic_form[column] = ""
    topic_form["proposed_topic_label"] = ""
    topic_form["good_article_id"] = ""
    topic_form["difficult_article_id"] = ""
    topic_form["review_reason"] = ""

    document_form = samples[
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
    document_form["document_relevance_1_5"] = ""
    document_form["error_tag"] = ""
    document_form["review_reason"] = ""

    files = {
        "evaluation_samples.csv": admin_samples,
        "topic_mapping.csv": mapping,
    }

    for reviewer in ("R1", "R2"):
        reviewer_topics = topic_form.copy()
        reviewer_topics.insert(
            0, "reviewer_id", reviewer
        )

        reviewer_documents = document_form.copy()
        reviewer_documents.insert(
            0, "reviewer_id", reviewer
        )

        files[
            f"reviewer_{reviewer}_topics.csv"
        ] = reviewer_topics
        files[
            f"reviewer_{reviewer}_documents.csv"
        ] = reviewer_documents

    existing = [
        name
        for name in files
        if (OUTPUT_DIR / name).exists()
    ]
    if existing:
        raise FileExistsError(
            "Không ghi đè phiếu hiện có: "
            + ", ".join(existing)
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in files.items():
        frame.to_csv(
            OUTPUT_DIR / name,
            index=False,
            encoding="utf-8-sig",
        )


def main():
    assignments, topics, dataset = load_inputs()
    samples, mapping, n_eligible = build_samples(
        assignments, topics, dataset
    )
    export_files(samples, mapping)

    print("Cụm đủ điều kiện:", n_eligible)
    print(
        "Cụm được chọn:",
        mapping["cluster_id"].tolist(),
    )
    print("Số cặp bài–cụm:", len(samples))
    print("Đã lưu trong:", OUTPUT_DIR)


if __name__ == "__main__":
    main()