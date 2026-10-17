"""Đọc và kiểm tra dữ liệu đầu vào cho phân tích outlier M12."""

from pathlib import Path
import pandas as pd
import numpy as np
from src.features.embeddings import load_embedding_artifact
from src.data.dataset_loader import load_split_dataset

def load_m12_inputs(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Trả về tập train, summary M11 và toàn bộ assignments đã kiểm tra."""
    dataset_config = config["dataset"]
    train = load_split_dataset(
        dataset_config["path"],
        dataset_config["split_manifest_path"],
        dataset_config["split"],
    )

    expected_count = dataset_config["expected_articles"]
    if len(train) != expected_count:
        raise ValueError(
            f"Tập train có {len(train)} bài; mong đợi {expected_count}"
        )

    expected_ids = set(train["article_id"].astype(str))

    summary = pd.read_csv(
        config["m11"]["summary_path"],
        encoding="utf-8-sig",
    )
    required_summary = {
        "embedding_run",
        "experiment",
        "config_hash",
        "n_articles",
        "n_noise",
    }
    missing = required_summary - set(summary.columns)
    if missing:
        raise ValueError(f"Summary M11 thiếu cột: {sorted(missing)}")

    if summary.duplicated(["embedding_run", "experiment"]).any():
        raise ValueError("Summary M11 có run bị trùng")

    all_assignments = []

    for embedding_run, file_path in config["m11"]["assignment_paths"].items():
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy assignments: {path}")

        table = pd.read_csv(
            path,
            encoding="utf-8-sig",
            dtype={"article_id": str},
        )

        required_assignment = {
            "article_id",
            "cluster_id",
            "embedding_run",
            "experiment",
            "config_hash",
        }
        missing = required_assignment - set(table.columns)
        if missing:
            raise ValueError(
                f"{path.name} thiếu cột: {sorted(missing)}"
            )

        if table["article_id"].isna().any():
            raise ValueError(f"{path.name} có article_id bị thiếu")

        if not table["embedding_run"].eq(embedding_run).all():
            raise ValueError(
                f"{path.name} có embedding_run không đúng"
            )

        summary_for_embedding = summary.loc[
            summary["embedding_run"] == embedding_run
        ]
        expected_experiments = set(
            summary_for_embedding["experiment"]
        )
        actual_experiments = set(table["experiment"])

        if actual_experiments != expected_experiments:
            raise ValueError(
                f"{embedding_run}: danh sách experiment không khớp summary"
            )

        for experiment in expected_experiments:
            rows = table.loc[table["experiment"] == experiment]
            summary_row = summary_for_embedding.loc[
                summary_for_embedding["experiment"] == experiment
            ].iloc[0]

            if len(rows) != expected_count:
                raise ValueError(
                    f"{embedding_run}/{experiment}: "
                    f"có {len(rows)} bài, mong đợi {expected_count}"
                )

            if rows["article_id"].duplicated().any():
                raise ValueError(
                    f"{embedding_run}/{experiment}: article_id bị trùng"
                )

            if set(rows["article_id"]) != expected_ids:
                raise ValueError(
                    f"{embedding_run}/{experiment}: ID không khớp tập train"
                )

            actual_noise = int(rows["cluster_id"].eq(-1).sum())
            if actual_noise != int(summary_row["n_noise"]):
                raise ValueError(
                    f"{embedding_run}/{experiment}: số bài nhiễu "
                    "không khớp summary"
                )

            if int(summary_row["n_articles"]) != expected_count:
                raise ValueError(
                    f"{embedding_run}/{experiment}: n_articles sai"
                )

            if not rows["config_hash"].eq(
                summary_row["config_hash"]
            ).all():
                raise ValueError(
                    f"{embedding_run}/{experiment}: config_hash không khớp"
                )

        all_assignments.append(table)

    assignments = pd.concat(
        all_assignments,
        ignore_index=True,
    )
    return train, summary, assignments

def calculate_outlier_rates(
    train: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Tính tỷ lệ outlier theo run, nguồn báo và độ dài nội dung."""
    metadata = train[
        ["article_id", "source", "processed_content"]
    ].copy()

    metadata["source"] = (
        metadata["source"].fillna("unknown").astype(str)
    )
    metadata["content_words"] = (
        metadata["processed_content"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )
    metadata["length_group"] = pd.cut(
        metadata["content_words"],
        bins=[-1, 99, 299, 599, float("inf")],
        labels=["0-99", "100-299", "300-599", "600+"],
    )

    data = assignments[
        ["article_id", "embedding_run", "experiment", "cluster_id"]
    ].merge(
        metadata[["article_id", "source", "length_group"]],
        on="article_id",
        how="left",
        validate="many_to_one",
    )

    if data["length_group"].isna().any():
        raise ValueError("Có bài không ghép được với dữ liệu train")

    data["is_outlier"] = data["cluster_id"].eq(-1)

    def summarize(group_type: str, extra_column: str | None) -> pd.DataFrame:
        columns = ["embedding_run", "experiment"]
        if extra_column is not None:
            columns.append(extra_column)

        result = (
            data.groupby(columns, observed=True, dropna=False)["is_outlier"]
            .agg(n_articles="size", n_outliers="sum")
            .reset_index()
        )

        result["group_type"] = group_type
        if extra_column is None:
            result["group_value"] = "all"
        else:
            result["group_value"] = result[extra_column].astype(str)
            result = result.drop(columns=extra_column)

        result["outlier_percent"] = (
            100 * result["n_outliers"] / result["n_articles"]
        ).round(2)
        return result

    rates = pd.concat(
        [
            summarize("overall", None),
            summarize("source", "source"),
            summarize("content_length", "length_group"),
        ],
        ignore_index=True,
    )

    return rates[
        [
            "embedding_run",
            "experiment",
            "group_type",
            "group_value",
            "n_articles",
            "n_outliers",
            "outlier_percent",
        ]
    ].sort_values(
        ["embedding_run", "experiment", "group_type", "group_value"]
    ).reset_index(drop=True)

def sample_outlier_candidates(
    train: pd.DataFrame,
    assignments: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Lấy mẫu bài mang nhãn -1 để đọc và phân loại thủ công."""
    review = config["case_review"]
    embedding_run = review["embedding_run"]
    experiments = review["experiments"]
    sample_size = int(review["sample_size_per_experiment"])
    random_state = int(review["random_state"])

    if sample_size <= 0:
        raise ValueError("sample_size_per_experiment phải lớn hơn 0")

    metadata = train[
        [
            "article_id",
            "title",
            "content",
            "processed_content",
            "source",
            "published_at",
            "url",
        ]
    ].copy()

    metadata["source"] = (
        metadata["source"].fillna("unknown").astype(str)
    )
    metadata["content_words"] = (
        metadata["processed_content"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )
    metadata["length_group"] = pd.cut(
        metadata["content_words"],
        bins=[-1, 99, 299, 599, float("inf")],
        labels=["0-99", "100-299", "300-599", "600+"],
    )
    metadata["content_excerpt"] = (
        metadata["content"].fillna("").astype(str).str.slice(0, 500)
    )
    metadata = metadata.drop(columns=["content", "processed_content"])

    samples = []

    for experiment in experiments:
        outliers = assignments.loc[
            (assignments["embedding_run"] == embedding_run)
            & (assignments["experiment"] == experiment)
            & (assignments["cluster_id"] == -1),
            [
                "article_id",
                "embedding_run",
                "experiment",
                "cluster_id",
                "config_hash",
            ],
        ]

        if len(outliers) < sample_size:
            raise ValueError(
                f"{embedding_run}/{experiment} chỉ có "
                f"{len(outliers)} outlier; cần {sample_size}"
            )

        pool = outliers.merge(
            metadata,
            on="article_id",
            how="left",
            validate="one_to_one",
        )

        if pool["title"].isna().any():
            raise ValueError(
                f"{embedding_run}/{experiment}: có ID không tìm thấy bài gốc"
            )

        pool["length_group"] = pool["length_group"].astype(str)
        pool["stratum"] = (
            pool["source"] + "|" + pool["length_group"]
        )

        # Trộn bài trong từng nhóm nguồn báo × độ dài.
        groups = {
            name: group.sample(
                frac=1,
                random_state=random_state,
            ).index.tolist()
            for name, group in pool.groupby("stratum")
        }

        # Lấy lần lượt từ các nhóm để mẫu có nhiều loại bài.
        selected_indices = []
        while len(selected_indices) < sample_size:
            found = False
            for name in sorted(groups):
                if groups[name] and len(selected_indices) < sample_size:
                    selected_indices.append(groups[name].pop())
                    found = True
            if not found:
                break

        selected = pool.loc[selected_indices].drop(columns="stratum")
        samples.append(selected)

    candidates = pd.concat(samples, ignore_index=True)

    return candidates[
        [
            "embedding_run",
            "experiment",
            "config_hash",
            "article_id",
            "cluster_id",
            "title",
            "content_excerpt",
            "source",
            "published_at",
            "url",
            "content_words",
            "length_group",
        ]
    ]

def find_nearest_cluster_articles(
    candidates: pd.DataFrame,
    assignments: pd.DataFrame,
    train: pd.DataFrame,
    embedding_path: str,
    top_k: int = 3,
) -> pd.DataFrame:
    """Tìm bài đã vào cụm gần mỗi outlier trong cùng một run."""
    if top_k <= 0:
        raise ValueError("top_k phải lớn hơn 0")

    embeddings = load_embedding_artifact(embedding_path)
    embedding_ids = embeddings["article_id"].astype(str)

    if embedding_ids.duplicated().any():
        raise ValueError("Artifact embedding có article_id trùng")

    vectors = np.stack(
        embeddings["embedding"].to_numpy()
    ).astype(np.float32)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or (norms == 0).any():
        raise ValueError("Embedding có vector không hợp lệ")

    vectors = vectors / norms
    positions = pd.Series(
        np.arange(len(embedding_ids)),
        index=embedding_ids,
    )
    titles = train.set_index("article_id")["title"]
    results = []

    for (embedding_run, experiment), sample in candidates.groupby(
        ["embedding_run", "experiment"],
        sort=False,
    ):
        assigned = assignments.loc[
            (assignments["embedding_run"] == embedding_run)
            & (assignments["experiment"] == experiment)
            & (assignments["cluster_id"] != -1),
            ["article_id", "cluster_id"],
        ].reset_index(drop=True)

        if assigned.empty:
            raise ValueError(
                f"{embedding_run}/{experiment} không có bài đã vào cụm"
            )

        sample_ids = sample["article_id"].astype(str).tolist()
        assigned_ids = assigned["article_id"].astype(str).tolist()

        if not set(sample_ids + assigned_ids).issubset(positions.index):
            raise ValueError("Có article_id không tồn tại trong embedding")

        sample_vectors = vectors[positions.loc[sample_ids].to_numpy()]
        assigned_vectors = vectors[
            positions.loc[assigned_ids].to_numpy()
        ]
        similarities = sample_vectors @ assigned_vectors.T

        count = min(top_k, len(assigned))
        nearest_indices = np.argsort(
            -similarities,
            axis=1,
        )[:, :count]

        for sample_row, candidate_id in enumerate(sample_ids):
            for rank, assigned_row in enumerate(
                nearest_indices[sample_row],
                start=1,
            ):
                neighbor = assigned.iloc[assigned_row]
                neighbor_id = str(neighbor["article_id"])

                results.append({
                    "embedding_run": embedding_run,
                    "experiment": experiment,
                    "outlier_article_id": candidate_id,
                    "outlier_title": titles.loc[candidate_id],
                    "neighbor_rank": rank,
                    "neighbor_article_id": neighbor_id,
                    "neighbor_title": titles.loc[neighbor_id],
                    "neighbor_cluster_id": int(
                        neighbor["cluster_id"]
                    ),
                    "cosine_similarity": round(
                        float(similarities[sample_row, assigned_row]),
                        4,
                    ),
                })

    return pd.DataFrame(results)