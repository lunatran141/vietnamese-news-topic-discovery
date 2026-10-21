"""Tổng hợp stability HDBSCAN và độ ổn định từ khóa qua các seed."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "minilm_title_content_eom_15_5"
SEEDS = tuple(range(42, 52))
TOP_N_KEYWORDS = 10

RESULT_DIR = ROOT / "results/hdbscan/stability" / f"{RUN_ID}_seeds42_51"
TRAIN_ASSIGNMENTS_PATH = RESULT_DIR / "train_assignments.csv"
TRAIN_SEED_METRICS_PATH = RESULT_DIR / "train_seed_metrics.csv"
TRAIN_PAIRWISE_ARI_PATH = RESULT_DIR / "train_pairwise_ari.csv"
ARTICLES_PATH = ROOT / "data/processed/p0_minimal.csv"

STABILITY_SUMMARY_PATH = RESULT_DIR / "stability_summary.csv"
PAIRWISE_KEYWORD_JACCARD_PATH = (
    RESULT_DIR / "pairwise_keyword_jaccard.csv"
)


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    """Tính Jaccard của hai tập từ khóa."""
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc và kiểm tra các kết quả E04 cần cho bước tổng hợp."""
    required_paths = (
        TRAIN_ASSIGNMENTS_PATH,
        TRAIN_SEED_METRICS_PATH,
        TRAIN_PAIRWISE_ARI_PATH,
        ARTICLES_PATH,
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Thiếu file đầu vào stability:\n" + "\n".join(missing)
        )

    assignments = pd.read_csv(
        TRAIN_ASSIGNMENTS_PATH,
        dtype={"article_id": str},
    )
    metrics = pd.read_csv(TRAIN_SEED_METRICS_PATH)
    pairwise_ari = pd.read_csv(TRAIN_PAIRWISE_ARI_PATH)
    articles = pd.read_csv(
        ARTICLES_PATH,
        usecols=["article_id", "processed_title", "processed_content"],
        dtype={"article_id": str},
    )

    expected_assignment_columns = {
        "article_id",
        "cluster_id",
        "umap_seed",
    }
    if not expected_assignment_columns.issubset(assignments.columns):
        raise ValueError("train_assignments.csv thiếu cột bắt buộc")
    if assignments[["article_id", "cluster_id", "umap_seed"]].isna().any().any():
        raise ValueError("train_assignments.csv có giá trị thiếu")
    if assignments.duplicated(["umap_seed", "article_id"]).any():
        raise ValueError("Assignments có article_id trùng trong cùng seed")

    actual_seeds = tuple(sorted(assignments["umap_seed"].unique()))
    if actual_seeds != SEEDS:
        raise ValueError(
            f"Cần các seed {SEEDS}, nhận {actual_seeds}"
        )

    counts = assignments.groupby("umap_seed").size()
    if counts.nunique() != 1:
        raise ValueError("Số article không đồng nhất giữa các seed")

    if articles["article_id"].isna().any():
        raise ValueError("Dữ liệu bài báo có article_id thiếu")
    if articles["article_id"].duplicated().any():
        raise ValueError("Dữ liệu bài báo có article_id trùng")

    return assignments, metrics, pairwise_ari, articles


def build_keyword_sets(
    seed_assignments: pd.DataFrame,
    articles: pd.DataFrame,
    top_n: int = TOP_N_KEYWORDS,
) -> dict[int, set[str]]:
    """Tạo tập top-N từ khóa TF-IDF cho mỗi cụm thường của một seed."""
    clustered = seed_assignments.loc[
        seed_assignments["cluster_id"].ne(-1),
        ["article_id", "cluster_id"],
    ].copy()

    clustered = clustered.merge(
        articles,
        on="article_id",
        how="left",
        validate="one_to_one",
    )
    if clustered[["processed_title", "processed_content"]].isna().all(axis=1).any():
        raise ValueError("Có assignment không ghép được nội dung bài báo")

    titles = (
        clustered["processed_title"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    short_contents = (
        clustered["processed_content"]
        .fillna("")
        .astype(str)
        .map(lambda value: " ".join(value.split()[:60]))
    )
    clustered["keyword_text"] = (
        titles + ". " + short_contents
    ).str.strip(" .")
    if clustered["keyword_text"].eq("").any():
        raise ValueError("Có bài trong cụm không có văn bản từ khóa")

    cluster_texts = (
        clustered.groupby("cluster_id", sort=True)["keyword_text"]
        .agg(" ".join)
    )
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.9,
        sublinear_tf=True,
        max_features=50000,
    )
    matrix = vectorizer.fit_transform(cluster_texts.tolist())
    vocabulary = vectorizer.get_feature_names_out()

    result: dict[int, set[str]] = {}
    for row_index, topic_id in enumerate(cluster_texts.index):
        row = matrix.getrow(row_index)
        ranked = sorted(
            zip(row.indices, row.data),
            key=lambda item: (-item[1], vocabulary[item[0]]),
        )[:top_n]
        if len(ranked) != top_n:
            raise ValueError(
                f"Cụm {topic_id} chỉ có {len(ranked)} từ khóa"
            )
        result[int(topic_id)] = {
            str(vocabulary[word_index])
            for word_index, _ in ranked
        }

    return result


def match_topics_by_article_overlap(
    assignments_a: pd.DataFrame,
    assignments_b: pd.DataFrame,
) -> list[tuple[int, int]]:
    """Ghép cụm giữa hai seed bằng Hungarian trên số article chung."""
    joined = assignments_a[["article_id", "cluster_id"]].merge(
        assignments_b[["article_id", "cluster_id"]],
        on="article_id",
        how="inner",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    joined = joined.loc[
        joined["cluster_id_a"].ne(-1)
        & joined["cluster_id_b"].ne(-1)
    ]
    if joined.empty:
        return []

    overlap = pd.crosstab(
        joined["cluster_id_a"],
        joined["cluster_id_b"],
    )
    row_indices, column_indices = linear_sum_assignment(
        -overlap.to_numpy(dtype=float)
    )

    matches = []
    for row_index, column_index in zip(row_indices, column_indices):
        if overlap.iat[row_index, column_index] <= 0:
            continue
        matches.append(
            (
                int(overlap.index[row_index]),
                int(overlap.columns[column_index]),
            )
        )
    return matches


def compute_pairwise_keyword_jaccard(
    assignments: pd.DataFrame,
    keyword_sets: dict[int, dict[int, set[str]]],
) -> pd.DataFrame:
    """Tính Jaccard từ khóa sau khi ghép topic cho 45 cặp seed."""
    rows = []
    assignments_by_seed = {
        int(seed): group[["article_id", "cluster_id"]].copy()
        for seed, group in assignments.groupby("umap_seed", sort=True)
    }

    for seed_a, seed_b in combinations(SEEDS, 2):
        matches = match_topics_by_article_overlap(
            assignments_by_seed[seed_a],
            assignments_by_seed[seed_b],
        )
        scores = [
            jaccard_similarity(
                keyword_sets[seed_a][topic_a],
                keyword_sets[seed_b][topic_b],
            )
            for topic_a, topic_b in matches
        ]

        # Chia cho số topic lớn hơn để các topic không ghép được nhận điểm 0.
        denominator = max(
            len(keyword_sets[seed_a]),
            len(keyword_sets[seed_b]),
        )
        score = sum(scores) / denominator if denominator else 1.0
        rows.append(
            {
                "seed_a": seed_a,
                "seed_b": seed_b,
                "matched_topic_count": len(matches),
                "keyword_jaccard": score,
            }
        )

    return pd.DataFrame(rows)


def mean_pairwise_metric_by_seed(
    pairwise: pd.DataFrame,
    value_column: str,
) -> dict[int, float]:
    """Tính trung bình pairwise metric của mỗi seed với chín seed còn lại."""
    result = {}
    for seed in SEEDS:
        selected = pairwise.loc[
            pairwise["seed_a"].eq(seed)
            | pairwise["seed_b"].eq(seed),
            value_column,
        ]
        if len(selected) != len(SEEDS) - 1:
            raise ValueError(
                f"Seed {seed} không có đủ {len(SEEDS) - 1} cặp "
                f"cho {value_column}"
            )
        result[seed] = float(selected.mean())
    return result


def build_stability_summary(
    metrics: pd.DataFrame,
    pairwise_ari: pd.DataFrame,
    pairwise_keywords: pd.DataFrame,
) -> pd.DataFrame:
    """Tạo bảng stability đúng schema dùng cho common metadata/UI."""
    ari_by_seed = mean_pairwise_metric_by_seed(
        pairwise_ari,
        "pairwise_ari",
    )
    keyword_jaccard_by_seed = mean_pairwise_metric_by_seed(
        pairwise_keywords,
        "keyword_jaccard",
    )

    indexed_metrics = metrics.set_index("umap_seed")
    rows = []
    for seed in SEEDS:
        row = indexed_metrics.loc[seed]
        outlier_rate = float(row["n_noise"] / row["n_articles"])
        rows.append(
            {
                "seed": seed,
                "topic_count": int(row["n_clusters"]),
                "outlier_rate": outlier_rate,
                "coverage": 1.0 - outlier_rate,
                "ari_between_runs": ari_by_seed[seed],
                "keyword_jaccard": keyword_jaccard_by_seed[seed],
            }
        )

    summary = pd.DataFrame(rows)
    numeric_columns = [
        "outlier_rate",
        "coverage",
        "ari_between_runs",
        "keyword_jaccard",
    ]
    if not np.isfinite(summary[numeric_columns].to_numpy()).all():
        raise ValueError("Stability summary có giá trị không hữu hạn")
    if not summary[numeric_columns].apply(lambda column: column.between(0, 1)).all().all():
        raise ValueError("Stability summary có metric ngoài [0, 1]")

    return summary


def main() -> None:
    assignments, metrics, pairwise_ari, articles = load_inputs()

    keyword_sets = {}
    for seed in SEEDS:
        print(f"Đang tạo top-{TOP_N_KEYWORDS} keywords cho seed {seed}...")
        seed_assignments = assignments.loc[
            assignments["umap_seed"].eq(seed)
        ]
        keyword_sets[seed] = build_keyword_sets(
            seed_assignments,
            articles,
        )

    pairwise_keywords = compute_pairwise_keyword_jaccard(
        assignments,
        keyword_sets,
    )
    summary = build_stability_summary(
        metrics,
        pairwise_ari,
        pairwise_keywords,
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    pairwise_keywords.to_csv(
        PAIRWISE_KEYWORD_JACCARD_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        STABILITY_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nStability summary:")
    print(summary.to_string(index=False))
    print(f"\nĐã lưu: {PAIRWISE_KEYWORD_JACCARD_PATH}")
    print(f"Đã lưu: {STABILITY_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
