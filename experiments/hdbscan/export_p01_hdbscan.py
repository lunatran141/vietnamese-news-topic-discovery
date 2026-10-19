"""Xuất HDBSCAN minilm_title_content + eom_15_5 theo schema chung P01."""

import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

from src.artifacts.loader import load_artifact_bundle


ROOT = Path(__file__).resolve().parents[2]

ASSIGNMENTS_PATH = (
    ROOT / "results/hdbscan/minilm_title_content_assignments.csv"
)
SUMMARY_PATH = ROOT / "results/hdbscan/m11_summary.csv"
ARTICLES_PATH = ROOT / "data/processed/p0_minimal.csv"
SPLIT_MANIFEST_PATH = ROOT / "data/splits/split_manifest.csv"
EMBEDDING_CONFIG_PATH = ROOT / "configs/embeddings.yaml"
HDBSCAN_CONFIG_PATH = ROOT / "configs/hdbscan.yaml"

OUTPUT_DIR = (
    ROOT / "artifacts/common/hdbscan/minilm_title_content_eom_15_5"
)
DOCUMENT_TOPICS_PATH = OUTPUT_DIR / "document_topics.csv"
TOPIC_KEYWORDS_PATH = OUTPUT_DIR / "topic_keywords.csv"
TOPIC_SUMMARY_PATH = OUTPUT_DIR / "topic_summary.csv"
CONFIG_USED_PATH = OUTPUT_DIR / "config_used.yaml"
MODEL_SUMMARY_PATH = OUTPUT_DIR / "model_summary.json"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
RUN_METADATA_PATH = OUTPUT_DIR / "run_metadata.json"

EMBEDDING_RUN = "minilm_title_content"
EXPERIMENT = "eom_15_5"
RUN_ID = f"{EMBEDDING_RUN}_{EXPERIMENT}"
EXPECTED_ARTICLES = 6493
TOP_N_KEYWORDS = 10


def build_topic_keywords(run: pd.DataFrame) -> pd.DataFrame:
    """Lấy 10 từ khóa TF-IDF cho mỗi cụm, không gồm outlier -1."""
    clustered = run.loc[run["cluster_id"].ne(-1)].copy()

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
        raise ValueError("Có bài trong cụm không có văn bản")

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

    records = []
    for row_index, topic_id in enumerate(cluster_texts.index):
        row = matrix.getrow(row_index)
        ranked = sorted(
            zip(row.indices, row.data),
            key=lambda item: (
                -item[1],
                vocabulary[item[0]],
            ),
        )[:TOP_N_KEYWORDS]

        if len(ranked) != TOP_N_KEYWORDS:
            raise ValueError(
                f"Cụm {topic_id} chỉ có {len(ranked)} từ khóa"
            )

        for rank, (word_index, weight) in enumerate(
            ranked, start=1
        ):
            records.append(
                {
                    "topic_id": int(topic_id),
                    "rank": rank,
                    "keyword": vocabulary[word_index],
                    "weight": float(weight),
                }
            )

    return pd.DataFrame(
        records,
        columns=["topic_id", "rank", "keyword", "weight"],
    )


def build_topic_summary(
    document_topics: pd.DataFrame,
    topic_keywords: pd.DataFrame,
) -> pd.DataFrame:
    """Tạo thống kê từng cụm, gồm cả dòng outlier -1."""
    counts = document_topics.groupby("topic_id", sort=True).size()
    keyword_lookup = (
        topic_keywords.sort_values(["topic_id", "rank"])
        .groupby("topic_id")["keyword"]
        .apply(list)
        .to_dict()
    )

    total_articles = len(document_topics)
    rows = []

    for topic_id, document_count in counts.items():
        topic_id = int(topic_id)

        if topic_id == -1:
            topic_name = "Outlier"
            top_keywords = ""
        else:
            words = keyword_lookup.get(topic_id)
            if not words:
                raise ValueError(
                    f"Cụm {topic_id} không có từ khóa"
                )

            # Tên tự động; chưa được duyệt thủ công.
            topic_name = " / ".join(words[:3])
            top_keywords = ", ".join(words)

        rows.append(
            {
                "topic_id": topic_id,
                "topic_name": topic_name,
                "document_count": int(document_count),
                # Tỷ phần trong khoảng 0–1.
                "percentage": document_count / total_articles,
                "top_keywords": top_keywords,
            }
        )

    result = pd.DataFrame(
        rows,
        columns=[
            "topic_id",
            "topic_name",
            "document_count",
            "percentage",
            "top_keywords",
        ],
    )

    if result["topic_id"].duplicated().any():
        raise ValueError("topic_summary có topic_id trùng")
    if result["document_count"].sum() != total_articles:
        raise ValueError(
            "Tổng document_count không khớp assignments"
        )
    if not result["percentage"].between(0, 1).all():
        raise ValueError("percentage phải nằm trong [0, 1]")

    return result


def build_config_used() -> dict:
    """Lấy cấu hình thực dùng từ YAML M10 và M11."""
    with EMBEDDING_CONFIG_PATH.open(
        "r", encoding="utf-8"
    ) as file:
        embedding_config = yaml.safe_load(file)

    with HDBSCAN_CONFIG_PATH.open(
        "r", encoding="utf-8"
    ) as file:
        hdbscan_config = yaml.safe_load(file)

    embedding_runs = embedding_config["runs"]
    if EMBEDDING_RUN not in embedding_runs:
        raise ValueError(
            f"Không tìm thấy embedding run {EMBEDDING_RUN}"
        )

    experiments = [
        item
        for item in hdbscan_config["hdbscan"]["experiments"]
        if item["name"] == EXPERIMENT
    ]
    if len(experiments) != 1:
        raise ValueError(
            f"Cần đúng một cấu hình HDBSCAN {EXPERIMENT}"
        )

    embedding_run = embedding_runs[EMBEDDING_RUN]

    return {
        "dataset": hdbscan_config["dataset"],
        "embedding": {
            "run_id": EMBEDDING_RUN,
            **embedding_run,
            "normalize_embeddings": embedding_config[
                "encoding"
            ]["normalize_embeddings"],
        },
        "umap": hdbscan_config["umap"],
        "hdbscan": {
            "metric": hdbscan_config["hdbscan"]["metric"],
            **experiments[0],
        },
        "keyword_extraction": {
            "method": "TF-IDF trên văn bản gom theo cụm",
            "text": (
                "processed_title + 60 từ đầu "
                "processed_content"
            ),
            "ngram_range": [1, 2],
            "max_df": 0.9,
            "sublinear_tf": True,
            "max_features": 50000,
            "top_n_keywords": TOP_N_KEYWORDS,
            "exclude_outlier_topic": True,
        },
    }


def write_bundle_metadata(
    config_used: dict,
    n_articles: int,
    n_clusters: int,
    n_outliers: int,
) -> None:
    """Xuất ba file metadata bắt buộc của bundle chung."""
    with CONFIG_USED_PATH.open(
        "w", encoding="utf-8"
    ) as file:
        yaml.safe_dump(
            config_used,
            file,
            allow_unicode=True,
            sort_keys=False,
        )

    model_name = f"HDBSCAN {RUN_ID}"
    model_summary = {
        "model_name": model_name,
        "run_id": RUN_ID,
        "algorithm": "HDBSCAN",
        "representation": config_used["embedding"]["model_name"],
        "topic_count": n_clusters,
        "outlier_rate": n_outliers / n_articles,
        # M11 chưa lưu runtime của run này.
        "runtime_seconds": None,
        "metrics": {
            "train_count": n_articles,
            "train_outlier_count": n_outliers,
            "train_assigned_count": n_articles - n_outliers,
        },
        "config": config_used,
        "topic_probability_meaning": (
            "membership_strength của HDBSCAN trong [0, 1]; "
            "không phải xác suất đã hiệu chuẩn"
        ),
        "topic_name_method": (
            "Tên tự động ghép từ ba từ khóa TF-IDF đầu; "
            "chưa được duyệt thủ công"
        ),
    }

    with MODEL_SUMMARY_PATH.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            model_summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    manifest = {
        "schema_version": "1.0",
        "model_name": model_name,
        "run_id": RUN_ID,
        "model_summary_path": "model_summary.json",
        "topic_summary_path": "topic_summary.csv",
        "document_topics_path": "document_topics.csv",
        "topic_keywords_path": "topic_keywords.csv",
        "config_path": "config_used.yaml",
    }

    with MANIFEST_PATH.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )


def write_run_metadata(
    summary_row: pd.Series,
    config_used: dict,
    n_articles: int,
    n_clusters: int,
    n_outliers: int,
) -> None:
    """Lưu nguồn và kết quả của đúng run M11 được xuất."""
    metadata = {
        "run_id": RUN_ID,
        "embedding_run": EMBEDDING_RUN,
        "experiment": EXPERIMENT,
        "split": "train",
        "config_hash": str(summary_row["config_hash"]),
        "source_files": {
            "assignments": (
                ASSIGNMENTS_PATH.relative_to(ROOT).as_posix()
            ),
            "m11_summary": (
                SUMMARY_PATH.relative_to(ROOT).as_posix()
            ),
            "articles": ARTICLES_PATH.relative_to(
                ROOT
            ).as_posix(),
            "split_manifest": (
                SPLIT_MANIFEST_PATH.relative_to(ROOT).as_posix()
            ),
            "embedding_artifact": config_used["embedding"][
                "artifact_path"
            ],
        },
        "n_articles": n_articles,
        "n_clusters": n_clusters,
        "n_outliers": n_outliers,
        "outlier_rate": n_outliers / n_articles,
        "n_assigned": n_articles - n_outliers,
        "keyword_method": (
            "TF-IDF trên processed_title và 60 từ đầu "
            "processed_content, gom theo cụm; không gồm -1"
        ),
    }

    with RUN_METADATA_PATH.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    assignments = pd.read_csv(
        ASSIGNMENTS_PATH,
        dtype={"article_id": str},
    )
    summary = pd.read_csv(SUMMARY_PATH)
    split_manifest = pd.read_csv(
        SPLIT_MANIFEST_PATH,
        usecols=["article_id", "split"],
        dtype={"article_id": str},
    )

    run = assignments.loc[
        assignments["embedding_run"].eq(EMBEDDING_RUN)
        & assignments["experiment"].eq(EXPERIMENT)
    ].copy()
    summary_run = summary.loc[
        summary["embedding_run"].eq(EMBEDDING_RUN)
        & summary["experiment"].eq(EXPERIMENT)
    ]

    if len(summary_run) != 1:
        raise ValueError(
            "Cần đúng một dòng M11 summary cho run"
        )
    if len(run) != EXPECTED_ARTICLES:
        raise ValueError(
            f"Run có {len(run)} bài, cần {EXPECTED_ARTICLES}"
        )
    if run["article_id"].isna().any():
        raise ValueError("article_id bị thiếu")
    if run["article_id"].duplicated().any():
        raise ValueError("article_id bị trùng")
    if run["cluster_id"].isna().any():
        raise ValueError("cluster_id bị thiếu")
    if run["membership_strength"].isna().any():
        raise ValueError("membership_strength bị thiếu")
    if not run["membership_strength"].between(0, 1).all():
        raise ValueError(
            "membership_strength phải nằm trong [0, 1]"
        )

    train_ids = split_manifest.loc[
        split_manifest["split"].eq("train"),
        "article_id",
    ]
    if train_ids.isna().any():
        raise ValueError(
            "Split train có article_id bị thiếu"
        )
    if train_ids.duplicated().any():
        raise ValueError(
            "Split train có article_id trùng"
        )
    if len(train_ids) != EXPECTED_ARTICLES:
        raise ValueError(
            "Split train không có đúng 6.493 ID"
        )
    if set(run["article_id"]) != set(train_ids):
        raise ValueError(
            "Run không khớp các ID của split train"
        )

    expected = summary_run.iloc[0]
    n_outliers = int(run["cluster_id"].eq(-1).sum())
    n_clusters = int(
        run.loc[
            run["cluster_id"].ne(-1),
            "cluster_id",
        ].nunique()
    )

    if int(expected["n_articles"]) != len(run):
        raise ValueError("Số bài không khớp M11 summary")
    if int(expected["n_noise"]) != n_outliers:
        raise ValueError(
            "Số bài -1 không khớp M11 summary"
        )
    if int(expected["n_clusters"]) != n_clusters:
        raise ValueError(
            "Số cụm không khớp M11 summary"
        )
    if "config_hash" not in run.columns:
        raise ValueError(
            "Assignments thiếu cột config_hash"
        )
    if pd.isna(expected["config_hash"]):
        raise ValueError(
            "M11 summary thiếu config_hash"
        )
    if not run["config_hash"].eq(
        expected["config_hash"]
    ).all():
        raise ValueError(
            "config_hash trong assignments "
            "không khớp M11 summary"
        )

    articles = pd.read_csv(
        ARTICLES_PATH,
        usecols=[
            "article_id",
            "title",
            "processed_title",
            "processed_content",
        ],
        dtype={"article_id": str},
    )
    if articles["article_id"].duplicated().any():
        raise ValueError(
            "article_id trong dữ liệu gốc bị trùng"
        )

    run = run.merge(
        articles,
        on="article_id",
        how="left",
        validate="one_to_one",
    )
    if run["title"].isna().any():
        raise ValueError(
            "Có bài không ghép được tiêu đề"
        )

    # Cột chung chứa membership_strength của HDBSCAN,
    # không phải xác suất đã hiệu chuẩn.
    document_topics = pd.DataFrame(
        {
            "article_id": run["article_id"],
            "split": "train",
            "topic_id": run["cluster_id"].astype(int),
            "topic_probability": run[
                "membership_strength"
            ].astype(float),
            "is_outlier": run["cluster_id"].eq(-1),
            "title": run["title"],
        }
    )

    topic_keywords = build_topic_keywords(run)

    expected_topic_ids = set(
        run.loc[
            run["cluster_id"].ne(-1),
            "cluster_id",
        ].astype(int)
    )
    actual_topic_ids = set(topic_keywords["topic_id"])

    if actual_topic_ids != expected_topic_ids:
        raise ValueError(
            "Có cụm thường bị thiếu từ khóa"
        )
    if topic_keywords["topic_id"].eq(-1).any():
        raise ValueError(
            "Không được trích từ khóa cho cụm -1"
        )
    if (
        topic_keywords.groupby("topic_id")
        .size()
        .ne(TOP_N_KEYWORDS)
        .any()
    ):
        raise ValueError(
            "Mỗi cụm phải có đúng 10 từ khóa"
        )

    topic_summary = build_topic_summary(
        document_topics,
        topic_keywords,
    )
    if len(topic_summary) != n_clusters + 1:
        raise ValueError(
            "topic_summary phải gồm các cụm và dòng -1"
        )

    summary_outliers = int(
        topic_summary.loc[
            topic_summary["topic_id"].eq(-1),
            "document_count",
        ].sum()
    )
    if summary_outliers != n_outliers:
        raise ValueError(
            "Số outlier trong topic_summary không khớp"
        )

    config_used = build_config_used()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    document_topics.to_csv(
        DOCUMENT_TOPICS_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    topic_keywords.to_csv(
        TOPIC_KEYWORDS_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    topic_summary.to_csv(
        TOPIC_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    write_bundle_metadata(
        config_used=config_used,
        n_articles=len(document_topics),
        n_clusters=n_clusters,
        n_outliers=n_outliers,
    )
    write_run_metadata(
        summary_row=expected,
        config_used=config_used,
        n_articles=len(document_topics),
        n_clusters=n_clusters,
        n_outliers=n_outliers,
    )

    # Dùng chính loader mà Streamlit sử dụng để kiểm tra bundle.
    bundle = load_artifact_bundle(OUTPUT_DIR)
    if len(bundle.document_topics) != EXPECTED_ARTICLES:
        raise ValueError(
            "Loader đọc số bài không khớp artifact"
        )
    if len(bundle.topic_summary) != n_clusters + 1:
        raise ValueError(
            "Loader đọc số dòng topic summary không khớp"
        )
    if len(bundle.topic_keywords) != (
        n_clusters * TOP_N_KEYWORDS
    ):
        raise ValueError(
            "Loader đọc số dòng từ khóa không khớp"
        )

    print("Đã lưu:", DOCUMENT_TOPICS_PATH)
    print("Đã lưu:", TOPIC_KEYWORDS_PATH)
    print("Đã lưu:", TOPIC_SUMMARY_PATH)
    print("Đã lưu:", CONFIG_USED_PATH)
    print("Đã lưu:", MODEL_SUMMARY_PATH)
    print("Đã lưu:", MANIFEST_PATH)
    print("Đã lưu:", RUN_METADATA_PATH)
    print("Số bài:", len(bundle.document_topics))
    print("Số cụm:", n_clusters)
    print("Số bài -1:", n_outliers)
    print(
        "ID duy nhất:",
        bundle.document_topics["article_id"].nunique(),
    )
    print(
        "Số cụm có từ khóa:",
        bundle.topic_keywords["topic_id"].nunique(),
    )
    print("Số dòng từ khóa:", len(bundle.topic_keywords))
    print("Số dòng topic summary:", len(bundle.topic_summary))
    print(
        "Tổng document_count:",
        int(bundle.topic_summary["document_count"].sum()),
    )
    print("Loader chung đã đọc thành công:", OUTPUT_DIR)


if __name__ == "__main__":
    main()