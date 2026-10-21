"""Xuất HDBSCAN MiniLM + eom_15_5 theo schema chung P01.

Các bảng dữ liệu xuất trên train + validation + test:
- document_topics.csv
- topic_summary.csv
- topic_keywords.csv
- representative_docs.csv
- projection_2d.csv

HDBSCAN đã fit trên train ở M11. Validation chỉ được gán nhãn
bằng model train đã fit; exporter này không fit lại HDBSCAN.

UMAP 2D chỉ phục vụ hiển thị: fit trên train rồi transform val.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP

from src.artifacts.loader import load_artifact_bundle
from src.data.dataset_loader import load_split_dataset
from src.features.embeddings import (
    create_embeddings,
    load_embedding_artifact,
)


ROOT = Path(__file__).resolve().parents[2]

TRAIN_ASSIGNMENTS_PATH = (
    ROOT / "results/hdbscan/minilm_title_content_assignments.csv"
)
VAL_ASSIGNMENTS_PATH = (
    ROOT
    / "results/hdbscan/validation/"
    "minilm_title_content_eom_15_5_val_assignments.csv"
)
TEST_ASSIGNMENTS_PATH = (
    ROOT / "results/hdbscan/test/"
    "minilm_title_content_eom_15_5_test_assignments.csv"
)
TEST_METRICS_PATH = (
    ROOT / "results/hdbscan/test/"
    "minilm_title_content_eom_15_5_test_metrics.json"
)
M11_SUMMARY_PATH = ROOT / "results/hdbscan/m11_summary.csv"
ARTICLES_PATH = ROOT / "data/processed/p0_minimal.csv"
SPLIT_MANIFEST_PATH = ROOT / "data/splits/split_manifest.csv"
EMBEDDING_CONFIG_PATH = ROOT / "configs/embeddings.yaml"
HDBSCAN_CONFIG_PATH = ROOT / "configs/hdbscan.yaml"

OUTPUT_DIR = (
    ROOT
    / "artifacts/common/hdbscan/"
    "minilm_title_content_eom_15_5"
)
DOCUMENT_TOPICS_PATH = OUTPUT_DIR / "document_topics.csv"
TOPIC_SUMMARY_PATH = OUTPUT_DIR / "topic_summary.csv"
TOPIC_KEYWORDS_PATH = OUTPUT_DIR / "topic_keywords.csv"
REPRESENTATIVE_DOCS_PATH = OUTPUT_DIR / "representative_docs.csv"
PROJECTION_2D_PATH = OUTPUT_DIR / "projection_2d.csv"
CONFIG_USED_PATH = OUTPUT_DIR / "config_used.yaml"
MODEL_SUMMARY_PATH = OUTPUT_DIR / "model_summary.json"
RUN_METADATA_PATH = OUTPUT_DIR / "run_metadata.json"
RUNTIME_SUMMARY_PATH = OUTPUT_DIR / "runtime_summary.json"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
BENCHMARK_PATH = ROOT / "results/hdbscan/benchmark_p01_runtime.json"
EMBEDDING_RUN = "minilm_title_content"
EXPERIMENT = "eom_15_5"
RUN_ID = f"{EMBEDDING_RUN}_{EXPERIMENT}"

N_TRAIN = 6493
N_VAL = 1391
N_TEST = 1390
N_TOTAL = N_TRAIN + N_VAL + N_TEST
N_TRAIN_OUTLIERS = 1608
N_VAL_OUTLIERS = 484
N_CLUSTERS = 101

TOP_N_KEYWORDS = 10
TOP_N_REPRESENTATIVES = 5


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def read_split_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(
        SPLIT_MANIFEST_PATH,
        usecols=["article_id", "split"],
        dtype={"article_id": str},
    )
    if manifest["article_id"].isna().any():
        raise ValueError("Split manifest có article_id thiếu")
    if manifest["article_id"].duplicated().any():
        raise ValueError("Split manifest có article_id trùng")

    counts = manifest["split"].value_counts().to_dict()
    if counts.get("train") != N_TRAIN:
        raise ValueError("Split train không có 6.493 bài")
    if counts.get("val") != N_VAL:
        raise ValueError("Split val không có 1.391 bài")
    if counts.get("test") != N_TEST:
        raise ValueError("Split test không có 1.390 bài")

    return manifest


def read_articles() -> pd.DataFrame:
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
    if articles["article_id"].isna().any():
        raise ValueError("Dữ liệu bài có article_id thiếu")
    if articles["article_id"].duplicated().any():
        raise ValueError("Dữ liệu bài có article_id trùng")
    return articles


def select_train_assignments(
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    assignments = pd.read_csv(
        TRAIN_ASSIGNMENTS_PATH,
        dtype={"article_id": str},
    )
    summary = pd.read_csv(M11_SUMMARY_PATH)

    train = assignments.loc[
        assignments["embedding_run"].eq(EMBEDDING_RUN)
        & assignments["experiment"].eq(EXPERIMENT)
    ].copy()
    summary_rows = summary.loc[
        summary["embedding_run"].eq(EMBEDDING_RUN)
        & summary["experiment"].eq(EXPERIMENT)
    ]

    if len(summary_rows) != 1:
        raise ValueError("Cần đúng một dòng M11 summary")
    if len(train) != N_TRAIN:
        raise ValueError("M11 train không có 6.493 bài")
    if train["article_id"].isna().any():
        raise ValueError("M11 train có article_id thiếu")
    if train["article_id"].duplicated().any():
        raise ValueError("M11 train có article_id trùng")

    train_ids = set(
        manifest.loc[
            manifest["split"].eq("train"),
            "article_id",
        ]
    )
    if set(train["article_id"]) != train_ids:
        raise ValueError(
            "ID M11 train không khớp split manifest"
        )

    expected = summary_rows.iloc[0]
    if pd.isna(expected["config_hash"]):
        raise ValueError("M11 summary thiếu config_hash")
    if "config_hash" not in train.columns:
        raise ValueError("M11 train thiếu config_hash")
    if not train["config_hash"].eq(
        expected["config_hash"]
    ).all():
        raise ValueError(
            "config_hash train không khớp M11 summary"
        )

    if train["cluster_id"].isna().any():
        raise ValueError("M11 train có cluster_id thiếu")
    if train["membership_strength"].isna().any():
        raise ValueError(
            "M11 train có membership_strength thiếu"
        )
    if not train["membership_strength"].between(0, 1).all():
        raise ValueError(
            "membership_strength train ngoài [0, 1]"
        )

    n_outliers = int(train["cluster_id"].eq(-1).sum())
    n_clusters = int(
        train.loc[
            train["cluster_id"].ne(-1),
            "cluster_id",
        ].nunique()
    )

    if int(expected["n_articles"]) != N_TRAIN:
        raise ValueError("M11 summary sai số bài train")
    if int(expected["n_noise"]) != n_outliers:
        raise ValueError("M11 summary sai số outlier")
    if int(expected["n_clusters"]) != n_clusters:
        raise ValueError("M11 summary sai số cụm")
    if n_outliers != N_TRAIN_OUTLIERS:
        raise ValueError("Train không có 1.608 outlier")
    if n_clusters != N_CLUSTERS:
        raise ValueError("Train không có 101 cụm")

    train["split"] = "train"
    return train, expected


def select_val_assignments(
    manifest: pd.DataFrame,
    config_hash: str,
    train_topic_ids: set[int],
) -> pd.DataFrame:
    assignments = pd.read_csv(
        VAL_ASSIGNMENTS_PATH,
        encoding="utf-8-sig",
        dtype={"article_id": str},
    )

    required = {
        "article_id",
        "split",
        "cluster_id",
        "membership_strength",
        "embedding_run",
        "experiment",
        "config_hash",
    }
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(
            f"Validation assignments thiếu: {sorted(missing)}"
        )

    val = assignments.loc[
        assignments["embedding_run"].eq(EMBEDDING_RUN)
        & assignments["experiment"].eq(EXPERIMENT)
    ].copy()

    if len(val) != N_VAL:
        raise ValueError("M11 validation không có 1.391 bài")
    if val["article_id"].isna().any():
        raise ValueError("Validation có article_id thiếu")
    if val["article_id"].duplicated().any():
        raise ValueError("Validation có article_id trùng")
    if not val["split"].eq("val").all():
        raise ValueError("Validation có split khác val")
    if not val["config_hash"].eq(config_hash).all():
        raise ValueError(
            "config_hash validation không khớp train"
        )

    val_ids = set(
        manifest.loc[
            manifest["split"].eq("val"),
            "article_id",
        ]
    )
    if set(val["article_id"]) != val_ids:
        raise ValueError(
            "ID validation không khớp split manifest"
        )

    if val["cluster_id"].isna().any():
        raise ValueError("Validation có cluster_id thiếu")
    if not set(val["cluster_id"]).issubset(
        train_topic_ids
    ):
        raise ValueError(
            "Validation xuất hiện cụm không có ở train"
        )

    strengths = pd.to_numeric(
        val["membership_strength"],
        errors="coerce",
    )
    if strengths.isna().any():
        raise ValueError(
            "Validation có membership_strength không hợp lệ"
        )
    if not strengths.between(0, 1).all():
        raise ValueError(
            "membership_strength validation ngoài [0, 1]"
        )
    val["membership_strength"] = strengths

    if int(val["cluster_id"].eq(-1).sum()) != (
        N_VAL_OUTLIERS
    ):
        raise ValueError(
            "Validation không có 484 outlier"
        )

    return val


def select_test_assignments(
    manifest: pd.DataFrame,
    config_hash: str,
    train_topic_ids: set[int],
) -> tuple[pd.DataFrame, dict]:
    """Read held-out predictions from the locked train-fitted model."""
    test = pd.read_csv(
        TEST_ASSIGNMENTS_PATH, encoding="utf-8-sig", dtype={"article_id": str}
    )
    metrics = json.loads(TEST_METRICS_PATH.read_text(encoding="utf-8"))
    required = {
        "article_id", "split", "cluster_id", "membership_strength",
        "embedding_run", "experiment", "config_hash",
    }
    if not required.issubset(test.columns):
        raise ValueError(f"Test assignments thiếu: {sorted(required - set(test.columns))}")
    if len(test) != N_TEST or test["article_id"].duplicated().any():
        raise ValueError("Test assignments không có đúng 1.390 ID duy nhất")
    if not test["split"].eq("test").all():
        raise ValueError("Test assignments có split khác test")
    if not test["embedding_run"].eq(EMBEDDING_RUN).all():
        raise ValueError("Test assignments thuộc embedding khác")
    if not test["experiment"].eq(EXPERIMENT).all():
        raise ValueError("Test assignments thuộc experiment khác")
    if not test["config_hash"].eq(config_hash).all():
        raise ValueError("Test assignments có config_hash khác train")
    expected_ids = set(manifest.loc[manifest["split"].eq("test"), "article_id"])
    if set(test["article_id"]) != expected_ids:
        raise ValueError("Test assignments không khớp split manifest")
    if test["cluster_id"].isna().any() or not set(test["cluster_id"]).issubset(train_topic_ids):
        raise ValueError("Test có cụm không tồn tại trên train")
    strengths = pd.to_numeric(test["membership_strength"], errors="coerce")
    if strengths.isna().any() or not strengths.between(0, 1).all():
        raise ValueError("Test có membership_strength không hợp lệ")
    test["membership_strength"] = strengths
    if (
        metrics.get("run_id") != RUN_ID
        or metrics.get("config_hash") != config_hash
        or metrics.get("fit_split") != "train"
        or metrics.get("split") != "test"
        or metrics.get("test_count") != N_TEST
        or metrics.get("test_outlier_count") != int(test["cluster_id"].eq(-1).sum())
    ):
        raise ValueError("Test metrics không khớp assignments hoặc run đã khóa")
    return test, metrics


def build_document_topics(
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Đổi train + val + test assignments sang schema chung."""
    result = pd.DataFrame(
        {
            "article_id": assignments["article_id"],
            "split": assignments["split"],
            "topic_id": assignments["cluster_id"].astype(int),
            # HDBSCAN membership strength, không phải
            # xác suất đã hiệu chuẩn.
            "topic_probability": assignments[
                "membership_strength"
            ].astype(float),
            "is_outlier": assignments["cluster_id"].eq(-1),
            "title": assignments["title"],
        }
    )

    if len(result) != N_TOTAL:
        raise ValueError(
            "document_topics không có 9.274 bài"
        )
    if result["article_id"].duplicated().any():
        raise ValueError(
            "document_topics có article_id trùng"
        )
    if result["split"].value_counts().to_dict() != {
        "train": N_TRAIN,
        "val": N_VAL,
        "test": N_TEST,
    }:
        raise ValueError(
            "document_topics sai số bài theo split"
        )
    if result["title"].isna().any():
        raise ValueError(
            "Có bài không ghép được tiêu đề"
        )

    return result


def build_topic_keywords(
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Trích TF-IDF từ bài train + val + test đã được gán cụm."""
    clustered = assignments.loc[
        assignments["cluster_id"].ne(-1)
    ].copy()

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
        raise ValueError(
            "Có bài trong cụm không có văn bản từ khóa"
        )

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
                f"Cụm {topic_id} chỉ có "
                f"{len(ranked)} từ khóa"
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
    documents: pd.DataFrame,
    keywords: pd.DataFrame,
) -> pd.DataFrame:
    """Thống kê cụm trên train + val + test."""
    counts = documents.groupby("topic_id", sort=True).size()
    keyword_lookup = (
        keywords.sort_values(["topic_id", "rank"])
        .groupby("topic_id")["keyword"]
        .apply(list)
        .to_dict()
    )

    rows = []
    for topic_id, count in counts.items():
        topic_id = int(topic_id)

        if topic_id == -1:
            name = "Outlier"
            top_keywords = ""
        else:
            words = keyword_lookup.get(topic_id)
            if not words:
                raise ValueError(
                    f"Cụm {topic_id} không có từ khóa"
                )
            name = " / ".join(words[:3])
            top_keywords = ", ".join(words)

        rows.append(
            {
                "topic_id": topic_id,
                "topic_name": name,
                "document_count": int(count),
                "percentage": count / len(documents),
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
    if int(result["document_count"].sum()) != N_TOTAL:
        raise ValueError(
            "Tổng document_count không bằng 9.274"
        )
    if not result["percentage"].between(0, 1).all():
        raise ValueError(
            "percentage phải nằm trong [0, 1]"
        )

    return result


def build_representative_docs(
    documents: pd.DataFrame,
) -> pd.DataFrame:
    """Chọn top 5 bài từ train + val + test cho mỗi cụm."""
    regular = documents.loc[
        documents["topic_id"].ne(-1)
    ].copy()
    regular = regular.sort_values(
        ["topic_id", "topic_probability", "article_id"],
        ascending=[True, False, True],
        kind="stable",
    )

    selected = (
        regular.groupby("topic_id", sort=True)
        .head(TOP_N_REPRESENTATIVES)
        .copy()
    )
    selected["rank"] = (
        selected.groupby("topic_id").cumcount() + 1
    )

    result = selected[
        [
            "topic_id",
            "rank",
            "article_id",
            "title",
            "topic_probability",
            "split",
        ]
    ].reset_index(drop=True)

    if len(result) != (
        N_CLUSTERS * TOP_N_REPRESENTATIVES
    ):
        raise ValueError(
            "Không có đúng 5 bài đại diện mỗi cụm"
        )
    if result["article_id"].duplicated().any():
        raise ValueError(
            "Bài đại diện có article_id trùng"
        )

    return result


def build_config_used() -> dict:
    """Lưu cấu hình fit train và phạm vi artifact ba split."""
    m10 = read_yaml(EMBEDDING_CONFIG_PATH)
    m11 = read_yaml(HDBSCAN_CONFIG_PATH)

    if EMBEDDING_RUN not in m10["runs"]:
        raise ValueError(
            f"Thiếu embedding run {EMBEDDING_RUN}"
        )

    experiments = [
        item
        for item in m11["hdbscan"]["experiments"]
        if item["name"] == EXPERIMENT
    ]
    if len(experiments) != 1:
        raise ValueError(
            f"Cần đúng một experiment {EXPERIMENT}"
        )

    return {
        "dataset": m11["dataset"],
        "model_fit_split": "train",
        "artifact_splits": ["train", "val", "test"],
        "embedding": {
            "run_id": EMBEDDING_RUN,
            **m10["runs"][EMBEDDING_RUN],
            "normalize_embeddings": m10[
                "encoding"
            ]["normalize_embeddings"],
        },
        # Đây là UMAP 5D dùng để phân cụm M11.
        "umap": m11["umap"],
        "hdbscan": {
            "metric": m11["hdbscan"]["metric"],
            **experiments[0],
        },
        "keyword_extraction": {
            "method": "TF-IDF trên văn bản gom theo cụm",
            "source_splits": ["train", "val", "test"],
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
        "representative_selection": {
            "source_splits": ["train", "val", "test"],
            "method": (
                "Top 5 topic_probability mỗi cụm, "
                "hòa điểm theo article_id"
            ),
        },
        "projection_2d": {
            "method": "UMAP",
            "n_components": 2,
            "fit_split": "train",
            "transform_split": ["val", "test"],
            "purpose": "visualization_only",
        },
    }


def build_projection_2d(
    documents: pd.DataFrame,
    config_used: dict,
) -> pd.DataFrame:
    """Fit UMAP 2D train, transform val/test; ghép nhãn theo ID."""
    train_documents = documents.loc[
        documents["split"].eq("train")
    ]
    val_documents = documents.loc[
        documents["split"].eq("val")
    ]
    test_documents = documents.loc[
        documents["split"].eq("test")
    ]

    embedding_path = (
        ROOT / config_used["embedding"]["artifact_path"]
    )
    train_embeddings = load_embedding_artifact(
        embedding_path
    )
    train_ids = (
        train_embeddings["article_id"].astype(str).tolist()
    )

    if len(train_ids) != N_TRAIN:
        raise ValueError(
            "Embedding train không có 6.493 bài"
        )
    if len(set(train_ids)) != N_TRAIN:
        raise ValueError(
            "Embedding train có ID trùng"
        )
    if set(train_ids) != set(
        train_documents["article_id"]
    ):
        raise ValueError(
            "Embedding train không khớp document_topics"
        )

    train_vectors = np.stack(
        train_embeddings["embedding"].to_numpy()
    )
    dimensions = config_used["embedding"][
        "expected_dimensions"
    ]
    if train_vectors.shape != (N_TRAIN, dimensions):
        raise ValueError(
            "Kích thước embedding train không đúng"
        )
    if not np.isfinite(train_vectors).all():
        raise ValueError(
            "Embedding train chứa NaN/Infinity"
        )

    m10 = read_yaml(EMBEDDING_CONFIG_PATH)
    val_data = load_split_dataset(
        ROOT / m10["dataset"]["path"],
        ROOT / m10["dataset"]["split_manifest_path"],
        "val",
    ).reset_index(drop=True)

    val_ids = val_data["article_id"].astype(str).tolist()
    if len(val_ids) != N_VAL:
        raise ValueError(
            "Validation không có 1.391 bài"
        )
    if len(set(val_ids)) != N_VAL:
        raise ValueError(
            "Validation có ID trùng"
        )
    if set(val_ids) != set(
        val_documents["article_id"]
    ):
        raise ValueError(
            "Validation không khớp document_topics"
        )
    if set(train_ids) & set(val_ids):
        raise ValueError(
            "Train và validation có ID giao nhau"
        )

    # Chỉ tải MiniLM khi tạo phép chiếu.
    from sentence_transformers import SentenceTransformer

    print("Đang mã hóa validation bằng MiniLM...")
    encoder = SentenceTransformer(
        config_used["embedding"]["model_name"]
    )
    val_embeddings = create_embeddings(
        val_data,
        encoder,
        model_name=config_used["embedding"]["model_name"],
        data_version=m10["dataset"]["data_version"],
        content_words=config_used["embedding"]["content_words"],
        title_col=m10["dataset"]["title_col"],
        content_col=m10["dataset"]["content_col"],
        batch_size=m10["encoding"]["batch_size"],
    )

    if (
        val_embeddings["article_id"].astype(str).tolist()
        != val_ids
    ):
        raise ValueError(
            "Thứ tự ID validation đổi khi mã hóa"
        )

    val_vectors = np.stack(
        val_embeddings["embedding"].to_numpy()
    )
    if val_vectors.shape != (N_VAL, dimensions):
        raise ValueError(
            "Kích thước embedding validation không đúng"
        )
    if not np.isfinite(val_vectors).all():
        raise ValueError(
            "Embedding validation chứa NaN/Infinity"
        )

    test_data = load_split_dataset(
        ROOT / m10["dataset"]["path"],
        ROOT / m10["dataset"]["split_manifest_path"],
        "test",
    ).reset_index(drop=True)
    test_ids = test_data["article_id"].astype(str).tolist()
    if len(test_ids) != N_TEST or len(set(test_ids)) != N_TEST:
        raise ValueError("Test không có 1.390 ID duy nhất")
    if set(test_ids) != set(test_documents["article_id"]):
        raise ValueError("Test không khớp document_topics")
    if set(test_ids) & (set(train_ids) | set(val_ids)):
        raise ValueError("Test có ID giao với train/validation")
    print("Đang mã hóa test bằng MiniLM...")
    test_embeddings = create_embeddings(
        test_data,
        encoder,
        model_name=config_used["embedding"]["model_name"],
        data_version=m10["dataset"]["data_version"],
        content_words=config_used["embedding"]["content_words"],
        title_col=m10["dataset"]["title_col"],
        content_col=m10["dataset"]["content_col"],
        batch_size=m10["encoding"]["batch_size"],
    )
    if test_embeddings["article_id"].astype(str).tolist() != test_ids:
        raise ValueError("Thứ tự ID test đổi khi mã hóa")
    test_vectors = np.stack(test_embeddings["embedding"].to_numpy())
    if test_vectors.shape != (N_TEST, dimensions) or not np.isfinite(test_vectors).all():
        raise ValueError("Embedding test sai kích thước hoặc chứa NaN/Infinity")

    umap_params = dict(config_used["umap"])
    umap_params["n_components"] = 2
    reducer = UMAP(**umap_params)

    print("Đang fit UMAP 2D trên train...")
    train_xy = reducer.fit_transform(train_vectors)

    print("Đang transform validation lên cùng bản đồ...")
    val_xy = reducer.transform(val_vectors)
    print("Đang transform test lên cùng bản đồ...")
    test_xy = reducer.transform(test_vectors)

    if train_xy.shape != (N_TRAIN, 2):
        raise ValueError(
            "Tọa độ train không có hai chiều"
        )
    if val_xy.shape != (N_VAL, 2):
        raise ValueError(
            "Tọa độ validation không có hai chiều"
        )
    if test_xy.shape != (N_TEST, 2):
        raise ValueError("Tọa độ test không có hai chiều")
    if not np.isfinite(train_xy).all():
        raise ValueError(
            "Tọa độ train chứa NaN/Infinity"
        )
    if not np.isfinite(val_xy).all():
        raise ValueError(
            "Tọa độ validation chứa NaN/Infinity"
        )
    if not np.isfinite(test_xy).all():
        raise ValueError("Tọa độ test chứa NaN/Infinity")

    train_positions = pd.DataFrame(
        {
            "article_id": train_ids,
            "x": train_xy[:, 0],
            "y": train_xy[:, 1],
        }
    ).merge(
        train_documents[
            ["article_id", "split", "topic_id"]
        ],
        on="article_id",
        how="left",
        validate="one_to_one",
    )

    val_positions = pd.DataFrame(
        {
            "article_id": val_ids,
            "x": val_xy[:, 0],
            "y": val_xy[:, 1],
        }
    ).merge(
        val_documents[
            ["article_id", "split", "topic_id"]
        ],
        on="article_id",
        how="left",
        validate="one_to_one",
    )

    test_positions = pd.DataFrame(
        {"article_id": test_ids, "x": test_xy[:, 0], "y": test_xy[:, 1]}
    ).merge(
        test_documents[["article_id", "split", "topic_id"]],
        on="article_id", how="left", validate="one_to_one",
    )

    result = pd.concat(
        [train_positions, val_positions, test_positions],
        ignore_index=True,
    )
    if len(result) != N_TOTAL:
        raise ValueError(
            "projection_2d không có 9.274 điểm"
        )
    if result["article_id"].duplicated().any():
        raise ValueError(
            "projection_2d có ID trùng"
        )
    if result["topic_id"].isna().any():
        raise ValueError(
            "Có điểm 2D không ghép được topic_id"
        )

    result["topic_id"] = result["topic_id"].astype(int)
    return result[
        ["article_id", "split", "topic_id", "x", "y"]
    ]


def write_metadata(
    config_used: dict,
    config_hash: str,
    n_outliers: int,
    keywords: pd.DataFrame,
    test_metrics: dict,
) -> None:
    """Ghi config, summary, run metadata và manifest."""
    with CONFIG_USED_PATH.open(
        "w", encoding="utf-8"
    ) as file:
        yaml.safe_dump(
            config_used,
            file,
            allow_unicode=True,
            sort_keys=False,
        )

    runtime_seconds = None
    runtime_scope = None
    runtime_benchmark_path = None
    train_runtime_seconds = None
    val_runtime_seconds = None
    benchmark = None

    if BENCHMARK_PATH.is_file():
        benchmark = json.loads(
            BENCHMARK_PATH.read_text(encoding="utf-8")
        )

        if benchmark.get("run_id") != RUN_ID:
            raise ValueError("Benchmark runtime thuộc run khác")
        if benchmark.get("config_hash") != config_hash:
            raise ValueError("Benchmark runtime có config_hash khác")
        if benchmark.get("train_count") != N_TRAIN:
            raise ValueError("Benchmark runtime có số bài train khác")
        if benchmark.get("val_count") != N_VAL:
            raise ValueError("Benchmark runtime có số bài val khác")

        runtime_seconds = float(benchmark["runtime_seconds"])
        if (
            not np.isfinite(runtime_seconds)
            or runtime_seconds <= 0
        ):
            raise ValueError("runtime_seconds không hợp lệ")

        phases = benchmark["phases"]
        train_runtime_seconds = sum(
            float(phases[name])
            for name in (
                "train_representation_seconds",
                "umap_fit_seconds",
                "hdbscan_fit_seconds",
            )
        )
        val_runtime_seconds = sum(
            float(phases[name])
            for name in (
                "val_representation_seconds",
                "val_predict_seconds",
            )
        )

        if not np.isclose(
            train_runtime_seconds + val_runtime_seconds,
            runtime_seconds,
        ):
            raise ValueError("Runtime từng split không khớp tổng")

        runtime_scope = benchmark.get("runtime_scope")
        if not isinstance(runtime_scope, str) or not runtime_scope.strip():
            raise ValueError("Benchmark thiếu runtime_scope")

        runtime_benchmark_path = (
            BENCHMARK_PATH.relative_to(ROOT).as_posix()
        )
    top_keywords = keywords.loc[
        keywords["topic_id"].ne(-1)
        & keywords["rank"].le(TOP_N_KEYWORDS),
        "keyword",
    ].astype(str).str.strip()

    expected_count = N_CLUSTERS * TOP_N_KEYWORDS
    if len(top_keywords) != expected_count:
        raise ValueError("Không đủ top 10 từ khóa cho 101 cụm")
    if top_keywords.eq("").any():
        raise ValueError("Có từ khóa rỗng")

    topic_diversity = round(
        top_keywords.nunique() / len(top_keywords),
        4,
    )
    model_name = f"HDBSCAN {RUN_ID}"
    model_summary = {
        "model_name": model_name,
        "run_id": RUN_ID,
        "algorithm": "HDBSCAN",
        "representation": config_used["embedding"]["model_name"],
        "topic_count": N_CLUSTERS,
        "outlier_rate": n_outliers / N_TOTAL,
        "runtime_seconds": runtime_seconds,
        "runtime_scope": runtime_scope,
        "metrics": {
            "train_runtime_seconds": train_runtime_seconds,
            "val_runtime_seconds": val_runtime_seconds,
            "test_runtime_seconds": test_metrics["runtime_seconds"],
            "test_predict_seconds": test_metrics["test_predict_seconds"],
            "document_count": N_TOTAL,
            "train_count": N_TRAIN,
            "val_count": N_VAL,
            "test_count": N_TEST,
            "outlier_count": n_outliers,
            "train_outlier_count": N_TRAIN_OUTLIERS,
            "val_outlier_count": N_VAL_OUTLIERS,
            "test_outlier_count": test_metrics["test_outlier_count"],
            "train_assigned_count": (
                N_TRAIN - N_TRAIN_OUTLIERS
            ),
            "val_assigned_count": (
                N_VAL - N_VAL_OUTLIERS
            ),
            "test_assigned_count": N_TEST - test_metrics["test_outlier_count"],
            "test_ari": test_metrics["test_ari"],
            "test_nmi": test_metrics["test_nmi"],
            "test_outlier_rate": test_metrics["test_outlier_rate"],
            "test_coverage": test_metrics["test_coverage"],
            "topic_diversity": topic_diversity,
        },
        "config": config_used,
        "model_fit_split": "train",
        "artifact_splits": ["train", "val", "test"],
        "topic_summary_scope": "train+val+test",
        "keyword_source_splits": ["train", "val", "test"],
        "representative_source_splits": ["train", "val", "test"],
        "test_metric_scope": test_metrics["metric_scope"],
        "test_runtime_scope": test_metrics["runtime_scope"],
        "topic_probability_meaning": (
            "HDBSCAN membership_strength trong [0, 1]; "
            "không phải xác suất đã hiệu chuẩn"
        ),
        "topic_name_method": (
            "Ghép ba từ khóa TF-IDF đầu; "
            "tên tự động chưa duyệt thủ công"
        ),
        "validation_note": (
            "Từ khóa và bài đại diện của bundle được tạo "
            "sau khi chốt cấu hình, dùng train, val và test "
            "để hiển thị; không dùng để chọn lại model"
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
    runtime_summary = {
    "run_id": RUN_ID,
    "config_hash": config_hash,
    "measurement_note": (
        "Train/validation và test được đo ở hai lần chạy riêng. "
        "Không cộng các runtime này thành một phép đo toàn run."
    ),
    "train_validation": (
        None
        if benchmark is None
        else {
            "source_path": (
                BENCHMARK_PATH.relative_to(ROOT).as_posix()
            ),
            "runtime_seconds": runtime_seconds,
            "runtime_scope": runtime_scope,
            "train_runtime_seconds": train_runtime_seconds,
            "validation_runtime_seconds": val_runtime_seconds,
            "model_load_seconds": benchmark[
                "model_load_seconds"
            ],
            "representation_seconds": benchmark[
                "representation_seconds"
            ],
            "train_representation_seconds": benchmark[
                "phases"
            ]["train_representation_seconds"],
            "validation_representation_seconds": benchmark[
                "phases"
            ]["val_representation_seconds"],
            "umap_fit_seconds": benchmark[
                "phases"
            ]["umap_fit_seconds"],
            "hdbscan_fit_seconds": benchmark[
                "phases"
            ]["hdbscan_fit_seconds"],
            "validation_predict_seconds": benchmark[
                "validation_predict_seconds"
            ],
            "evaluation_seconds": benchmark[
                "evaluation_seconds"
            ],
            "evaluation_scope": benchmark.get(
                "evaluation_scope"
            ),
            "n_repeats": benchmark["n_repeats"],
        }
    ),
    "test": {
        "source_path": (
            TEST_METRICS_PATH.relative_to(ROOT).as_posix()
        ),
        "runtime_seconds": test_metrics[
            "runtime_seconds"
        ],
        "runtime_scope": test_metrics[
            "runtime_scope"
        ],
        "model_load_seconds": test_metrics[
            "model_load_seconds"
        ],
        "representation_seconds": test_metrics[
            "test_representation_seconds"
        ],
        "test_predict_seconds": test_metrics[
            "test_predict_seconds"
        ],
        "evaluation_seconds": test_metrics[
            "evaluation_seconds"
        ],
        "metric_scope": test_metrics[
            "metric_scope"
        ],
        "test_count": test_metrics[
            "test_count"
        ],
        "test_outlier_count": test_metrics[
            "test_outlier_count"
        ],
        "test_outlier_rate": test_metrics[
            "test_outlier_rate"
        ],
        "test_coverage": test_metrics[
            "test_coverage"
        ],
        "test_ari": test_metrics[
            "test_ari"
        ],
        "test_nmi": test_metrics[
            "test_nmi"
        ],
    },
}

    with RUNTIME_SUMMARY_PATH.open(
    "w",
    encoding="utf-8",
    ) as file:
        json.dump(
        runtime_summary,
        file,
        ensure_ascii=False,
        indent=2,
    )

    run_metadata = {
    "run_id": RUN_ID,
    "embedding_run": EMBEDDING_RUN,
    "experiment": EXPERIMENT,
    "fit_split": "train",
    "exported_splits": [
        "train",
        "val",
        "test",
    ],
    "config_hash": config_hash,
    "n_articles": N_TOTAL,
    "n_clusters": N_CLUSTERS,
    "n_outliers": n_outliers,
    "n_assigned": N_TOTAL - n_outliers,
    "outlier_rate": n_outliers / N_TOTAL,

    # runtime_seconds chỉ là benchmark train + validation.
    "runtime_seconds": runtime_seconds,
    "runtime_scope": runtime_scope,

    # Liên kết đến ba nguồn runtime.
    "runtime_benchmark_path": runtime_benchmark_path,
    "test_metrics_path": (
        TEST_METRICS_PATH.relative_to(ROOT).as_posix()
    ),
    "runtime_summary_path": (
        RUNTIME_SUMMARY_PATH.relative_to(ROOT).as_posix()
    ),

    # Phạm vi test được lưu riêng.
    "test_runtime_scope": test_metrics[
        "runtime_scope"
    ],
    "test_metric_scope": test_metrics[
        "metric_scope"
    ],

    "split_counts": {
        "train": N_TRAIN,
        "val": N_VAL,
        "test": N_TEST,
    },
    "split_outlier_counts": {
        "train": N_TRAIN_OUTLIERS,
        "val": N_VAL_OUTLIERS,
        "test": test_metrics[
            "test_outlier_count"
        ],
    },
    "source_files": {
        "train_assignments": (
            TRAIN_ASSIGNMENTS_PATH
            .relative_to(ROOT)
            .as_posix()
        ),
        "val_assignments": (
            VAL_ASSIGNMENTS_PATH
            .relative_to(ROOT)
            .as_posix()
        ),
        "test_assignments": (
            TEST_ASSIGNMENTS_PATH
            .relative_to(ROOT)
            .as_posix()
        ),
        "m11_summary": (
            M11_SUMMARY_PATH
            .relative_to(ROOT)
            .as_posix()
        ),
        "articles": (
            ARTICLES_PATH
            .relative_to(ROOT)
            .as_posix()
        ),
        "split_manifest": (
            SPLIT_MANIFEST_PATH
            .relative_to(ROOT)
            .as_posix()
        ),
        "embedding_artifact": config_used[
            "embedding"
        ]["artifact_path"],
        "runtime_benchmark": (
            None
            if runtime_benchmark_path is None
            else runtime_benchmark_path
        ),
        "test_metrics": (
            TEST_METRICS_PATH
            .relative_to(ROOT)
            .as_posix()
        ),
        "runtime_summary": (
            RUNTIME_SUMMARY_PATH
            .relative_to(ROOT)
            .as_posix()
        ),
    },
    "keyword_method": (
        "TF-IDF trên processed_title và 60 từ đầu "
        "processed_content của train + val + test; loại -1"
    ),
    "projection_2d_method": (
        "UMAP 2D fit train, transform val và test; "
        "topic_id lấy từ HDBSCAN M11 5D"
    ),
}

    with RUN_METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            run_metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )
    # Loader chung chỉ nhận các trường manifest sau.
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


def main() -> None:
    split_manifest = read_split_manifest()
    articles = read_articles()

    train, summary_row = select_train_assignments(
        split_manifest
    )
    train_topic_ids = set(
        train["cluster_id"].astype(int)
    )
    val = select_val_assignments(
        split_manifest,
        str(summary_row["config_hash"]),
        train_topic_ids,
    )
    test, test_metrics = select_test_assignments(
        split_manifest,
        str(summary_row["config_hash"]),
        train_topic_ids,
    )

    if (
        set(train["article_id"]) & set(val["article_id"])
        or set(test["article_id"]) &
        (set(train["article_id"]) | set(val["article_id"]))
    ):
        raise ValueError("Train, validation và test có ID giao nhau")

    # Ghép nội dung gốc cho cả ba split sau khi model đã được chốt.
    assignments = pd.concat(
        [train, val, test],
        ignore_index=True,
    )
    if len(assignments) != N_TOTAL:
        raise ValueError(
            "Không có đúng 9.274 assignments"
        )
    if assignments["article_id"].duplicated().any():
        raise ValueError(
            "Assignments train + val + test có ID trùng"
        )

    assignments = assignments.merge(
        articles,
        on="article_id",
        how="left",
        validate="one_to_one",
    )
    if assignments["title"].isna().any():
        raise ValueError(
            "Có bài không ghép được tiêu đề"
        )

    documents = build_document_topics(assignments)
    keywords = build_topic_keywords(assignments)
    topic_summary = build_topic_summary(
        documents,
        keywords,
    )
    representatives = build_representative_docs(documents)
    config_used = build_config_used()
    projection = build_projection_2d(
        documents,
        config_used,
    )

    regular_topics = set(
        documents.loc[
            documents["topic_id"].ne(-1),
            "topic_id",
        ]
    )
    if len(regular_topics) != N_CLUSTERS:
        raise ValueError(
            "Bundle không có đúng 101 cụm"
        )
    if set(keywords["topic_id"]) != regular_topics:
        raise ValueError(
            "Có cụm bị thiếu từ khóa"
        )
    if keywords.groupby("topic_id").size().ne(
        TOP_N_KEYWORDS
    ).any():
        raise ValueError(
            "Mỗi cụm phải có 10 từ khóa"
        )
    if len(topic_summary) != N_CLUSTERS + 1:
        raise ValueError(
            "topic_summary phải có 102 dòng"
        )

    n_outliers = int(documents["topic_id"].eq(-1).sum())
    if n_outliers != (
        N_TRAIN_OUTLIERS + N_VAL_OUTLIERS + test_metrics["test_outlier_count"]
    ):
        raise ValueError(
            "Tổng số outlier không khớp ba split"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    documents.to_csv(
        DOCUMENT_TOPICS_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    topic_summary.to_csv(
        TOPIC_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    keywords.to_csv(
        TOPIC_KEYWORDS_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    representatives.to_csv(
        REPRESENTATIVE_DOCS_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    projection.to_csv(
        PROJECTION_2D_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    write_metadata(
        config_used=config_used,
        config_hash=str(summary_row["config_hash"]),
        n_outliers=n_outliers,
        keywords=keywords,
        test_metrics=test_metrics,
    )

    bundle = load_artifact_bundle(OUTPUT_DIR)
    if len(bundle.document_topics) != N_TOTAL:
        raise ValueError(
            "Loader không đọc đủ 9.274 bài"
        )
    if int(
        bundle.topic_summary["document_count"].sum()
    ) != N_TOTAL:
        raise ValueError(
            "topic_summary không cộng đủ 9.274 bài"
        )
    if len(bundle.topic_keywords) != (
        N_CLUSTERS * TOP_N_KEYWORDS
    ):
        raise ValueError(
            "Loader không đọc đủ từ khóa"
        )

    for path in (
        DOCUMENT_TOPICS_PATH,
        TOPIC_SUMMARY_PATH,
        TOPIC_KEYWORDS_PATH,
        REPRESENTATIVE_DOCS_PATH,
        PROJECTION_2D_PATH,
        CONFIG_USED_PATH,
        MODEL_SUMMARY_PATH,
        RUN_METADATA_PATH,
        MANIFEST_PATH,
    ):
        print("Đã lưu:", path)

    print("Tổng số bài:", len(bundle.document_topics))
    print(
        "Theo split:",
        bundle.document_topics["split"]
        .value_counts()
        .to_dict(),
    )
    print("Số cụm:", N_CLUSTERS)
    print("Số bài -1:", n_outliers)
    print(
        "Tổng document_count:",
        int(
            bundle.topic_summary["document_count"].sum()
        ),
    )
    print("Số bài đại diện:", len(representatives))
    print("Số điểm 2D:", len(projection))
    print("Loader chung đã đọc thành công:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
