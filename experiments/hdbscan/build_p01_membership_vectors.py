"""Tạo full membership vector cho run HDBSCAN P01 đã khóa."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from hdbscan.prediction import (
    all_points_membership_vectors,
    approximate_predict,
    membership_vector,
)
from sentence_transformers import SentenceTransformer

from src.data.dataset_loader import load_split_dataset
from src.features.embeddings import (
    create_embeddings,
)


ROOT = Path(__file__).resolve().parents[2]

RUN_ID = "minilm_title_content_eom_15_5"
EMBEDDING_RUN = "minilm_title_content"
EXPERIMENT = "eom_15_5"

MODEL_DIR = ROOT / "artifacts/hdbscan/validation"
REDUCER_PATH = (
    MODEL_DIR / f"{EMBEDDING_RUN}_umap.joblib"
)
CLUSTERER_PATH = (
    MODEL_DIR / f"{RUN_ID}_hdbscan.joblib"
)

TRAIN_ASSIGNMENTS_PATH = (
    ROOT
    / "results/hdbscan/"
    "minilm_title_content_assignments.csv"
)
VAL_ASSIGNMENTS_PATH = (
    ROOT
    / "results/hdbscan/validation/"
    "minilm_title_content_eom_15_5_val_assignments.csv"
)
TEST_ASSIGNMENTS_PATH = (
    ROOT
    / "results/hdbscan/test/"
    "minilm_title_content_eom_15_5_test_assignments.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results/hdbscan/membership/"
    / RUN_ID
)
SUMMARY_PATH = OUTPUT_DIR / "membership_summary.csv"
VECTORS_PATH = OUTPUT_DIR / "membership_vectors.npz"

# Quy tắc heuristic cần ghi rõ trong metadata/tài liệu.
MULTI_TOPIC_GAP_MAX = 0.10


def read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def read_saved_labels(
    path: Path,
    article_ids: list[str],
    *,
    filter_run: bool = False,
) -> np.ndarray:
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"article_id": str},
    )

    if filter_run:
        frame = frame.loc[
            frame["embedding_run"].eq(EMBEDDING_RUN)
            & frame["experiment"].eq(EXPERIMENT)
        ].copy()

    if frame["article_id"].duplicated().any():
        raise ValueError(
            f"Assignments có article_id trùng: {path}"
        )

    if set(frame["article_id"]) != set(article_ids):
        raise ValueError(
            f"Assignments không khớp article_id: {path}"
        )

    return (
        frame.set_index("article_id")
        .loc[article_ids, "cluster_id"]
        .to_numpy(dtype=int)
    )


def get_membership_topic_ids(clusterer) -> np.ndarray:
    """Ánh xạ cột membership vector sang topic_id HDBSCAN."""
    selected_clusters = sorted(
        clusterer.condensed_tree_._select_clusters()
    )

    cluster_map = (
        clusterer.prediction_data_.cluster_map
    )

    topic_ids = np.asarray(
        [
            int(cluster_map[cluster])
            for cluster in selected_clusters
        ],
        dtype=int,
    )

    if len(topic_ids) != len(set(topic_ids)):
        raise ValueError(
            "Topic ID trong membership vector bị trùng"
        )

    return topic_ids


def summarize_memberships(
    article_ids: list[str],
    split: str,
    vectors: np.ndarray,
    topic_ids: np.ndarray,
) -> pd.DataFrame:
    """Lấy hai topic có membership lớn nhất."""
    if vectors.ndim != 2:
        raise ValueError(
            f"Membership {split} không phải ma trận 2 chiều"
        )
    if vectors.shape[0] != len(article_ids):
        raise ValueError(
            f"Membership {split} sai số bài"
        )
    if vectors.shape[1] != len(topic_ids):
        raise ValueError(
            f"Membership {split} sai số topic"
        )
    if vectors.shape[1] < 2:
        raise ValueError(
            "Cần ít nhất hai topic để tính top-2"
        )
    if not np.isfinite(vectors).all():
        raise ValueError(
            f"Membership {split} chứa NaN hoặc Infinity"
        )
    if np.any(vectors < 0) or np.any(vectors > 1):
        raise ValueError(
            f"Membership {split} nằm ngoài [0, 1]"
        )

    order = np.argsort(
        -vectors,
        axis=1,
        kind="stable",
    )

    row_indexes = np.arange(len(article_ids))
    first_indexes = order[:, 0]
    second_indexes = order[:, 1]

    first_strengths = vectors[
        row_indexes,
        first_indexes,
    ]
    second_strengths = vectors[
        row_indexes,
        second_indexes,
    ]
    gaps = first_strengths - second_strengths

    return pd.DataFrame(
        {
            "article_id": article_ids,
            "split": split,

            # Giữ hai trường này để giải thích chính xác gap.
            "soft_primary_topic_id": topic_ids[
                first_indexes
            ],
            "soft_primary_membership_strength": (
                first_strengths
            ),

            # Bốn trường leader yêu cầu.
            "second_topic_id": topic_ids[
                second_indexes
            ],
            "second_membership_strength": (
                second_strengths
            ),
            "top_two_share_gap": gaps,
            "is_multi_topic": (
                gaps <= MULTI_TOPIC_GAP_MAX
            ),
        }
    )


def encode_and_transform(
    frame: pd.DataFrame,
    encoder,
    reducer,
    embedding_config: dict,
    dataset_config: dict,
    encoding_config: dict,
) -> np.ndarray:
    embeddings = create_embeddings(
        frame,
        encoder,
        model_name=embedding_config["model_name"],
        data_version=dataset_config["data_version"],
        content_words=embedding_config[
            "content_words"
        ],
        title_col=dataset_config["title_col"],
        content_col=dataset_config["content_col"],
        batch_size=encoding_config["batch_size"],
    )

    if (
        embeddings["article_id"].astype(str).tolist()
        != frame["article_id"].astype(str).tolist()
    ):
        raise ValueError(
            "Thứ tự article_id bị thay đổi khi encode"
        )

    dense_vectors = np.stack(
        embeddings["embedding"].to_numpy()
    )

    reduced_vectors = reducer.transform(
        dense_vectors
    )

    if reduced_vectors.shape != (len(frame), 5):
        raise ValueError(
            "UMAP transform không tạo ma trận 5D"
        )

    return reduced_vectors


def main() -> None:
    embedding_config_all = read_yaml(
        ROOT / "configs/embeddings.yaml"
    )
    hdbscan_config = read_yaml(
        ROOT / "configs/hdbscan.yaml"
    )

    embedding_config = embedding_config_all[
        "runs"
    ][EMBEDDING_RUN]
    dataset_config = embedding_config_all["dataset"]
    encoding_config = embedding_config_all[
        "encoding"
    ]

    train = load_split_dataset(
        dataset_config["path"],
        dataset_config["split_manifest_path"],
        "train",
    ).reset_index(drop=True)

    val = load_split_dataset(
        dataset_config["path"],
        dataset_config["split_manifest_path"],
        "val",
    ).reset_index(drop=True)

    test = load_split_dataset(
        dataset_config["path"],
        dataset_config["split_manifest_path"],
        "test",
    ).reset_index(drop=True)

    if (len(train), len(val), len(test)) != (
        6493,
        1391,
        1390,
    ):
        raise ValueError(
            "Số bài train/val/test không đúng"
        )

    reducer = joblib.load(REDUCER_PATH)
    clusterer = joblib.load(CLUSTERER_PATH)

    if clusterer.prediction_data_ is None:
        raise ValueError(
            "HDBSCAN model không có prediction_data_"
        )

    topic_ids = get_membership_topic_ids(
        clusterer
    )

    if len(topic_ids) != 101:
        raise ValueError(
            "Full membership vector không có 101 topic"
        )

    # Train phải giữ đúng thứ tự lúc fit HDBSCAN.
    train_ids = (
        train["article_id"].astype(str).tolist()
    )
    train_labels = read_saved_labels(
        TRAIN_ASSIGNMENTS_PATH,
        train_ids,
        filter_run=True,
    )

    fitted_labels = np.asarray(
        clusterer.labels_,
        dtype=int,
    )

    if not np.array_equal(
        fitted_labels,
        train_labels,
    ):
        raise ValueError(
            "Model lưu sẵn không khớp assignments train"
        )

    train_vectors = np.asarray(
        all_points_membership_vectors(clusterer),
        dtype=float,
    )

    encoder = SentenceTransformer(
        embedding_config["model_name"]
    )

    val_5d = encode_and_transform(
        val,
        encoder,
        reducer,
        embedding_config,
        dataset_config,
        encoding_config,
    )
    test_5d = encode_and_transform(
        test,
        encoder,
        reducer,
        embedding_config,
        dataset_config,
        encoding_config,
    )

    val_labels, _ = approximate_predict(
        clusterer,
        val_5d,
    )
    test_labels, _ = approximate_predict(
        clusterer,
        test_5d,
    )

    expected_val_labels = read_saved_labels(
        VAL_ASSIGNMENTS_PATH,
        val["article_id"].astype(str).tolist(),
    )
    expected_test_labels = read_saved_labels(
        TEST_ASSIGNMENTS_PATH,
        test["article_id"].astype(str).tolist(),
    )

    if not np.array_equal(
        val_labels,
        expected_val_labels,
    ):
        raise ValueError(
            "Nhãn validation tạo lại không khớp"
        )

    if not np.array_equal(
        test_labels,
        expected_test_labels,
    ):
        raise ValueError(
            "Nhãn test tạo lại không khớp"
        )

    val_vectors = np.asarray(
        membership_vector(clusterer, val_5d),
        dtype=float,
    )
    test_vectors = np.asarray(
        membership_vector(clusterer, test_5d),
        dtype=float,
    )

    train_summary = summarize_memberships(
        train_ids,
        "train",
        train_vectors,
        topic_ids,
    )
    val_summary = summarize_memberships(
        val["article_id"].astype(str).tolist(),
        "val",
        val_vectors,
        topic_ids,
    )
    test_summary = summarize_memberships(
        test["article_id"].astype(str).tolist(),
        "test",
        test_vectors,
        topic_ids,
    )

    summary = pd.concat(
        [
            train_summary,
            val_summary,
            test_summary,
        ],
        ignore_index=True,
    )

    if len(summary) != 9274:
        raise ValueError(
            "Membership summary không có 9.274 bài"
        )
    if summary["article_id"].duplicated().any():
        raise ValueError(
            "Membership summary có article_id trùng"
        )

    all_vectors = np.vstack(
        [
            train_vectors,
            val_vectors,
            test_vectors,
        ]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    np.savez_compressed(
        VECTORS_PATH,
        article_ids=summary[
            "article_id"
        ].to_numpy(dtype=str),
        splits=summary[
            "split"
        ].to_numpy(dtype=str),
        topic_ids=topic_ids,
        membership_vectors=all_vectors.astype(
            np.float32
        ),
    )

    print(f"Đã lưu: {SUMMARY_PATH}")
    print(f"Đã lưu: {VECTORS_PATH}")
    print(
        "Số bài multi-topic:",
        int(summary["is_multi_topic"].sum()),
    )


if __name__ == "__main__":
    main()