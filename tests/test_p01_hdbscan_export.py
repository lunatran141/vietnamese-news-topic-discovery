"""Kiểm tra bundle P01 HDBSCAN cho train, validation và test."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.artifacts.loader import load_artifact_bundle
from src.data.dataset_loader import load_split_dataset


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = (
    ROOT
    / "artifacts/common/hdbscan/minilm_title_content_eom_15_5"
)
TRAIN_ASSIGNMENTS = (
    ROOT / "results/hdbscan/minilm_title_content_assignments.csv"
)
VAL_ASSIGNMENTS = (
    ROOT
    / "results/hdbscan/validation/"
    "minilm_title_content_eom_15_5_val_assignments.csv"
)
TEST_ASSIGNMENTS = (
    ROOT / "results/hdbscan/test/"
    "minilm_title_content_eom_15_5_test_assignments.csv"
)
TEST_METRICS = (
    ROOT / "results/hdbscan/test/"
    "minilm_title_content_eom_15_5_test_metrics.json"
)
EXPECTED_TOTAL = 9274

EXPECTED_FILES = {
    "manifest.json",
    "model_summary.json",
    "config_used.yaml",
    "run_metadata.json",
    "runtime_summary.json",
    "document_topics.csv",
    "topic_summary.csv",
    "topic_keywords.csv",
    "representative_docs.csv",
    "projection_2d.csv",
}

EXPECTED_SPLIT_COUNTS = {"train": 6493, "val": 1391, "test": 1390}


def expected_outliers() -> dict[str, int]:
    test_assignments = pd.read_csv(TEST_ASSIGNMENTS)
    return {
        "train": 1608,
        "val": 484,
        "test": int(test_assignments["cluster_id"].eq(-1).sum()),
    }


@pytest.fixture(scope="module")
def bundle():
    return load_artifact_bundle(ARTIFACT_DIR)


def _read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(
        ARTIFACT_DIR / name,
        dtype={"article_id": str},
    )


def _check_source_assignments(
    documents: pd.DataFrame,
    source_path: Path,
    split: str,
) -> None:
    source = pd.read_csv(
        source_path,
        dtype={"article_id": str},
    )
    source = source.loc[
        source["embedding_run"].eq("minilm_title_content")
        & source["experiment"].eq("eom_15_5"),
        ["article_id", "cluster_id", "membership_strength"],
    ].rename(
        columns={
            "membership_strength": "source_membership_strength",
        }
    )

    selected_documents = documents.loc[
        documents["split"].eq(split),
        [
            "article_id",
            "topic_id",
            "topic_probability",
            "membership_strength",
        ],
    ]

    comparison = selected_documents.merge(
        source,
        on="article_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    assert comparison["_merge"].eq("both").all()
    assert comparison["topic_id"].eq(
        comparison["cluster_id"]
    ).all()
    assert np.allclose(
        comparison["topic_probability"].to_numpy(dtype=float),
        comparison["source_membership_strength"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
    )
    assert np.allclose(
        comparison["membership_strength"].to_numpy(dtype=float),
        comparison["source_membership_strength"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
    )


def test_bundle_co_du_file_va_loader_doc_duoc(bundle):
    actual_files = {
        path.name
        for path in ARTIFACT_DIR.iterdir()
        if path.is_file()
    }
    assert EXPECTED_FILES.issubset(actual_files)
    assert "validation_document_topics.csv" not in actual_files

    manifest = bundle.manifest
    summary = bundle.model_summary

    assert manifest.schema_version == "1.0"
    assert manifest.run_id == "minilm_title_content_eom_15_5"
    assert manifest.model_name == summary.model_name
    assert summary.algorithm == "HDBSCAN"
    assert summary.topic_count == 101

    for relative_path in (
        manifest.model_summary_path,
        manifest.topic_summary_path,
        manifest.document_topics_path,
        manifest.topic_keywords_path,
        manifest.config_path,
    ):
        assert (ARTIFACT_DIR / relative_path).is_file()

    config = yaml.safe_load(
        (ARTIFACT_DIR / "config_used.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["model_fit_split"] == "train"
    assert config["artifact_splits"] == ["train", "val", "test"]
    assert config["embedding"]["run_id"] == "minilm_title_content"
    assert config["embedding"]["content_words"] == 60
    assert config["hdbscan"]["name"] == "eom_15_5"
    assert config["hdbscan"]["min_cluster_size"] == 15
    assert config["hdbscan"]["min_samples"] == 5
    assert config["projection_2d"]["fit_split"] == "train"
    assert config["projection_2d"]["transform_split"] == ["val", "test"]


def test_document_topics_chua_ca_ba_split(bundle):
    documents = bundle.document_topics

    required_columns = {
        "article_id",
        "split",
        "topic_id",
        "topic_probability",
        "membership_strength",
        "is_outlier",
        "title",
        "category_gold",
    }
    assert required_columns.issubset(documents.columns)
    assert len(documents) == EXPECTED_TOTAL
    assert documents["article_id"].notna().all()
    assert documents["article_id"].is_unique
    assert (
        documents["split"].value_counts().to_dict()
        == EXPECTED_SPLIT_COUNTS
    )

    assert documents["topic_probability"].notna().all()
    assert documents["topic_probability"].between(0, 1).all()

    # Kiểm tra trường gốc của HDBSCAN.
    assert documents["membership_strength"].notna().all()
    assert documents["membership_strength"].between(0, 1).all()

    # Hai cột phải chứa cùng một giá trị.
    assert np.allclose(
        documents["topic_probability"].to_numpy(dtype=float),
        documents["membership_strength"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
    )

    # Nhãn vàng phải tồn tại cho tất cả bài.
    assert documents["category_gold"].notna().all()
    assert (
        documents["category_gold"]
        .astype(str)
        .str.strip()
        .ne("")
        .all()
    )

    flags = (
        documents["is_outlier"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )
    assert flags.notna().all()
    assert flags.eq(documents["topic_id"].eq(-1)).all()

    outliers_by_split = (
        documents.loc[
            documents["topic_id"].eq(-1),
            "split",
        ]
        .value_counts()
        .to_dict()
    )
    assert outliers_by_split == expected_outliers()


def test_category_gold_khop_split_manifest(bundle):
    expected = pd.read_csv(
        ROOT / "data/splits/split_manifest.csv",
        usecols=["article_id", "split", "category_gold"],
        dtype={
            "article_id": str,
            "category_gold": str,
        },
    )
    actual = bundle.document_topics[
        ["article_id", "split", "category_gold"]
    ].copy()
    actual["article_id"] = actual["article_id"].astype(str)

    comparison = actual.merge(
        expected,
        on="article_id",
        how="outer",
        suffixes=("_artifact", "_manifest"),
        validate="one_to_one",
        indicator=True,
    )

    assert comparison["_merge"].eq("both").all()
    assert comparison["split_artifact"].eq(
        comparison["split_manifest"]
    ).all()
    assert comparison["category_gold_artifact"].eq(
        comparison["category_gold_manifest"]
    ).all()


def test_train_khop_assignments_m11(bundle):
    _check_source_assignments(
        bundle.document_topics,
        TRAIN_ASSIGNMENTS,
        "train",
    )


def test_validation_khop_assignments_da_predict(bundle):
    _check_source_assignments(
        bundle.document_topics,
        VAL_ASSIGNMENTS,
        "val",
    )


def test_test_khop_assignments_da_predict(bundle):
    _check_source_assignments(bundle.document_topics, TEST_ASSIGNMENTS, "test")
    test_metrics = json.loads(TEST_METRICS.read_text(encoding="utf-8"))
    test_rows = bundle.document_topics.loc[
        bundle.document_topics["split"].eq("test")
    ].copy()
    assert len(test_rows) == test_metrics["test_count"] == 1390
    assert int(test_rows["topic_id"].eq(-1).sum()) == (
        test_metrics["test_outlier_count"]
    )
    assert test_metrics["test_coverage"] == pytest.approx(
        1 - test_metrics["test_outlier_count"] / 1390
    )

    with (ROOT / "configs/hdbscan.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    dataset = config["dataset"]
    gold = load_split_dataset(
        ROOT / dataset["path"],
        ROOT / dataset["split_manifest_path"],
        "test",
    )[["article_id", "category_gold"]].copy()

    gold["article_id"] = gold["article_id"].astype(str)
    test_rows["article_id"] = test_rows["article_id"].astype(str)

    joined = test_rows.merge(
        gold,
        on="article_id",
        how="inner",
        suffixes=("_artifact", "_dataset"),
        validate="one_to_one",
    )

    assert len(joined) == len(test_rows)
    assert joined["category_gold_artifact"].eq(
        joined["category_gold_dataset"]
    ).all()

    assert test_metrics["test_ari"] == pytest.approx(
        adjusted_rand_score(
            joined["category_gold_artifact"],
            joined["topic_id"],
        )
    )
    assert test_metrics["test_nmi"] == pytest.approx(
        normalized_mutual_info_score(
            joined["category_gold_artifact"],
            joined["topic_id"],
        )
    )


def test_topic_summary_va_keywords_khop_document_topics(bundle):
    documents = bundle.document_topics
    topics = bundle.topic_summary
    keywords = bundle.topic_keywords

    assert len(topics) == 102  # 101 cụm và dòng outlier -1
    assert topics["topic_id"].is_unique
    assert set(topics["topic_id"]) == set(documents["topic_id"])
    assert int(topics["document_count"].sum()) == EXPECTED_TOTAL
    assert topics["percentage"].between(0, 1).all()

    counts = documents["topic_id"].value_counts().to_dict()
    for row in topics.itertuples(index=False):
        assert row.document_count == counts[row.topic_id]
        assert row.percentage == pytest.approx(
            row.document_count / len(documents)
        )

    outlier = topics.loc[topics["topic_id"].eq(-1)]
    assert len(outlier) == 1
    assert int(outlier.iloc[0]["document_count"]) == sum(expected_outliers().values())

    regular_topic_ids = set(topics["topic_id"]) - {-1}
    assert len(regular_topic_ids) == 101
    assert set(keywords["topic_id"]) == regular_topic_ids
    assert len(keywords) == 1010
    assert keywords["keyword"].notna().all()
    assert keywords["keyword"].astype(str).str.strip().ne("").all()
    assert np.isfinite(
        keywords["weight"].to_numpy(dtype=float)
    ).all()
    assert keywords["weight"].ge(0).all()

    for _, group in keywords.groupby("topic_id"):
        assert sorted(group["rank"].tolist()) == list(range(1, 11))

    top_keywords = keywords.loc[
        keywords["topic_id"].ne(-1) & keywords["rank"].le(10),
        "keyword",
    ].astype(str).str.strip()
    expected_diversity = round(
        top_keywords.nunique() / len(top_keywords),
        4,
    )
    assert bundle.model_summary.metrics["topic_diversity"] == (
        pytest.approx(expected_diversity)
    )

def test_run_metadata_va_model_summary_khop_bundle(bundle):
    metadata = json.loads(
        (ARTIFACT_DIR / "run_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    summary = bundle.model_summary

    assert metadata["run_id"] == bundle.manifest.run_id
    assert metadata["embedding_run"] == "minilm_title_content"
    assert metadata["experiment"] == "eom_15_5"
    assert metadata["fit_split"] == "train"
    assert metadata["exported_splits"] == ["train", "val", "test"]
    assert metadata["n_articles"] == EXPECTED_TOTAL
    assert metadata["n_clusters"] == 101
    total_outliers = sum(expected_outliers().values())
    assert metadata["n_outliers"] == total_outliers
    assert metadata["n_assigned"] == EXPECTED_TOTAL - total_outliers
    assert metadata["outlier_rate"] == pytest.approx(total_outliers / EXPECTED_TOTAL)
    assert metadata["split_counts"] == EXPECTED_SPLIT_COUNTS
    assert (
        metadata["split_outlier_counts"]
        == expected_outliers()
    )
    assert metadata["document_topics_columns"] == [
        "article_id",
        "split",
        "topic_id",
        "topic_probability",
        "membership_strength",
        "is_outlier",
        "title",
        "category_gold",
    ]
    assert "HDBSCAN" in metadata["membership_strength_meaning"]
    assert "không dùng để fit" in metadata["category_gold_usage"]
    assert metadata["stability_summary_path"] == (
        "results/hdbscan/stability/"
        "minilm_title_content_eom_15_5_seeds42_51/"
        "stability_summary.csv"
    )
    assert "10 UMAP seeds 42-51" in metadata["stability_scope"]
    assert "keyword Jaccard" in metadata["stability_scope"]

    metrics = summary.metrics
    assert metrics["document_count"] == EXPECTED_TOTAL
    assert metrics["train_count"] == 6493
    assert metrics["val_count"] == 1391
    assert metrics["test_count"] == 1390
    assert metrics["outlier_count"] == total_outliers
    assert metrics["train_outlier_count"] == 1608
    assert metrics["val_outlier_count"] == 484
    assert metrics["test_outlier_count"] == expected_outliers()["test"]
    assert summary.outlier_rate == pytest.approx(total_outliers / EXPECTED_TOTAL)

    for relative_path in metadata["source_files"].values():
        assert (ROOT / relative_path).is_file()

    m11_summary = pd.read_csv(
        ROOT / metadata["source_files"]["m11_summary"]
    )
    selected = m11_summary.loc[
        m11_summary["embedding_run"].eq("minilm_title_content")
        & m11_summary["experiment"].eq("eom_15_5")
    ]
    assert len(selected) == 1
    assert metadata["config_hash"] == selected.iloc[0]["config_hash"]


def test_representative_docs_khop_document_topics(bundle):
    representatives = _read_csv("representative_docs.csv")
    documents = bundle.document_topics.copy()
    documents["article_id"] = documents["article_id"].astype(str)

    required_columns = {
        "topic_id",
        "rank",
        "article_id",
        "title",
        "topic_probability",
        "split",
    }
    assert required_columns.issubset(representatives.columns)
    assert len(representatives) == 505
    assert representatives["article_id"].is_unique
    assert set(representatives["split"]).issubset({"train", "val", "test"})
    assert not representatives["topic_id"].eq(-1).any()
    assert representatives.groupby("topic_id").size().eq(5).all()

    expected = documents.loc[
        documents["topic_id"].ne(-1)
    ].sort_values(
        ["topic_id", "topic_probability", "article_id"],
        ascending=[True, False, True],
        kind="stable",
    )
    expected = expected.groupby(
        "topic_id",
        sort=True,
    ).head(5).copy()
    expected["rank"] = (
        expected.groupby("topic_id").cumcount() + 1
    )

    comparison = representatives.merge(
        expected[
            [
                "topic_id",
                "rank",
                "article_id",
                "split",
                "topic_probability",
            ]
        ],
        on=["topic_id", "rank"],
        how="outer",
        suffixes=("_actual", "_expected"),
        validate="one_to_one",
        indicator=True,
    )

    assert comparison["_merge"].eq("both").all()
    assert comparison["article_id_actual"].eq(
        comparison["article_id_expected"]
    ).all()
    assert comparison["split_actual"].eq(
        comparison["split_expected"]
    ).all()
    assert np.allclose(
        comparison["topic_probability_actual"].to_numpy(
            dtype=float
        ),
        comparison["topic_probability_expected"].to_numpy(
            dtype=float
        ),
        rtol=0,
        atol=1e-12,
    )


def test_projection_2d_khop_document_topics_ba_split(bundle):
    projection = _read_csv("projection_2d.csv")
    documents = bundle.document_topics[
        ["article_id", "split", "topic_id"]
    ].copy()
    documents["article_id"] = documents["article_id"].astype(str)

    required_columns = {
        "article_id",
        "split",
        "topic_id",
        "x",
        "y",
    }
    assert required_columns.issubset(projection.columns)
    assert len(projection) == EXPECTED_TOTAL
    assert projection["article_id"].is_unique
    assert (
        projection["split"].value_counts().to_dict()
        == EXPECTED_SPLIT_COUNTS
    )
    assert np.isfinite(
        projection[["x", "y"]].to_numpy(dtype=float)
    ).all()

    comparison = projection.merge(
        documents,
        on="article_id",
        how="outer",
        suffixes=("_projection", "_documents"),
        validate="one_to_one",
        indicator=True,
    )
    assert comparison["_merge"].eq("both").all()
    assert comparison["split_projection"].eq(
        comparison["split_documents"]
    ).all()
    assert comparison["topic_id_projection"].eq(
        comparison["topic_id_documents"]
    ).all()

def test_runtime_benchmark_khop_bundle(bundle):
    benchmark_path = ROOT / "results/hdbscan/benchmark_p01_runtime.json"
    assert benchmark_path.is_file()

    benchmark = json.loads(
        benchmark_path.read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (ARTIFACT_DIR / "run_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    assert benchmark["run_id"] == bundle.model_summary.run_id
    assert benchmark["config_hash"] == metadata["config_hash"]
    assert benchmark["train_count"] == 6493
    assert benchmark["val_count"] == 1391
    assert benchmark["n_repeats"] == 1

    phases = benchmark["phases"]
    assert set(phases) == {
        "train_representation_seconds",
        "umap_fit_seconds",
        "hdbscan_fit_seconds",
        "val_representation_seconds",
        "val_predict_seconds",
    }
    assert all(
        np.isfinite(value) and value > 0
        for value in phases.values()
    )

    expected_train = sum(
        phases[name]
        for name in (
            "train_representation_seconds",
            "umap_fit_seconds",
            "hdbscan_fit_seconds",
        )
    )
    expected_val = sum(
        phases[name]
        for name in (
            "val_representation_seconds",
            "val_predict_seconds",
        )
    )
    total = benchmark["runtime_seconds"]

    assert total == pytest.approx(sum(phases.values()))
    assert expected_train + expected_val == pytest.approx(total)

    metrics = bundle.model_summary.metrics
    assert metrics["train_runtime_seconds"] == pytest.approx(
        expected_train
    )
    assert metrics["val_runtime_seconds"] == pytest.approx(
        expected_val
    )

    assert bundle.model_summary.runtime_seconds == pytest.approx(total)
    assert metadata["runtime_seconds"] == pytest.approx(total)
    assert metadata["runtime_scope"] == benchmark["runtime_scope"]
    assert metadata["runtime_benchmark_path"] == (
        "results/hdbscan/benchmark_p01_runtime.json"
    )


def test_runtime_summary_lien_ket_hai_lan_do(bundle):
    path = ARTIFACT_DIR / "runtime_summary.json"
    assert path.is_file()

    summary = json.loads(path.read_text(encoding="utf-8"))
    benchmark = json.loads(
        (ROOT / "results/hdbscan/benchmark_p01_runtime.json")
        .read_text(encoding="utf-8")
    )
    test_metrics = json.loads(
        TEST_METRICS.read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (ARTIFACT_DIR / "run_metadata.json")
        .read_text(encoding="utf-8")
    )

    assert summary["run_id"] == bundle.model_summary.run_id
    assert summary["config_hash"] == benchmark["config_hash"]
    assert summary["config_hash"] == test_metrics["config_hash"]
    assert metadata["runtime_summary_path"] == (
        "artifacts/common/hdbscan/"
        "minilm_title_content_eom_15_5/runtime_summary.json"
    )

    train_val = summary["train_validation"]
    test = summary["test"]
    assert train_val is not None
    assert train_val["runtime_seconds"] == pytest.approx(
        benchmark["runtime_seconds"]
    )
    assert test["runtime_seconds"] == pytest.approx(
        test_metrics["runtime_seconds"]
    )
    assert test["test_predict_seconds"] == pytest.approx(
        test_metrics["test_predict_seconds"]
    )
    assert (ROOT / train_val["source_path"]).is_file()
    assert (ROOT / test["source_path"]).is_file()
    assert "hai lần chạy riêng" in summary["measurement_note"]
