"""Kiểm tra bundle P01 HDBSCAN cho train và validation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.artifacts.loader import load_artifact_bundle


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

EXPECTED_FILES = {
    "manifest.json",
    "model_summary.json",
    "config_used.yaml",
    "run_metadata.json",
    "document_topics.csv",
    "topic_summary.csv",
    "topic_keywords.csv",
    "representative_docs.csv",
    "projection_2d.csv",
}

EXPECTED_SPLIT_COUNTS = {"train": 6493, "val": 1391}
EXPECTED_SPLIT_OUTLIERS = {"train": 1608, "val": 484}


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
    ]

    selected_documents = documents.loc[
        documents["split"].eq(split),
        ["article_id", "topic_id", "topic_probability"],
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
        comparison["membership_strength"].to_numpy(dtype=float),
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
    assert config["artifact_splits"] == ["train", "val"]
    assert config["embedding"]["run_id"] == "minilm_title_content"
    assert config["embedding"]["content_words"] == 60
    assert config["hdbscan"]["name"] == "eom_15_5"
    assert config["hdbscan"]["min_cluster_size"] == 15
    assert config["hdbscan"]["min_samples"] == 5
    assert config["projection_2d"]["fit_split"] == "train"
    assert config["projection_2d"]["transform_split"] == "val"


def test_document_topics_chua_ca_train_va_validation(bundle):
    documents = bundle.document_topics

    required_columns = {
        "article_id",
        "split",
        "topic_id",
        "topic_probability",
        "is_outlier",
        "title",
    }
    assert required_columns.issubset(documents.columns)
    assert len(documents) == 7884
    assert documents["article_id"].notna().all()
    assert documents["article_id"].is_unique
    assert (
        documents["split"].value_counts().to_dict()
        == EXPECTED_SPLIT_COUNTS
    )
    assert documents["topic_probability"].notna().all()
    assert documents["topic_probability"].between(0, 1).all()

    flags = (
        documents["is_outlier"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )
    assert flags.notna().all()
    assert flags.eq(documents["topic_id"].eq(-1)).all()

    outliers_by_split = (
        documents.loc[documents["topic_id"].eq(-1), "split"]
        .value_counts()
        .to_dict()
    )
    assert outliers_by_split == EXPECTED_SPLIT_OUTLIERS


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


def test_topic_summary_va_keywords_khop_document_topics(bundle):
    documents = bundle.document_topics
    topics = bundle.topic_summary
    keywords = bundle.topic_keywords

    assert len(topics) == 102  # 101 cụm và dòng outlier -1
    assert topics["topic_id"].is_unique
    assert set(topics["topic_id"]) == set(documents["topic_id"])
    assert int(topics["document_count"].sum()) == 7884
    assert topics["percentage"].between(0, 1).all()

    counts = documents["topic_id"].value_counts().to_dict()
    for row in topics.itertuples(index=False):
        assert row.document_count == counts[row.topic_id]
        assert row.percentage == pytest.approx(
            row.document_count / len(documents)
        )

    outlier = topics.loc[topics["topic_id"].eq(-1)]
    assert len(outlier) == 1
    assert int(outlier.iloc[0]["document_count"]) == 2092

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
    assert metadata["exported_splits"] == ["train", "val"]
    assert metadata["n_articles"] == 7884
    assert metadata["n_clusters"] == 101
    assert metadata["n_outliers"] == 2092
    assert metadata["n_assigned"] == 5792
    assert metadata["outlier_rate"] == pytest.approx(2092 / 7884)
    assert metadata["split_counts"] == EXPECTED_SPLIT_COUNTS
    assert (
        metadata["split_outlier_counts"]
        == EXPECTED_SPLIT_OUTLIERS
    )

    metrics = summary.metrics
    assert metrics["document_count"] == 7884
    assert metrics["train_count"] == 6493
    assert metrics["val_count"] == 1391
    assert metrics["outlier_count"] == 2092
    assert metrics["train_outlier_count"] == 1608
    assert metrics["val_outlier_count"] == 484
    assert summary.outlier_rate == pytest.approx(2092 / 7884)

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
    assert set(representatives["split"]).issubset({"train", "val"})
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


def test_projection_2d_khop_document_topics_hai_split(bundle):
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
    assert len(projection) == 7884
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