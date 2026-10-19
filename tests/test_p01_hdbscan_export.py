"""Kiểm tra artifact P01 HDBSCAN đã xuất."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
import json

from src.artifacts.loader import load_artifact_bundle


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = (
    ROOT
    / "artifacts/common/hdbscan/minilm_title_content_eom_15_5"
)
ASSIGNMENTS_PATH = (
    ROOT / "results/hdbscan/minilm_title_content_assignments.csv"
)

EXPECTED_FILES = {
    "manifest.json",
    "model_summary.json",
    "config_used.yaml",
    "topic_summary.csv",
    "document_topics.csv",
    "topic_keywords.csv",
}


@pytest.fixture(scope="module")
def bundle():
    """Đọc bundle một lần cho các phép kiểm tra."""
    return load_artifact_bundle(ARTIFACT_DIR)


def test_du_sau_file_va_loader_doc_duoc(bundle):
    actual_files = {
        path.name for path in ARTIFACT_DIR.iterdir() if path.is_file()
    }
    assert EXPECTED_FILES.issubset(actual_files)

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

    with (ARTIFACT_DIR / "config_used.yaml").open(
        encoding="utf-8"
    ) as file:
        config = yaml.safe_load(file)

    assert config["embedding"]["run_id"] == "minilm_title_content"
    assert config["embedding"]["content_words"] == 60
    assert config["hdbscan"]["name"] == "eom_15_5"
    assert config["hdbscan"]["min_cluster_size"] == 15
    assert config["hdbscan"]["min_samples"] == 5


def test_bai_va_outlier_khop_assignments_m11(bundle):
    documents = bundle.document_topics
    assert len(documents) == 6493
    assert documents["article_id"].notna().all()
    assert documents["article_id"].is_unique
    assert documents["split"].eq("train").all()
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
    assert int(flags.sum()) == 1608

    assignments = pd.read_csv(
        ASSIGNMENTS_PATH,
        dtype={"article_id": str},
    )
    selected = assignments.loc[
        assignments["embedding_run"].eq("minilm_title_content")
        & assignments["experiment"].eq("eom_15_5"),
        ["article_id", "cluster_id", "membership_strength"],
    ]

    comparison = documents.merge(
        selected,
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
        comparison["topic_probability"].to_numpy(),
        comparison["membership_strength"].to_numpy(),
        rtol=0,
        atol=1e-12,
    )

    assert bundle.model_summary.outlier_rate == pytest.approx(
        1608 / 6493
    )


def test_topic_summary_va_keywords_khop_bai(bundle):
    documents = bundle.document_topics
    topics = bundle.topic_summary
    keywords = bundle.topic_keywords

    assert len(topics) == 102  # 101 cụm + dòng -1
    assert topics["topic_id"].is_unique
    assert set(topics["topic_id"]) == set(documents["topic_id"])
    assert int(topics["document_count"].sum()) == 6493
    assert topics["percentage"].between(0, 1).all()
    assert topics["topic_name"].notna().all()

    counts = documents["topic_id"].value_counts().to_dict()
    for row in topics.itertuples(index=False):
        assert row.document_count == counts[row.topic_id]
        assert row.percentage == pytest.approx(
            row.document_count / len(documents)
        )

    outlier_row = topics.loc[topics["topic_id"].eq(-1)]
    assert len(outlier_row) == 1
    assert int(outlier_row.iloc[0]["document_count"]) == 1608

    regular_topic_ids = set(topics["topic_id"]) - {-1}
    assert len(regular_topic_ids) == 101
    assert set(keywords["topic_id"]) == regular_topic_ids
    assert len(keywords) == 1010
    assert keywords["keyword"].notna().all()
    assert keywords["keyword"].astype(str).str.strip().ne("").all()
    assert np.isfinite(keywords["weight"]).all()
    assert keywords["weight"].ge(0).all()

    for _, group in keywords.groupby("topic_id"):
        assert sorted(group["rank"].tolist()) == list(range(1, 11))

def test_run_metadata_khop_m11(bundle):
    path = ARTIFACT_DIR / "run_metadata.json"
    assert path.is_file()

    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert metadata["run_id"] == bundle.manifest.run_id
    assert metadata["embedding_run"] == "minilm_title_content"
    assert metadata["experiment"] == "eom_15_5"
    assert metadata["split"] == "train"
    assert metadata["n_articles"] == 6493
    assert metadata["n_clusters"] == 101
    assert metadata["n_outliers"] == 1608
    assert metadata["n_assigned"] == 4885
    assert metadata["outlier_rate"] == pytest.approx(1608 / 6493)

    source = pd.read_csv(
        ROOT / metadata["source_files"]["m11_summary"]
    )
    selected = source.loc[
        source["embedding_run"].eq("minilm_title_content")
        & source["experiment"].eq("eom_15_5")
    ]
    assert len(selected) == 1
    assert metadata["config_hash"] == selected.iloc[0]["config_hash"]

    for relative_path in metadata["source_files"].values():
        assert (ROOT / relative_path).is_file()