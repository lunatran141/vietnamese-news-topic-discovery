
"""Tests for P01 LDA common artifact export."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.artifacts.adapters import export_lda_artifacts
from src.artifacts.loader import load_artifact_bundle


COMMON_DIR = Path(
    "artifacts/common/lda/lda_k15_seed42"
)


def test_export_function_is_available():
    assert callable(export_lda_artifacts)


def test_common_loader_is_available():
    assert callable(load_artifact_bundle)


def test_export_rejects_existing_directory(tmp_path):
    output = tmp_path / "already_exists"
    output.mkdir()

    with pytest.raises(FileExistsError):
        export_lda_artifacts(output_dir=output)


def test_common_artifact_files_exist():
    required = {
        "manifest.json",
        "model_summary.json",
        "topic_summary.csv",
        "document_topics.csv",
        "topic_keywords.csv",
        "config_used.yaml",
    }

    actual = {
        path.name
        for path in COMMON_DIR.iterdir()
        if path.is_file()
    }

    assert required.issubset(actual)


def test_common_artifact_loads():
    bundle = load_artifact_bundle(COMMON_DIR)

    assert bundle.model_summary.algorithm == "LDA"
    assert bundle.model_summary.run_id == "lda_k15_seed42"
    assert bundle.model_summary.topic_count == 15


def test_document_assignments():
    bundle = load_artifact_bundle(COMMON_DIR)
    docs = bundle.document_topics

    assert len(docs) == 9274
    assert docs["article_id"].nunique() == 9274
    assert docs["article_id"].is_unique
    assert docs["topic_id"].between(0, 14).all()

    probs = pd.to_numeric(
        docs["topic_probability"],
        errors="coerce",
    )

    assert probs.notna().all()
    assert probs.between(0, 1).all()
    assert (~docs["is_outlier"].astype(str).str.lower().eq(
        "true"
    )).all()


def test_split_counts():
    bundle = load_artifact_bundle(COMMON_DIR)
    counts = bundle.document_topics["split"].value_counts()

    assert counts["train"] == 6493
    assert counts["val"] == 1391
    assert counts["test"] == 1390


def test_topic_summary():
    bundle = load_artifact_bundle(COMMON_DIR)

    summary = bundle.topic_summary

    assert len(summary) == 15
    assert set(summary["topic_id"]) == set(range(15))
    assert summary["document_count"].sum() == 9274
    assert summary["percentage"].sum() == pytest.approx(1.0)


def test_topic_keywords():
    bundle = load_artifact_bundle(COMMON_DIR)

    keywords = bundle.topic_keywords

    assert len(keywords) == 150
    assert set(keywords["topic_id"]) == set(range(15))

    counts = keywords.groupby("topic_id").size()
    assert (counts == 10).all()
    assert keywords["rank"].between(1, 10).all()


def test_manifest_consistency():
    manifest = json.loads(
        (COMMON_DIR / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    summary = json.loads(
        (COMMON_DIR / "model_summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["schema_version"] == "1.0"
    assert manifest["run_id"] == summary["run_id"]
    assert manifest["model_name"] == summary["model_name"]
    assert summary["topic_count"] == 15
    assert summary["outlier_rate"] == 0.0


def test_export_rejects_missing_source(tmp_path):
    source = tmp_path / "missing_model"
    output = tmp_path / "new_output"

    with pytest.raises(FileNotFoundError):
        export_lda_artifacts(
            source_dir=source,
            output_dir=output,
        )

    assert not output.exists()


def test_export_missing_metadata_does_not_create_output(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"

    with pytest.raises(FileNotFoundError):
        export_lda_artifacts(
            source_dir=source,
            output_dir=output,
        )

    assert not output.exists()


def test_export_does_not_overwrite_existing_files(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()

    marker = output / "marker.txt"
    marker.write_text("KEEP ORIGINAL", encoding="utf-8")

    with pytest.raises(FileExistsError):
        export_lda_artifacts(output_dir=output)

    assert marker.read_text(encoding="utf-8") == "KEEP ORIGINAL"
