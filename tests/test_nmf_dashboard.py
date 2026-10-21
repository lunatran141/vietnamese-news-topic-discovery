"""Kiểm tra split độc lập, trường giao diện và phân biệt gap với đa chủ đề."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from app.data_service import filter_documents, get_article_distribution, get_split_metrics
from experiments.nmf.prepare_dashboard import prepare_dashboard
from src.artifacts.adapters import export_nmf_artifacts
from tests.test_artifact_nmf_adapter import _write_nmf_source


def test_dashboard_artifacts_have_top_two_for_all_splits(tmp_path):
    source, results, output = tmp_path / "source", tmp_path / "results", tmp_path / "common"
    _write_nmf_source(source, results)
    bundle = export_nmf_artifacts(source, output, results)
    docs = bundle.document_topics
    assert docs["second_topic_share"].notna().all()
    assert docs["topic_entropy"].notna().all()
    assert docs["topic_probability"].isna().all()
    assert docs.loc[docs["article_id"].eq("A1"), "is_multi_topic"].item()
    # A1 có topic phụ 0.2 nhưng gap 0.6: đa chủ đề không đồng nghĩa gap nhỏ.
    assert filter_documents(bundle, "train", multi_topic_only=True)["article_id"].tolist() == ["A1"]
    assert filter_documents(bundle, "train", gap_max=0.10).empty
    np.testing.assert_allclose(get_article_distribution(bundle, "A1")["topic_share"], [0.8, 0.2])
    assert get_article_distribution(bundle, "missing").empty


def test_fixed_k_run_keeps_validation_and_test_independent(tmp_path):
    rows = []
    for split, size in [("train", 8), ("val", 4), ("test", 4)]:
        for i in range(size):
            topic = i % 2
            terms = "business finance market" if topic == 0 else "football sport match"
            suffix = " valonly" if split == "val" else " testonly" if split == "test" else ""
            rows.append(dict(article_id=f"{split}_{i}", split=split, title=f"Article {i}",
                             text=terms + suffix, source="demo", published_at="2026-01-01", category_gold=str(topic)))
    dataset = pd.DataFrame(rows)
    data_path = tmp_path / "data.csv"
    dataset.to_csv(data_path, index=False)
    config = {
        "data": {"text_column": "text", "id_column": "article_id", "gold_column": "category_gold",
                 "dataset_version": "v1.0", "preprocessing_variant": "demo", "dataset_path": str(data_path)},
        "representation": {"ngram_range": [1, 1], "min_df": 1, "max_df": 1.0,
                           "max_features": 100, "sublinear_tf": True, "token_pattern": r"(?u)\b\w\w+\b"},
        "model": {"random_seed": 42, "init": "nndsvda", "solver": "cd", "beta_loss": "frobenius",
                  "max_iter": 100, "tolerance": 0.0001},
        "topic_selection": {"selected_n_topics": 2, "candidates": [2], "coherence_top_n": 3},
        "analysis": {"secondary_topic_min_share": 0.2, "max_topics_per_document": 3,
                     "top_words": 3, "representative_documents_per_topic": 2,
                     "preview_characters": 100, "multi_topic_cases_limit": 5},
        "outputs": {"artifact_root": str(tmp_path / "artifacts"), "result_root": str(tmp_path / "results")},
    }
    source, results = prepare_dashboard(config, dataset, "independent")
    runtime = json.loads((results / "runtime.json").read_text())
    assert runtime["runtime_seconds"] == pytest.approx(sum(runtime[name] for name in (
        "representation_seconds", "fit_seconds", "validation_transform_seconds", "test_transform_seconds", "evaluation_seconds")))
    assert all(runtime[name] >= 0 for name in runtime if name.endswith("_seconds"))
    val = json.loads((results / "validation_metrics.json").read_text())
    test = json.loads((results / "test_metrics.json").read_text())
    assert val["fit_split"] == "train"
    assert test["fit_split"] == "train+val"
    pipeline = joblib.load(source / "model/nmf_pipeline.joblib")
    assert "testonly" not in pipeline["vectorizer"].vocabulary_
    assert "valonly" in pipeline["vectorizer"].vocabulary_
    reps = pd.DataFrame(json.loads((source / "representative_docs.json").read_text()))
    assert not reps["article_id"].str.startswith("test_").any()
    # Fit cuối đã thấy valonly, nhưng model validation không thấy nó:
    val_docs = pd.read_parquet(source / "validation_document_topics.parquet")
    dev_docs = pd.read_parquet(source / "development_document_topics.parquet")
    assert set(val_docs["split"]) == {"val"}
    assert set(dev_docs["split"]) == {"train", "val"}
    assert dev_docs["title"].notna().all()
    bundle = export_nmf_artifacts(source, tmp_path / "common", results)
    assert get_split_metrics(bundle)["split"].tolist() == ["validation", "test"]
    with pytest.raises(FileExistsError):
        prepare_dashboard(config, dataset, "independent")
