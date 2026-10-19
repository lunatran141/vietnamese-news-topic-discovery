
"""Tests for reproducible LDA–NMF human evaluation sampling."""

import pandas as pd
import pytest

from experiments.human_eval import build_lda_nmf_samples as he


@pytest.fixture(scope="module")
def evaluation_data():
    inputs = he.load_inputs()
    samples = he.select_samples(inputs)
    reviewer_data, mapping = he.build_evaluation_data(inputs, samples)
    return inputs, samples, reviewer_data, mapping


def test_sampling_counts(evaluation_data):
    _, samples, _, _ = evaluation_data

    assert len(samples) == 125

    counts = samples.groupby(["model", "sample_type"]).size()

    assert counts["LDA", "top"] == 45
    assert counts["LDA", "random"] == 30
    assert counts["NMF", "top"] == 30
    assert counts["NMF", "random"] == 20

    assert samples.groupby(["model", "topic_id"]).size().eq(5).all()


def test_no_test_leakage_or_duplicate_pairs(evaluation_data):
    inputs, samples, _, _ = evaluation_data

    development_ids = set(
        inputs["manifest"].loc[
            inputs["manifest"]["split"].isin(["train", "val"]),
            "article_id",
        ].astype(str)
    )

    assert samples["article_id"].isin(development_ids).all()
    assert not samples.duplicated(
        ["model", "topic_id", "article_id"]
    ).any()


def test_blind_topic_mapping(evaluation_data):
    _, _, reviewer_data, mapping = evaluation_data

    assert len(mapping) == 25
    assert mapping["blind_topic_id"].is_unique
    assert reviewer_data["blind_topic_id"].nunique() == 25
    assert reviewer_data.groupby("blind_topic_id").size().eq(5).all()

    assert reviewer_data["title"].notna().all()
    assert reviewer_data["content"].notna().all()


def test_export_refuses_overwrite(evaluation_data, tmp_path, monkeypatch):
    _, _, reviewer_data, mapping = evaluation_data

    monkeypatch.setattr(he, "OUTPUT_DIR", tmp_path)

    marker = tmp_path / "reviewer_R1_topics.csv"
    marker.write_text("DO NOT OVERWRITE", encoding="utf-8")

    with pytest.raises(FileExistsError):
        he.export_evaluation_files(reviewer_data, mapping)

    assert marker.read_text(encoding="utf-8") == "DO NOT OVERWRITE"
