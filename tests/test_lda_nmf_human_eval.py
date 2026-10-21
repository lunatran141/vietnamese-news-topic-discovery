
"""Tests for reproducible LDA–NMF human evaluation sampling."""

import pandas as pd
import pytest

from experiments.human_eval import build_lda_nmf_samples as he
from experiments.human_eval.summarize_lda_nmf import summarize, TOPIC_SCORES, FLAGS, DOC_SCORE


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


def test_pending_review_does_not_publish_scores(evaluation_data, tmp_path, monkeypatch):
    _, _, data, mapping = evaluation_data
    monkeypatch.setattr(he, "OUTPUT_DIR", tmp_path)
    he.export_evaluation_files(data, mapping)
    result = summarize(tmp_path)
    assert result["status"] == "pending"
    assert result["missing_ratings"] == 500
    assert result["by_model"] == []


def test_complete_reviews_and_invalid_ratings(evaluation_data, tmp_path, monkeypatch):
    _, _, data, mapping = evaluation_data
    monkeypatch.setattr(he, "OUTPUT_DIR", tmp_path)
    he.export_evaluation_files(data, mapping)
    # Điểm giả chỉ trong fixture tạm để kiểm tra tổng hợp, không phải đánh giá thật.
    for reviewer in ("R1", "R2"):
        for kind in ("topics", "documents"):
            path = tmp_path / f"reviewer_{reviewer}_{kind}.csv"
            frame = pd.read_csv(path)
            for column in TOPIC_SCORES if kind == "topics" else [DOC_SCORE]:
                frame[column] = 4
            if kind == "topics":
                for column in FLAGS:
                    frame[column] = "no"
            frame.to_csv(path, index=False)
    result = summarize(tmp_path)
    assert result["status"] == "complete"
    assert len(result["by_model"]) == 2
    assert all(row[DOC_SCORE] == 4 for row in result["by_model"])
    assert all(row["kappa"] is None for row in result["agreement"])
    path = tmp_path / "reviewer_R1_topics.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "keyword_coherence_1_5"] = 6
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="1-5"):
        summarize(tmp_path)
