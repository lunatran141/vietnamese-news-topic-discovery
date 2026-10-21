"""Kiểm tra bảng tổng hợp stability HDBSCAN E04."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.hdbscan.summarize_e04_stability import (
    RUN_ID,
    SEEDS,
    jaccard_similarity,
    match_topics_by_article_overlap,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = (
    ROOT
    / "results/hdbscan/stability"
    / f"{RUN_ID}_seeds42_51"
)
SUMMARY_PATH = RESULT_DIR / "stability_summary.csv"
PAIRWISE_KEYWORDS_PATH = RESULT_DIR / "pairwise_keyword_jaccard.csv"


def test_jaccard_similarity() -> None:
    assert jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard_similarity({"a"}, {"b"}) == 0.0
    assert jaccard_similarity({"a", "b"}, {"b", "c"}) == pytest.approx(
        1 / 3
    )
    assert jaccard_similarity(set(), set()) == 1.0


def test_match_topics_uses_article_overlap_instead_of_topic_id() -> None:
    assignments_a = pd.DataFrame(
        {
            "article_id": ["a", "b", "c", "d", "e"],
            "cluster_id": [0, 0, 1, 1, -1],
        }
    )
    assignments_b = pd.DataFrame(
        {
            "article_id": ["a", "b", "c", "d", "e"],
            "cluster_id": [9, 9, 7, 7, -1],
        }
    )

    matches = set(
        match_topics_by_article_overlap(assignments_a, assignments_b)
    )
    assert matches == {(0, 9), (1, 7)}


def test_stability_summary_schema_and_values() -> None:
    assert SUMMARY_PATH.is_file()
    summary = pd.read_csv(SUMMARY_PATH)

    assert list(summary.columns) == [
        "seed",
        "topic_count",
        "outlier_rate",
        "coverage",
        "ari_between_runs",
        "keyword_jaccard",
    ]
    assert summary["seed"].tolist() == list(SEEDS)
    assert summary["seed"].is_unique
    assert summary.notna().all().all()
    assert summary["topic_count"].gt(0).all()

    for column in (
        "outlier_rate",
        "coverage",
        "ari_between_runs",
        "keyword_jaccard",
    ):
        assert np.isfinite(summary[column]).all()
        assert summary[column].between(0, 1).all()

    assert np.allclose(
        summary["coverage"].to_numpy(dtype=float),
        1.0 - summary["outlier_rate"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
    )


def test_pairwise_keyword_jaccard_has_all_seed_pairs() -> None:
    assert PAIRWISE_KEYWORDS_PATH.is_file()
    pairwise = pd.read_csv(PAIRWISE_KEYWORDS_PATH)

    assert list(pairwise.columns) == [
        "seed_a",
        "seed_b",
        "matched_topic_count",
        "keyword_jaccard",
    ]
    assert len(pairwise) == 45
    assert pairwise[["seed_a", "seed_b"]].duplicated().sum() == 0
    assert pairwise["seed_a"].lt(pairwise["seed_b"]).all()
    assert pairwise["matched_topic_count"].gt(0).all()
    assert pairwise["keyword_jaccard"].between(0, 1).all()
    assert pairwise.notna().all().all()
