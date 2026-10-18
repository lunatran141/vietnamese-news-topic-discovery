
"""Tests for Task 3.7 LDA final evaluation."""

import pandas as pd
import pytest
import yaml

from experiments.lda.run_lda_final import (
    load_config,
    validate_splits,
)


def make_split(*article_ids):
    return pd.DataFrame({"article_id": list(article_ids)})


def test_config_has_expected_dataset():
    config, base = load_config("configs/lda_final.yaml")

    assert config["dataset"]["data_version"] == "v1.0"
    assert config["dataset"]["preprocessing_variant"] == "p2_segmented"
    assert config["dataset"]["path"] == base["dataset"]["path"]


def test_k_matches_m08_m09_candidate():
    config, _ = load_config("configs/lda_final.yaml")

    assert config["selection"]["selected_n_topics"] == 15
    assert config["model"]["random_state"] == 42



def test_final_test_is_locked():
    config, _ = load_config("configs/lda_final.yaml")

    assert config["selection"]["selected_n_topics"] == 15
    assert config["selection"]["comparison_k"] == 20
    assert config["selection"]["approval_status"] == "approved"
    assert config["safety"]["allow_test"] is False



def test_disjoint_splits_are_accepted():
    train = make_split("A", "B")
    val = make_split("C")
    test = make_split("D")

    validate_splits(train, val, test)


def test_overlapping_splits_are_rejected():
    train = make_split("A", "B")
    val = make_split("B", "C")

    with pytest.raises(ValueError, match="leakage"):
        validate_splits(train, val)


def test_duplicate_article_ids_are_rejected():
    train = make_split("A", "A")
    val = make_split("B")

    with pytest.raises(ValueError, match="Invalid article_id"):
        validate_splits(train, val)


def test_missing_article_ids_are_rejected():
    train = make_split("A", None)
    val = make_split("B")

    with pytest.raises(ValueError, match="Invalid article_id"):
        validate_splits(train, val)


def test_empty_split_is_rejected():
    train = make_split("A")
    val = make_split()

    with pytest.raises(ValueError, match="Invalid article_id"):
        validate_splits(train, val)


import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from experiments.lda.run_lda_final import evaluate_test
from experiments.nmf.metrics import npmi_coherence


def test_common_npmi_perfect_cooccurrence():
    matrix = csr_matrix([
        [1, 1],
        [1, 1],
        [0, 0],
        [0, 0],
    ])

    score, topic_scores = npmi_coherence(matrix, [[0, 1]])

    assert np.isclose(score, 1.0)
    assert np.isclose(topic_scores[0], 1.0)


def test_common_npmi_never_cooccurs():
    matrix = csr_matrix([
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1],
    ])

    score, topic_scores = npmi_coherence(matrix, [[0, 1]])

    assert np.isclose(score, -1.0)
    assert np.isclose(topic_scores[0], -1.0)


def test_lda_final_metric_schema():
    documents = [
        "bong_da cau_thu doi_bong",
        "bong_da cau_thu",
        "kinh_te doanh_nghiep tai_chinh",
        "kinh_te tai_chinh",
    ]

    vectorizer = CountVectorizer(
        token_pattern=r"(?u)\b\w\w+\b"
    )
    matrix = vectorizer.fit_transform(documents)

    model = LatentDirichletAllocation(
        n_components=2,
        random_state=42,
        max_iter=5,
    )
    model.fit(matrix)

    gold = pd.Series(["sport", "sport", "economy", "economy"])

    metrics, distributions, topics, per_topic_npmi = evaluate_test(
        model,
        vectorizer,
        matrix,
        gold,
        top_n=2,
    )

    expected_keys = {
        "n_topics",
        "npmi_coherence",
        "topic_diversity",
        "adjusted_rand_index",
        "normalized_mutual_info",
        "test_documents",
    }

    assert expected_keys.issubset(metrics)
    assert distributions.shape == (4, 2)
    assert len(topics) == 2
    assert len(per_topic_npmi) == 2
    assert -1 <= metrics["npmi_coherence"] <= 1
    assert 0 <= metrics["topic_diversity"] <= 1
    assert -1 <= metrics["adjusted_rand_index"] <= 1
    assert 0 <= metrics["normalized_mutual_info"] <= 1
