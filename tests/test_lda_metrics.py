
"""Unit tests for M08 LDA evaluation metrics."""

import numpy as np
import pytest

from src.features.bow import fit_bow_pipeline
from src.models.lda_pipeline import fit_lda_pipeline
from src.evaluation.lda_metrics import (
    extract_topic_words,
    topic_diversity,
    prepare_coherence_corpus,
    calculate_coherence,
)


@pytest.fixture
def sample_model():
    docs = [
        "alpha beta gamma delta",
        "alpha beta gamma epsilon",
        "alpha beta delta gamma",
        "school teacher student class",
        "school teacher class lesson",
        "school student teacher lesson",
    ]

    vectorizer, matrix = fit_bow_pipeline(
        docs, {"min_df": 1, "max_df": 1.0}
    )

    model = fit_lda_pipeline(
        matrix,
        {
            "n_components": 2,
            "max_iter": 10,
            "random_state": 42,
            "n_jobs": 1,
        },
    )

    return docs, vectorizer, model


def test_extract_topic_words(sample_model):
    _, vectorizer, model = sample_model

    topics = extract_topic_words(
        model,
        vectorizer.get_feature_names_out(),
        top_n=5,
    )

    assert len(topics) == 2
    assert all(len(topic) == 5 for topic in topics)
    assert all(isinstance(word, str) for t in topics for word in t)


def test_topic_diversity():
    topics = [
        ["alpha", "beta", "gamma"],
        ["alpha", "delta", "epsilon"],
    ]

    assert topic_diversity(topics) == pytest.approx(5 / 6)


def test_topic_diversity_empty():
    with pytest.raises(ValueError):
        topic_diversity([])


def test_coherence_corpus(sample_model):
    docs, vectorizer, _ = sample_model

    texts, dictionary = prepare_coherence_corpus(
        docs, vectorizer
    )

    assert len(texts) == len(docs)
    assert len(dictionary) > 0
    assert all(
        token in dictionary.token2id
        for text in texts
        for token in text
    )


@pytest.mark.parametrize("metric", ["c_v", "c_npmi"])
def test_coherence(sample_model, metric):
    docs, vectorizer, model = sample_model

    topics = extract_topic_words(
        model,
        vectorizer.get_feature_names_out(),
        top_n=5,
    )

    texts, dictionary = prepare_coherence_corpus(
        docs, vectorizer
    )

    score = calculate_coherence(
        topics,
        texts,
        dictionary,
        metric=metric,
        processes=1,
    )

    assert np.isfinite(score)


def test_invalid_coherence_metric(sample_model):
    docs, vectorizer, model = sample_model

    topics = extract_topic_words(
        model,
        vectorizer.get_feature_names_out(),
        top_n=5,
    )

    texts, dictionary = prepare_coherence_corpus(
        docs, vectorizer
    )

    with pytest.raises(ValueError):
        calculate_coherence(
            topics,
            texts,
            dictionary,
            metric="invalid",
        )
