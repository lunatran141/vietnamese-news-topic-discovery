
"""Unit tests for M09 LDA stability metrics."""

import numpy as np
import pytest

from src.evaluation.lda_stability import (
    jaccard_similarity,
    topic_similarity_matrix,
    match_topics,
    align_document_topics,
    document_topic_stability,
)


def test_jaccard_identical():
    assert jaccard_similarity(
        ["bank", "credit"], ["credit", "bank"]
    ) == 1.0


def test_jaccard_partial_overlap():
    assert jaccard_similarity(
        ["a", "b"], ["b", "c"]
    ) == pytest.approx(1 / 3)


def test_jaccard_empty_sets():
    assert jaccard_similarity([], []) == 1.0
    assert jaccard_similarity(["a"], []) == 0.0


def test_similarity_matrix_shape():
    topics_a = [["a"], ["b"]]
    topics_b = [["b"], ["a"]]

    matrix = topic_similarity_matrix(topics_a, topics_b)

    assert matrix.shape == (2, 2)
    assert np.allclose(matrix, [[0, 1], [1, 0]])


def test_topic_count_mismatch():
    with pytest.raises(ValueError):
        topic_similarity_matrix([["a"]], [["a"], ["b"]])


def test_empty_topic_list():
    with pytest.raises(ValueError):
        topic_similarity_matrix([], [])


def test_hungarian_permutation():
    reference = [["bank", "credit"], ["school", "student"]]
    candidate = [["school", "student"], ["bank", "credit"]]

    matches, mean_similarity = match_topics(
        reference, candidate
    )

    assert matches == [(0, 1, 1.0), (1, 0, 1.0)]
    assert mean_similarity == pytest.approx(1.0)


def test_align_document_topics():
    candidate = np.array([
        [0.2, 0.8],
        [0.7, 0.3],
    ])

    matches = [
        (0, 1, 1.0),
        (1, 0, 1.0),
    ]

    aligned = align_document_topics(candidate, matches)

    expected = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
    ])

    assert np.allclose(aligned, expected)


def test_align_duplicate_match_rejected():
    candidate = np.array([[0.5, 0.5]])

    matches = [
        (0, 0, 1.0),
        (1, 0, 1.0),
    ]

    with pytest.raises(ValueError):
        align_document_topics(candidate, matches)


def test_document_cosine_identical():
    distributions = np.array([
        [0.8, 0.2],
        [0.1, 0.9],
    ])

    score = document_topic_stability(
        distributions, distributions
    )

    assert score == pytest.approx(1.0)


def test_document_cosine_different():
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])

    assert document_topic_stability(a, b) == pytest.approx(0.0)


def test_document_topic_shape_mismatch():
    a = np.array([[0.5, 0.5]])
    b = np.array([[0.2, 0.3, 0.5]])

    with pytest.raises(ValueError):
        document_topic_stability(a, b)
