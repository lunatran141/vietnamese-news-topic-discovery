
"""M09: Topic matching and stability metrics for LDA."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def jaccard_similarity(words_a, words_b):
    """Jaccard similarity between two sets of topic keywords."""
    a, b = set(words_a), set(words_b)

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def topic_similarity_matrix(reference_topics, candidate_topics):
    """
    Build K x K Jaccard similarity matrix.

    Rows: reference topics.
    Columns: candidate topics.
    """
    if len(reference_topics) != len(candidate_topics):
        raise ValueError("Models must have the same number of topics.")

    if not reference_topics:
        raise ValueError("Topic lists must not be empty.")

    return np.array(
        [
            [jaccard_similarity(a, b) for b in candidate_topics]
            for a in reference_topics
        ],
        dtype=float,
    )


def match_topics(reference_topics, candidate_topics):
    """
    Match topics with the Hungarian algorithm.

    Returns:
        matches: list of (reference_id, candidate_id, similarity)
        mean_similarity: average Jaccard similarity of matched topics
    """
    similarity = topic_similarity_matrix(
        reference_topics, candidate_topics
    )

    row_ind, col_ind = linear_sum_assignment(
        similarity, maximize=True
    )

    matches = [
        (int(i), int(j), float(similarity[i, j]))
        for i, j in zip(row_ind, col_ind)
    ]

    mean_similarity = float(
        np.mean([match[2] for match in matches])
    )

    return matches, mean_similarity


def align_document_topics(candidate_distributions, matches):
    """
    Reorder candidate topic probabilities to reference topic order.

    candidate_distributions: shape (n_documents, K)
    matches: output from match_topics()
    """
    distributions = np.asarray(
        candidate_distributions, dtype=float
    )

    if distributions.ndim != 2:
        raise ValueError("Expected a 2D document-topic matrix.")

    k = distributions.shape[1]

    if len(matches) != k:
        raise ValueError("Number of matches must equal K.")

    aligned = np.empty_like(distributions)
    seen_reference = set()
    seen_candidate = set()

    for reference_id, candidate_id, _ in matches:
        if (
            not 0 <= reference_id < k
            or not 0 <= candidate_id < k
            or reference_id in seen_reference
            or candidate_id in seen_candidate
        ):
            raise ValueError("Invalid or duplicate topic match.")

        aligned[:, reference_id] = distributions[:, candidate_id]
        seen_reference.add(reference_id)
        seen_candidate.add(candidate_id)

    return aligned


def document_topic_stability(reference, candidate_aligned):
    """
    Mean cosine similarity of per-document topic distributions.

    Both matrices must have the same shape (n_documents, K).
    """
    a = np.asarray(reference, dtype=float)
    b = np.asarray(candidate_aligned, dtype=float)

    if a.ndim != 2 or a.shape != b.shape or a.shape[0] == 0:
        raise ValueError("Invalid document-topic matrix shapes.")

    numerator = np.sum(a * b, axis=1)
    denominator = (
        np.linalg.norm(a, axis=1)
        * np.linalg.norm(b, axis=1)
    )

    similarities = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )

    return float(np.mean(similarities))
