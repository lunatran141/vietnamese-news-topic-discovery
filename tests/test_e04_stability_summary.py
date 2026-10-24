"""Protect identity alignment, outlier accounting and descriptive aggregation."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.stability_summary import pairwise_assignment_ari, summarize_metrics


def test_assignment_ari_aligns_ids_and_ignores_label_permutation():
    data = pd.DataFrame({"seed": [42] * 4 + [52] * 4,
                         "article_id": ["a", "b", "c", "d", "d", "c", "b", "a"],
                         "topic_id": [0, 0, 1, 1, 8, 8, 7, 7]})
    result = pairwise_assignment_ari(data)
    assert len(result) == 1
    assert result.iloc[0].ari_between_runs == pytest.approx(1)


@pytest.mark.parametrize("ids", [["a", "a"], ["a", "c"]])
def test_invalid_document_identity_is_rejected(ids):
    data = pd.DataFrame({"seed": [42, 42, 52, 52],
                         "article_id": ["a", "b"] + ids, "topic_id": [0, 1, 0, 1]})
    with pytest.raises(ValueError):
        pairwise_assignment_ari(data)


def test_outlier_subset_reports_pair_specific_denominator():
    data = pd.DataFrame({"seed": [42] * 4 + [52] * 4,
                         "article_id": list("abcd") * 2,
                         "topic_id": [-1, 0, 0, 1, 2, -1, 2, 3]})
    row = pairwise_assignment_ari(data).iloc[0]
    assert row.n_documents == 4
    assert row.n_common_non_outliers == 2
    assert row.common_non_outlier_fraction == 0.5
    assert row.ari_common_non_outliers == 1


def test_insufficient_common_subset_is_missing_not_perfect():
    data = pd.DataFrame({"seed": [42, 42, 52, 52],
                         "article_id": ["a", "b"] * 2, "topic_id": [-1, 0, 0, -1]})
    assert np.isnan(pairwise_assignment_ari(data).iloc[0].ari_common_non_outliers)


def test_negative_ari_is_retained():
    data = pd.DataFrame({"seed": [42] * 4 + [52] * 4,
                         "article_id": list("abcd") * 2,
                         "topic_id": [0, 0, 1, 1, 0, 1, 0, 1]})
    assert pairwise_assignment_ari(data).iloc[0].ari_between_runs == pytest.approx(-0.5)


def test_summary_uses_sample_std_and_keeps_missing_metrics_missing():
    data = pd.DataFrame({"method": ["LDA"] * 3, "topic_count": [14, 15, 16],
                         "coherence": [np.nan] * 3})
    table = summarize_metrics(data, ["topic_count", "coherence"], "seed")
    assert len(table) == 1
    row = table.iloc[0]
    assert row["mean"] == 15
    assert row.sample_std == 1
    assert row["range"] == 2
