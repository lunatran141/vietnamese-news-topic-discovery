"""Summarize repeated runs without confusing external labels with stability."""

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def pairwise_assignment_ari(assignments):
    """Compare identical document sets by ID; retain outlier label -1.

    All-document ARI is reported together with ARI on documents assigned to
    ordinary topics in both runs. The latter has a pair-specific denominator.
    """
    required = ["seed", "article_id", "topic_id"]
    if not set(required).issubset(assignments.columns):
        raise ValueError("Assignments need seed, article_id and topic_id")
    if assignments[required].isna().any().any():
        raise ValueError("Missing assignment value")
    if assignments.duplicated(["seed", "article_id"]).any():
        raise ValueError("Duplicate article_id within seed")
    groups = {
        seed: frame.set_index("article_id")["topic_id"].sort_index()
        for seed, frame in assignments.groupby("seed", sort=True)
    }
    if len(groups) < 2:
        raise ValueError("At least two seeds are required")
    rows = []
    for a, b in combinations(groups, 2):
        left, right = groups[a], groups[b]
        if not left.index.equals(right.index):
            raise ValueError("Seed runs must contain identical article_id sets")
        common = left.ne(-1) & right.ne(-1)
        n_common = int(common.sum())
        rows.append({
            "seed_a": a, "seed_b": b, "n_documents": len(left),
            "ari_between_runs": adjusted_rand_score(left, right),
            "n_common_non_outliers": n_common,
            "common_non_outlier_fraction": n_common / len(left),
            "ari_common_non_outliers": (
                adjusted_rand_score(left[common], right[common])
                if n_common >= 2 else np.nan
            ),
        })
    return pd.DataFrame(rows)


def summarize_metrics(frame, metrics, unit):
    """Long-form descriptive statistics; sample SD, no imputation of missing data."""
    rows = []
    for method, group in frame.groupby("method", sort=True):
        for metric in metrics:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="raise").dropna()
            if values.empty:
                continue
            if not np.isfinite(values.to_numpy()).all():
                raise ValueError(f"Non-finite {method}/{metric}")
            rows.append({
                "method": method, "unit": unit, "metric": metric,
                "count": len(values), "mean": values.mean(),
                "sample_std": values.std(ddof=1),
                "min": values.min(), "max": values.max(),
                "range": values.max() - values.min(),
            })
    return pd.DataFrame(rows)
