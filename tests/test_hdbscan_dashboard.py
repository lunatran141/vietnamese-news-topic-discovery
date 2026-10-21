from pathlib import Path

import pytest

from app.data_service import (
    get_hdbscan_split_summary,
    load_dashboard_bundle,
    load_hdbscan_error_cases,
    load_hdbscan_experiment_comparison,
    load_hdbscan_projection,
    load_hdbscan_runtime,
    load_hdbscan_stability,
    load_hdbscan_transition_hard_cases,
)


ARTIFACT_DIR = Path(
    "artifacts/common/hdbscan/"
    "minilm_title_content_eom_15_5"
)


@pytest.fixture(scope="module")
def bundle():
    return load_dashboard_bundle(
        str(ARTIFACT_DIR)
    )


def test_hdbscan_runtime_co_du_fit_time():
    runtime = load_hdbscan_runtime(
        ARTIFACT_DIR
    )

    train_validation = runtime[
        "train_validation"
    ]

    assert train_validation[
        "umap_fit_seconds"
    ] > 0

    assert train_validation[
        "hdbscan_fit_seconds"
    ] > 0

    fit_time = (
        train_validation["umap_fit_seconds"]
        + train_validation[
            "hdbscan_fit_seconds"
        ]
    )

    assert fit_time == pytest.approx(
        23.628565,
        abs=0.01,
    )


def test_hdbscan_split_summary_khop_bundle(
    bundle,
):
    summary = get_hdbscan_split_summary(
        bundle
    )

    assert set(summary["split"]) == {
        "train",
        "val",
        "test",
    }

    assert (
        summary["document_count"].sum()
        == len(bundle.document_topics)
    )

    assert (
        summary["assigned_count"]
        + summary["outlier_count"]
        == summary["document_count"]
    ).all()

    assert summary["coverage"].between(
        0,
        1,
    ).all()

    assert summary["outlier_rate"].between(
        0,
        1,
    ).all()


def test_hdbscan_projection_co_metadata_hover(
    bundle,
):
    projection = load_hdbscan_projection(
        bundle,
        ARTIFACT_DIR,
    )

    required_columns = {
        "article_id",
        "split",
        "topic_id",
        "x",
        "y",
        "title",
        "source",
        "published_at",
        "membership_strength",
        "is_outlier",
        "topic_label",
    }

    assert required_columns.issubset(
        projection.columns
    )

    assert len(projection) == len(
        bundle.document_topics
    )

    assert projection[
        "article_id"
    ].is_unique

    assert projection["title"].notna().all()
    assert projection["source"].notna().all()

    assert projection[
        "membership_strength"
    ].between(0, 1).all()


def test_hdbscan_stability_co_du_schema():
    stability = load_hdbscan_stability(
        ARTIFACT_DIR
    )

    assert len(stability) == 10

    assert {
        "seed",
        "topic_count",
        "outlier_rate",
        "coverage",
        "ari_between_runs",
        "keyword_jaccard",
    }.issubset(stability.columns)

    assert stability["seed"].is_unique

    for column in (
        "outlier_rate",
        "coverage",
        "ari_between_runs",
        "keyword_jaccard",
    ):
        assert stability[column].between(
            0,
            1,
        ).all()

def test_hdbscan_experiment_comparison_co_du_bon_run():
    comparison = (
        load_hdbscan_experiment_comparison(
            ARTIFACT_DIR
        )
    )

    assert len(comparison) == 4

    assert set(
        comparison["embedding"]
    ) == {
        "MiniLM",
        "MPNet",
    }

    assert set(
        comparison["selection_method"]
    ) == {
        "EOM",
        "LEAF",
    }

    assert {
        "topic_count",
        "train_outlier_rate",
        "validation_coverage",
        "validation_outlier_rate",
        "validation_ari",
        "validation_nmi",
    }.issubset(comparison.columns)

    for column in (
        "train_outlier_rate",
        "validation_coverage",
        "validation_outlier_rate",
        "validation_ari",
        "validation_nmi",
    ):
        assert comparison[column].between(
            0,
            1,
        ).all()


def test_hdbscan_error_cases_co_du_schema():
    cases = load_hdbscan_error_cases(
        ARTIFACT_DIR
    )

    assert not cases.empty

    assert {
        "case_id",
        "split",
        "embedding_run",
        "experiment",
        "article_id",
        "cluster_id",
        "error_group",
        "title",
        "evidence",
        "interpretation",
    }.issubset(cases.columns)

    assert cases["case_id"].is_unique
    assert cases["article_id"].notna().all()


def test_hdbscan_transition_hard_cases(
    bundle,
):
    cases = (
        load_hdbscan_transition_hard_cases(
            bundle,
            ARTIFACT_DIR,
        )
    )

    assert not cases.empty

    assert {
        "article_id",
        "split",
        "transition_count",
        "compared_seed_pairs",
        "instability_rate",
        "noise_to_cluster",
        "cluster_to_noise",
        "title",
        "membership_strength",
        "top_two_share_gap",
    }.issubset(cases.columns)

    assert cases[
        "transition_count"
    ].gt(0).all()

    assert cases[
        "instability_rate"
    ].between(0, 1).all()

    assert (
        cases["noise_to_cluster"]
        + cases["cluster_to_noise"]
        == cases["transition_count"]
    ).all()