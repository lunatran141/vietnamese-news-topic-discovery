"""Chạy từng trang của dashboard bằng AppTest để bắt lỗi runtime."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.data_service import discover_artifact_runs, enrich_documents, get_document_table, load_dashboard_bundle
from app.ui.components import build_topic_labels
from app.views.articles import INFERENCE_MODELS, REVIEW_TYPES, _prepare_review_table, _select_review_cases

ROOT = Path(__file__).resolve().parents[1]
RUNS = {run.run_id: run for run in discover_artifact_runs()}
VIEWS = ["overview", "explorer", "articles", "organizer", "evaluation", "about"]


def _view_script(view: str, run_id: str, split: str, root: str) -> None:
    import sys

    sys.path.insert(0, root)
    import importlib

    from app.data_service import discover_artifact_runs, load_dashboard_bundle
    from app.ui.context import AnalysisContext
    from app.ui.theme import inject_css

    inject_css()
    runs = discover_artifact_runs()
    run = next(r for r in runs if r.run_id == run_id)
    ctx = AnalysisContext(runs, run, load_dashboard_bundle(str(run.path)), split)
    importlib.import_module(f"app.views.{view}").render(ctx)


def _run(view: str, run_id: str, split: str = "all") -> AppTest:
    test = AppTest.from_function(
        _view_script, args=(view, run_id, split, str(ROOT)), default_timeout=240
    )
    return test.run()


@pytest.mark.parametrize("run_id", sorted(RUNS))
@pytest.mark.parametrize("view", VIEWS)
def test_view_renders_without_exception(view: str, run_id: str) -> None:
    test = _run(view, run_id)
    assert not test.exception, [e.value for e in test.exception]


def test_split_filter_renders() -> None:
    run_id = next(r for r in RUNS if "e1_vietnamese_sbert" in r)
    test = _run("overview", run_id, "val")
    assert not test.exception, [e.value for e in test.exception]


def test_inference_models_default_order() -> None:
    assert list(INFERENCE_MODELS)[0] == "K-Means"
    assert INFERENCE_MODELS["K-Means"] == "kmeans"


@pytest.mark.parametrize("run_id", sorted(RUNS))
def test_review_cases_exclude_outliers_and_prepare_clean_csv_columns(run_id: str) -> None:
    run = RUNS[run_id]
    bundle = load_dashboard_bundle(str(run.path))
    labels = build_topic_labels(bundle, run_id)
    documents = enrich_documents(get_document_table(bundle, "all"))

    for review_type in REVIEW_TYPES:
        cases = _select_review_cases(documents, review_type)
        if review_type == "Outlier":
            if not cases.empty:
                assert (cases["topic_id"] == -1).all() or cases["is_outlier"].astype(bool).all()
        else:
            if not cases.empty:
                assert (cases["topic_id"] != -1).all(), f"{run_id} leaked topic_id=-1 into {review_type}"

        table = _prepare_review_table(cases, labels)
        assert "topic_label" in table.columns
        if not table.empty:
            for col in table.columns:
                if col not in {"article_id", "title", "topic_id", "topic_label"}:
                    assert table[col].notna().any(), f"{run_id} ({review_type}) has all-NaN column {col}"
            if "second_topic_id" in table.columns:
                assert "secondary_topic_id" not in table.columns
            if "second_topic_share" in table.columns:
                assert "secondary_topic_share" not in table.columns


def test_review_table_preserves_model_specific_columns() -> None:
    kmeans_run = next(r for r in RUNS.values() if "kmeans" in r.run_id.lower())
    kmeans_bundle = load_dashboard_bundle(str(kmeans_run.path))
    kmeans_docs = enrich_documents(get_document_table(kmeans_bundle, "all"))
    kmeans_table = _prepare_review_table(
        _select_review_cases(kmeans_docs, "Topic quá rộng"),
        build_topic_labels(kmeans_bundle, kmeans_run.run_id),
    )
    assert "distance_to_center" in kmeans_table.columns
    assert "topic_probability" not in kmeans_table.columns

    lda_run = next(r for r in RUNS.values() if "lda" in r.run_id.lower())
    lda_bundle = load_dashboard_bundle(str(lda_run.path))
    lda_docs = enrich_documents(get_document_table(lda_bundle, "all"))
    lda_table = _prepare_review_table(
        _select_review_cases(lda_docs, "Bài đa chủ đề"),
        build_topic_labels(lda_bundle, lda_run.run_id),
    )
    assert "second_topic_probability" in lda_table.columns
    assert "topic_probability" in lda_table.columns


def test_articles_review_radio_modes_render_without_exception() -> None:
    run_id = next(r for r in RUNS if "e1_vietnamese_sbert" in r)
    test = _run("articles", run_id, "all")
    assert not test.exception, [e.value for e in test.exception]
    for mode in REVIEW_TYPES:
        test.radio(key="review_type").set_value(mode).run()
        assert not test.exception, [e.value for e in test.exception]
