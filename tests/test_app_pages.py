"""Chạy từng trang của dashboard bằng AppTest để bắt lỗi runtime."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.data_service import discover_artifact_runs

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

