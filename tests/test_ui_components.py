"""Kiểm thử thành phần giao diện dùng chung và biểu đồ."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from app.data_service import discover_artifact_runs, get_topic_table, load_projection
from app.ui import charts
from app.ui.components import clean_topic_name, fmt_seconds
from app.ui.theme import CATEGORICAL, PASTEL_COLORS, OUTLIER_COLOR, OUTLIER_OPACITY, color_for
from src.artifacts.loader import load_artifact_bundle

ROOT = Path(__file__).resolve().parents[1]
EMOJI = re.compile("[\U00010000-\U0010ffff\u2600-\u27bf\u2b50-\u2b55]")


def test_clean_topic_name_variants() -> None:
    assert clean_topic_name("0_doanh_doanh nghiệp_phát triển") == "doanh doanh nghiệp phát triển"
    assert clean_topic_name("galaxy / huawei / smartphone") == "galaxy · huawei · smartphone"
    assert clean_topic_name("Topic 3", "bóng đá, cầu thủ, trận, giải") == "bóng đá · cầu thủ · trận"
    assert clean_topic_name("Topic 3") == "Chưa có tên"
    assert clean_topic_name("Outlier") == "Outlier"
    assert clean_topic_name(None, None) == "Chưa có tên"


def test_palette_is_distinct_and_cycles() -> None:
    assert CATEGORICAL == PASTEL_COLORS
    assert len(PASTEL_COLORS) == 15
    assert len(set(CATEGORICAL)) == len(CATEGORICAL)
    assert color_for(len(CATEGORICAL)) == CATEGORICAL[0]
    assert OUTLIER_COLOR == "#D4D4D8" and OUTLIER_OPACITY == 0.4


def test_charts_with_real_artifacts() -> None:
    for run in discover_artifact_runs():
        bundle = load_artifact_bundle(str(run.path))
        topics = get_topic_table(bundle, split="all")
        labels = {int(t): f"T{int(t)}" for t in bundle.topic_summary["topic_id"]}
        assert isinstance(charts.topic_distribution_bar(topics, labels), go.Figure)
        projection = load_projection(str(run.path))
        if not projection.empty:
            ids = sorted(int(v) for v in projection["topic_id"].unique() if v != -1)[:6]
            figure = charts.projection_scatter(projection.head(2000), labels, ids, ids[0] if ids else None, True)
            assert isinstance(figure, go.Figure)
            assert all(type(trace).__name__ == "Scattergl" for trace in figure.data)


def test_empty_inputs_do_not_raise() -> None:
    assert isinstance(charts.topic_distribution_bar(pd.DataFrame(columns=["topic_id", "document_count"]), {}), go.Figure)
    assert isinstance(charts.trend_line({"A": pd.DataFrame(columns=["period", "count"])}), go.Figure)
    assert isinstance(charts.model_compare_bar(pd.DataFrame(), ["ARI (run)"]), go.Figure)
    assert isinstance(charts.keyword_bar(pd.DataFrame({"keyword": [], "weight": []}), "weight"), go.Figure)


def test_trend_line_parses_weekly_periods() -> None:
    frame = pd.DataFrame({"period": ["2026-09-01/2026-09-07", "2026-09-08/2026-09-14"], "count": [3, 5]})
    figure = charts.trend_line({"T0": frame})
    assert len(figure.data) == 1 and list(figure.data[0].y) == [3, 5]


def test_ui_sources_have_no_emoji() -> None:
    files = [ROOT / "app/streamlit_app.py", *(ROOT / "app/ui").glob("*.py"), *(ROOT / "app/views").glob("*.py")]
    for path in files:
        assert not EMOJI.search(path.read_text(encoding="utf-8")), f"Emoji trong {path.name}"



def test_fmt_seconds() -> None:
    assert fmt_seconds(None) == "N/A"
    assert fmt_seconds(19.7644) == "19.764 giây"
    assert fmt_seconds(2) == "2 giây"
