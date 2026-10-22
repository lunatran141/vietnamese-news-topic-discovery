"""Biểu đồ Plotly dùng chung (template plotly_white, màu trầm)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from app.ui.theme import (
    BORDER, FONT_STACK, INK, MUTED, OUTLIER_COLOR, OUTLIER_OPACITY, ACCENT, color_for,
)

PLOT_CONFIG = {"displayModeBar": False}


def _base(height: int, **layout) -> dict:
    return {
        "template": "plotly_white",
        "height": height,
        "margin": {"l": 10, "r": 10, "t": 10, "b": 10},
        "font": {"family": FONT_STACK, "color": INK, "size": 12},
        "paper_bgcolor": "#FFFFFF",
        "plot_bgcolor": "#FFFFFF",
        **layout,
    }


def _short(text: str, limit: int = 42) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def empty_figure(message: str, height: int = 240) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False, font={"color": MUTED, "size": 13})
    figure.update_layout(**_base(height, xaxis={"visible": False}, yaxis={"visible": False}))
    return figure


def topic_distribution_bar(topics: pd.DataFrame, labels: dict[int, str], top_n: int = 15) -> go.Figure:
    top = topics.head(top_n)
    if top.empty:
        return empty_figure("Không có topic trên tập đang chọn.")
    names = [_short(labels.get(int(t), f"T{int(t)}")) for t in top["topic_id"]]
    figure = go.Figure(
        go.Bar(
            x=top["document_count"],
            y=names,
            orientation="h",
            marker={"color": ACCENT},
            text=top["document_count"],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>%{x:,} bài<extra></extra>",
        )
    )
    figure.update_layout(
        **_base(
            max(260, 28 * len(top) + 40),
            xaxis={"title": "Số bài", "gridcolor": "#F0F1F3"},
            yaxis={"autorange": "reversed", "showgrid": False},
            margin={"l": 10, "r": 40, "t": 10, "b": 30},
        )
    )
    return figure


def projection_scatter(
    points: pd.DataFrame,
    labels: dict[int, str],
    topic_ids: list[int],
    selected: int | None = None,
    show_outliers: bool = False,
) -> go.Figure:
    """Bản đồ 2D bằng WebGL; cụm nhiễu xám mờ, cụm đang chọn nổi bật."""
    figure = go.Figure()
    has_title = "title" in points
    has_split = "split" in points

    def add(frame: pd.DataFrame, name: str, color: str, size: int, opacity: float) -> None:
        titles = frame["title"].fillna("").astype(str).map(lambda t: _short(t, 80)) if has_title else [""] * len(frame)
        splits = frame["split"].astype(str) if has_split else [""] * len(frame)
        figure.add_trace(
            go.Scattergl(
                x=frame["x"], y=frame["y"], mode="markers", name=_short(name, 36),
                marker={"color": color, "size": size, "opacity": opacity},
                customdata=list(zip(titles, [name] * len(frame), splits)),
                hovertemplate="%{customdata[0]}<br><b>%{customdata[1]}</b> · %{customdata[2]}<extra></extra>",
            )
        )

    if show_outliers:
        noise = points.loc[points["topic_id"] == -1]
        if not noise.empty:
            add(noise, "Outlier", OUTLIER_COLOR, 4, OUTLIER_OPACITY)
    for index, topic_id in enumerate(topic_ids):
        frame = points.loc[points["topic_id"] == topic_id]
        if frame.empty:
            continue
        is_selected = selected is not None and topic_id == selected
        dimmed = selected is not None and not is_selected
        add(
            frame, labels.get(topic_id, f"T{topic_id}"), color_for(index),
            9 if is_selected else 6, 0.25 if dimmed else (1.0 if is_selected else 0.8),
        )
    figure.update_layout(
        **_base(
            560,
            xaxis={"title": "Trục 1", "showgrid": True, "gridcolor": "#F0F1F3", "zeroline": False},
            yaxis={"title": "Trục 2", "showgrid": True, "gridcolor": "#F0F1F3", "zeroline": False},
            showlegend=len(topic_ids) <= 12,
            legend={"orientation": "h", "y": -0.12, "font": {"size": 11}},
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
        )
    )
    return figure


def trend_line(series: dict[str, pd.DataFrame]) -> go.Figure:
    """`series` ánh xạ tên topic sang bảng (period, count)."""
    figure = go.Figure()
    for index, (name, frame) in enumerate(series.items()):
        if frame.empty:
            continue
        x = pd.to_datetime(frame["period"].astype(str).str.split("/").str[0], errors="coerce")
        figure.add_trace(
            go.Scatter(
                x=x, y=frame["count"], mode="lines+markers", name=_short(name, 36),
                line={"color": color_for(index), "width": 2}, marker={"size": 5},
                hovertemplate="%{x|%d/%m/%Y}<br>%{y} bài<extra>" + _short(name, 30) + "</extra>",
            )
        )
    if not figure.data:
        return empty_figure("Chưa có bài có ngày xuất bản hợp lệ.")
    figure.update_layout(
        **_base(
            400,
            xaxis={"title": "Thời gian", "gridcolor": "#F0F1F3"},
            yaxis={"title": "Số bài", "gridcolor": "#F0F1F3", "rangemode": "tozero"},
            legend={"orientation": "h", "y": -0.2},
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            hovermode="x unified",
        )
    )
    return figure


def keyword_bar(keywords: pd.DataFrame, score_column: str, axis_title: str = "Điểm") -> go.Figure:
    frame = keywords.dropna(subset=[score_column])
    if frame.empty:
        return empty_figure("Chưa có điểm từ khóa.")
    figure = go.Figure(
        go.Bar(
            x=frame[score_column], y=frame["keyword"].astype(str), orientation="h",
            marker={"color": ACCENT},
            hovertemplate="%{y}<br>%{x:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        **_base(
            max(220, 26 * len(frame) + 40),
            xaxis={"title": axis_title, "gridcolor": "#F0F1F3"},
            yaxis={"autorange": "reversed", "showgrid": False},
        )
    )
    return figure


def model_compare_bar(table: pd.DataFrame, metrics: list[str]) -> go.Figure:
    """Cột nhóm so sánh các metric (thang 0 đến 1) giữa các run."""
    available = [m for m in metrics if m in table and table[m].notna().any()]
    if table.empty or not available:
        return empty_figure("Chưa có metric để so sánh.")
    names = [_short(f"{m} · {r}", 30) for m, r in zip(table["Mô hình"], table["Run"])]
    figure = go.Figure()
    for index, metric in enumerate(available):
        figure.add_trace(
            go.Bar(
                name=metric.replace(" (run)", ""), x=names, y=table[metric],
                marker={"color": color_for(index)},
                hovertemplate="%{x}<br>" + metric + ": %{y:.3f}<extra></extra>",
            )
        )
    figure.update_layout(
        **_base(
            380,
            barmode="group",
            yaxis={"title": "Giá trị", "gridcolor": "#F0F1F3", "rangemode": "tozero"},
            xaxis={"tickangle": -20, "showgrid": False, "linecolor": BORDER},
            legend={"orientation": "h", "y": -0.35},
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
        )
    )
    return figure

