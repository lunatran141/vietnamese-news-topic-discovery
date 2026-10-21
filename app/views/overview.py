"""Trang 1: Tổng quan run đang chọn."""

from __future__ import annotations

import streamlit as st

from app.data_service import get_model_overview, get_topic_table
from app.ui import charts
from app.ui.components import (
    build_topic_labels, data_table, fmt_number, fmt_percent, fmt_seconds,
    key_value_panel, kpi_row, note, page_header, table_download,
)
from app.ui.context import AnalysisContext


def render(ctx: AnalysisContext) -> None:
    run, bundle, split = ctx.run, ctx.bundle, ctx.split
    page_header(
        "Tổng quan",
        "Quy mô, phân bố chủ đề và cấu hình của run đang xem.",
        {"Run": run.label, "Split": "Tất cả có sẵn" if split == "all" else split},
        eyebrow="Phân tích",
    )
    overview = get_model_overview(bundle, split, artifact_dir=run.path)
    topics = get_topic_table(bundle, split=split)
    labels = build_topic_labels(bundle, run.run_id)

    kpi_row([
        ("Số topic", fmt_number(overview["topic_count"]), None),
        ("Số bài", fmt_number(overview["document_count"]), None),
        ("Đã gán topic", fmt_number(overview["assigned_document_count"]), None),
        ("Outlier", fmt_number(overview["outlier_document_count"]), None),
        ("Tỷ lệ outlier", fmt_percent(overview["outlier_rate"]), None),
    ])
    st.write("")

    left, right = st.columns([1.7, 1])
    with left:
        st.subheader("Phân bố topic")
        st.plotly_chart(
            charts.topic_distribution_bar(topics, labels),
            width="stretch", config=charts.PLOT_CONFIG,
        )
    with right:
        st.subheader("Cấu hình run")
        key_value_panel([
            ("Thuật toán", str(overview["algorithm"])),
            ("Biểu diễn", str(overview["representation"])),
            ("Run ID", str(bundle.model_summary.run_id)),
            ("Tổng runtime", fmt_seconds(overview['runtime_seconds'])),
            ("Fit time", fmt_seconds(overview['fit_seconds'])),
            ("Representation time", fmt_seconds(overview['representation_seconds'])),
            ("Topic diversity", fmt_number(overview["topic_diversity"])),
        ])
        if overview["runtime_scope"]:
            st.caption(f"Phạm vi runtime: {overview['runtime_scope']}")
        if not overview["supports_outlier"]:
            note(
                "Mô hình này buộc mọi bài vào một topic nên chỉ số outlier "
                "không áp dụng; N/A không có nghĩa là 0%."
            )
        note(
            "Metric run có thể được đo trên tập khác với bộ lọc đang xem. Số bài, "
            "topic và outlier phía trên được tính trên split đã chọn."
        )

    st.subheader("Danh sách topic")
    display = topics[["topic_id", "topic_name", "document_count", "percentage", "top_keywords"]].copy()
    display["topic_name"] = [labels.get(int(t), "") for t in display["topic_id"]]
    display["percentage"] = display["percentage"] * 100
    table = display.rename(columns={
        "topic_id": "ID", "topic_name": "Chủ đề", "document_count": "Số bài",
        "percentage": "Tỷ lệ (%)", "top_keywords": "Từ khóa",
    })
    top_share = float(table["Tỷ lệ (%)"].max()) if not table.empty else 1.0
    data_table(table, {
        "ID": st.column_config.NumberColumn(width="small"),
        "Số bài": st.column_config.NumberColumn(format="%d"),
        "Tỷ lệ (%)": st.column_config.ProgressColumn(
            format="%.1f%%", min_value=0.0, max_value=max(top_share, 1.0),
        ),
    })
    table_download(table, f"{run.run_id}_{split}_topics.csv", "Tải bảng topic CSV")
