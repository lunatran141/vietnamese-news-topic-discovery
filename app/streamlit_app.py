"""Giao diện khám phá chủ đề tin tức tiếng Việt."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import requests
import streamlit as st

from app.data_service import (
    get_model_overview,
    get_topic_table,
    load_dashboard_bundle,
)


API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://localhost:8000",
)

PASTEL_COLORS = [
    "#A8DADC",
    "#F4ACB7",
    "#CDB4DB",
    "#BDE0FE",
    "#FFC8DD",
    "#B8E0D2",
    "#D8B4F8",
    "#FFD6A5",
]


st.set_page_config(
    page_title="Vietnamese News Topic Discovery",
    page_icon="📰",
    layout="wide",
)


def format_percentage(
    value: float | None,
) -> str:
    """Định dạng tỷ lệ thành phần trăm."""
    if value is None:
        return "N/A"

    return f"{value:.1%}"


def format_number(
    value: int | float | None,
) -> str:
    """Định dạng số dùng trên metric card."""
    if value is None:
        return "N/A"

    if isinstance(value, float):
        return f"{value:,.2f}"

    return f"{value:,}"


def render_sidebar() -> None:
    """Hiển thị thông tin và kiểm tra API."""
    with st.sidebar:
        st.title("📰 Topic Discovery")

        st.caption(
            "Khám phá chủ đề tin tức tiếng Việt "
            "bằng BERTopic."
        )

        st.divider()

        st.subheader("Trạng thái API")

        if st.button(
            "Kiểm tra API",
            width="stretch",
        ):
            try:
                response = requests.get(
                    f"{API_BASE_URL}/health",
                    timeout=5,
                )
                response.raise_for_status()

                st.success("API đang hoạt động")
            except requests.RequestException:
                st.warning(
                    "Chưa kết nối được API. "
                    "Trang Overview vẫn dùng "
                    "artifact cục bộ."
                )

        st.divider()

        st.info(
            "Các điểm gần nhau trên bản đồ "
            "embedding thể hiện các bài báo "
            "có nội dung ngữ nghĩa tương đồng."
        )


def render_metric_cards(
    overview: dict,
) -> None:
    """Hiển thị các chỉ số quan trọng."""
    first_row = st.columns(4)

    first_row[0].metric(
        label="Số topic",
        value=format_number(
            overview["topic_count"]
        ),
        help=(
            "Số cụm chủ đề được BERTopic "
            "phát hiện, không tính topic -1."
        ),
    )

    first_row[1].metric(
        label="Tổng số bài",
        value=format_number(
            overview["document_count"]
        ),
        help=(
            "Tổng số bài trong train "
            "và validation."
        ),
    )

    first_row[2].metric(
        label="Bài được phân cụm",
        value=format_number(
            overview["assigned_document_count"]
        ),
        help=(
            "Số bài không bị HDBSCAN "
            "xem là nhiễu."
        ),
    )

    first_row[3].metric(
        label="Bài outlier",
        value=format_number(
            overview["outlier_document_count"]
        ),
        help=(
            "Các bài thuộc topic -1, chưa được "
            "HDBSCAN gán vào cụm ổn định."
        ),
    )

    second_row = st.columns(3)

    second_row[0].metric(
        label="Tỷ lệ outlier validation",
        value=format_percentage(
            overview["outlier_rate"]
        ),
        help=(
            "Giá trị thấp hơn cho thấy mô hình "
            "gán được nhiều bài validation hơn."
        ),
    )

    second_row[1].metric(
        label="Topic diversity",
        value=format_number(
            overview["topic_diversity"]
        ),
        help=(
            "Giá trị cao cho thấy từ khóa giữa "
            "các topic ít bị trùng lặp hơn."
        ),
    )

    runtime_seconds = overview["runtime_seconds"]

    second_row[2].metric(
        label="Thời gian chạy",
        value=(
            f"{runtime_seconds:,.0f} giây"
            if runtime_seconds is not None
            else "N/A"
        ),
        help="Tổng thời gian chạy thí nghiệm E1.",
    )


def render_topic_chart(
    topic_table,
) -> None:
    """Vẽ phân bố số bài của các topic lớn nhất."""
    top_topics = topic_table.head(15).copy()

    labels = [
        f"T{int(topic_id)}"
        for topic_id in top_topics["topic_id"]
    ]

    colors = [
        PASTEL_COLORS[
            index % len(PASTEL_COLORS)
        ]
        for index in range(len(top_topics))
    ]

    figure, axis = plt.subplots(
        figsize=(12, 5.5)
    )

    bars = axis.bar(
        labels,
        top_topics["document_count"],
        color=colors,
        edgecolor="#FFFFFF",
        linewidth=0.8,
    )

    axis.bar_label(
        bars,
        fmt="%d",
        padding=3,
        fontsize=8,
    )

    axis.set_title(
        "15 topic có nhiều bài báo nhất",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )
    axis.set_xlabel("Topic")
    axis.set_ylabel("Số bài báo")

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    figure.tight_layout()

    st.pyplot(
        figure,
        width="stretch",
    )

    plt.close(figure)

    st.caption(
        "Thanh càng cao cho thấy topic chứa "
        "càng nhiều bài. Phân bố quá lệch có thể "
        "cho thấy một số topic còn quá tổng quát."
    )


def render_topic_table(
    topic_table,
) -> None:
    """Hiển thị bảng topic dễ đọc."""
    display_table = topic_table[
        [
            "topic_id",
            "topic_name",
            "document_count",
            "percentage",
            "top_keywords",
        ]
    ].copy()

    display_table["percentage"] = (
        display_table["percentage"] * 100
    ).round(2)

    display_table = display_table.rename(
        columns={
            "topic_id": "Topic",
            "topic_name": "Tên topic",
            "document_count": "Số bài",
            "percentage": "Tỷ lệ (%)",
            "top_keywords": "Từ khóa đại diện",
        }
    )

    st.dataframe(
        display_table,
        width="stretch",
        hide_index=True,
        column_config={
            "Topic": st.column_config.NumberColumn(
                format="T%d",
            ),
            "Số bài": (
                st.column_config.NumberColumn(
                    format="%d",
                )
            ),
            "Tỷ lệ (%)": (
                st.column_config.NumberColumn(
                    format="%.2f%%",
                )
            ),
        },
    )


def main() -> None:
    """Hiển thị trang tổng quan BERTopic."""
    render_sidebar()

    st.title(
        "Khám phá chủ đề tin tức tiếng Việt"
    )

    st.caption(
        "Tổng quan kết quả BERTopic E1 — "
        "Vietnamese SBERT → UMAP → "
        "HDBSCAN → c-TF-IDF"
    )

    try:
        bundle = load_dashboard_bundle()
    except (FileNotFoundError, ValueError) as error:
        st.error(
            "Không thể đọc dữ liệu mô hình."
        )
        st.code(str(error))
        st.stop()

    overview = get_model_overview(bundle)
    topic_table = get_topic_table(bundle)

    st.subheader("Tổng quan mô hình")
    render_metric_cards(overview)

    st.divider()

    left_column, right_column = st.columns(
        [1.5, 1]
    )

    with left_column:
        st.subheader("Phân bố topic")
        render_topic_chart(topic_table)

    with right_column:
        st.subheader("Thông tin cấu hình")

        st.markdown(
            f"""
- **Mô hình:** {overview["model_name"]}
- **Thuật toán:** {overview["algorithm"]}
- **Embedding:** `{overview["representation"]}`
- **Run ID:** `{bundle.model_summary.run_id}`
- **Schema:** `{bundle.manifest.schema_version}`
"""
        )

        st.info(
            "Topic `-1` là các bài được HDBSCAN "
            "xác định là outlier. Các bài này "
            "không được tính vào bảng topic."
        )

    st.divider()

    st.subheader("Danh sách topic")

    st.caption(
        "Bảng được sắp xếp theo số bài giảm dần."
    )

    render_topic_table(topic_table)


if __name__ == "__main__":
    main()