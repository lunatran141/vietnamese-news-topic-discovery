"""Các màn hình phân tích HDBSCAN từ common artifact."""

from __future__ import annotations

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import altair as alt
import pandas as pd
import streamlit as st

from app.ui.theme import OUTLIER_COLOR
from app.data_service import (
    filter_documents,
    get_hdbscan_split_summary,
    load_hdbscan_error_cases,
    load_hdbscan_experiment_comparison,
    load_hdbscan_human_reviews,
    load_hdbscan_projection,
    load_hdbscan_runtime,
    load_hdbscan_stability,
    load_hdbscan_transition_hard_cases,
    load_optional_table,
)


def _format_number(
    value: int | float | None,
    digits: int = 3,
) -> str:
    """Hiển thị N/A cho metric chưa được đo."""

    if value is None or pd.isna(value):
        return "N/A"

    if isinstance(value, int):
        return f"{value:,}"

    return f"{float(value):,.{digits}f}"


def _format_percentage(
    value: float | None,
) -> str:
    """Định dạng tỷ lệ và giữ N/A cho giá trị thiếu."""

    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value):.2%}"


def _render_quality_and_runtime(
    bundle,
    artifact_dir: str,
) -> None:
    """Hiển thị metric và runtime của run đại diện."""

    metrics = bundle.model_summary.metrics
    runtime = load_hdbscan_runtime(artifact_dir)

    train_validation = runtime.get(
        "train_validation",
        {},
    )
    test_runtime = runtime.get("test", {})

    umap_fit = train_validation.get(
        "umap_fit_seconds"
    )
    hdbscan_fit = train_validation.get(
        "hdbscan_fit_seconds"
    )

    fit_time = None
    if umap_fit is not None and hdbscan_fit is not None:
        fit_time = float(umap_fit) + float(hdbscan_fit)

    test_ari = test_runtime.get(
        "test_ari",
        metrics.get("test_ari"),
    )
    test_nmi = test_runtime.get(
        "test_nmi",
        metrics.get("test_nmi"),
    )
    test_coverage = test_runtime.get(
        "test_coverage",
        metrics.get("test_coverage"),
    )

    st.subheader("Chất lượng run đại diện")

    first_row = st.columns(4)

    first_row[0].metric(
        "Số bài",
        f"{len(bundle.document_topics):,}",
    )
    first_row[1].metric(
        "Số topic",
        f"{bundle.model_summary.topic_count:,}",
    )
    first_row[2].metric(
        "Test coverage",
        _format_percentage(test_coverage),
    )
    first_row[3].metric(
        "Topic diversity",
        _format_number(
            metrics.get("topic_diversity"),
            digits=4,
        ),
    )

    second_row = st.columns(4)

    second_row[0].metric(
        "Test ARI",
        _format_number(test_ari, digits=4),
    )
    second_row[1].metric(
        "Test NMI",
        _format_number(test_nmi, digits=4),
    )
    second_row[2].metric(
        "Coherence",
        "N/A",
        help=(
            "Run HDBSCAN này chưa đo coherence. "
            "Không thay giá trị thiếu bằng 0."
        ),
    )
    second_row[3].metric(
        "Fit time",
        (
            f"{fit_time:.2f} giây"
            if fit_time is not None
            else "N/A"
        ),
        help=(
            "Fit time = UMAP fit + HDBSCAN fit."
        ),
    )

    st.subheader("Runtime train và validation")

    train_runtime_rows = [
        {
            "Công đoạn": "Tải model",
            "Giây": train_validation.get(
                "model_load_seconds"
            ),
        },
        {
            "Công đoạn": "Biểu diễn văn bản",
            "Giây": train_validation.get(
                "representation_seconds"
            ),
        },
        {
            "Công đoạn": "UMAP fit",
            "Giây": umap_fit,
        },
        {
            "Công đoạn": "HDBSCAN fit",
            "Giây": hdbscan_fit,
        },
        {
            "Công đoạn": "Validation predict",
            "Giây": train_validation.get(
                "validation_predict_seconds"
            ),
        },
        {
            "Công đoạn": "Evaluation",
            "Giây": train_validation.get(
                "evaluation_seconds"
            ),
        },
        {
            "Công đoạn": "Pipeline runtime",
            "Giây": train_validation.get(
                "runtime_seconds"
            ),
        },
    ]

    st.dataframe(
        pd.DataFrame(train_runtime_rows),
        hide_index=True,
        width="stretch",
    )

    scope = train_validation.get("runtime_scope")

    if scope:
        st.caption(f"Runtime scope: {scope}")
    else:
        st.info("N/A — run chưa khai báo runtime scope.")

    measurement_note = runtime.get(
        "measurement_note"
    )

    if measurement_note:
        st.caption(measurement_note)

    st.subheader("Runtime test")

    test_runtime_rows = [
        {
            "Công đoạn": "Tải model",
            "Giây": test_runtime.get(
                "model_load_seconds"
            ),
        },
        {
            "Công đoạn": "Biểu diễn test",
            "Giây": test_runtime.get(
                "representation_seconds"
            ),
        },
        {
            "Công đoạn": "Test predict",
            "Giây": test_runtime.get(
                "test_predict_seconds"
            ),
        },
        {
            "Công đoạn": "Evaluation",
            "Giây": test_runtime.get(
                "evaluation_seconds"
            ),
        },
        {
            "Công đoạn": "Pipeline test",
            "Giây": test_runtime.get(
                "runtime_seconds"
            ),
        },
    ]

    st.dataframe(
        pd.DataFrame(test_runtime_rows),
        hide_index=True,
        width="stretch",
    )

    test_scope = test_runtime.get("runtime_scope")

    if test_scope:
        st.caption(f"Test runtime scope: {test_scope}")


def _render_coverage_and_outliers(
    bundle,
) -> None:
    """Hiển thị coverage và danh sách outlier."""

    st.subheader("Coverage theo split")

    split_summary = get_hdbscan_split_summary(
        bundle
    )

    st.dataframe(
        split_summary,
        hide_index=True,
        width="stretch",
        column_config={
            "coverage": st.column_config.NumberColumn(
                "Coverage",
                format="%.2f",
            ),
            "outlier_rate": (
                st.column_config.NumberColumn(
                    "Outlier rate",
                    format="%.2f",
                )
            ),
        },
    )

    st.subheader("Outlier explorer")

    documents = bundle.document_topics.copy()
    documents["is_outlier"] = (
        documents["is_outlier"]
        .astype(str)
        .str.lower()
        .isin({"true", "1"})
    )

    split_options = [
        "Tất cả",
        *documents["split"].dropna().unique().tolist(),
    ]

    selected_split = st.selectbox(
        "Split",
        split_options,
        key="hdbscan_outlier_split",
    )

    query = st.text_input(
        "Tìm tiêu đề hoặc article_id",
        key="hdbscan_outlier_query",
    )

    outliers = documents.loc[
        documents["is_outlier"]
    ].copy()

    if selected_split != "Tất cả":
        outliers = outliers.loc[
            outliers["split"].eq(selected_split)
        ]

    if query:
        matches = (
            outliers["article_id"]
            .astype(str)
            .str.contains(
                query,
                case=False,
                regex=False,
            )
        )

        if "title" in outliers.columns:
            matches |= (
                outliers["title"]
                .fillna("")
                .str.contains(
                    query,
                    case=False,
                    regex=False,
                )
            )

        outliers = outliers.loc[matches]

    if "membership_strength" in outliers.columns:
        outliers = outliers.sort_values(
            "membership_strength",
            ascending=True,
        )

    display_columns = [
        column
        for column in (
            "article_id",
            "split",
            "title",
            "topic_id",
            "membership_strength",
            "soft_primary_topic_id",
            "soft_primary_membership_strength",
            "second_topic_id",
            "second_membership_strength",
            "top_two_share_gap",
            "is_multi_topic",
            "category_gold",
        )
        if column in outliers.columns
    ]

    st.caption(
        f"{len(outliers):,} bài outlier phù hợp."
    )

    st.dataframe(
        outliers[display_columns],
        hide_index=True,
        width="stretch",
    )

    st.download_button(
        "Tải danh sách outlier đang lọc",
        outliers[display_columns]
        .to_csv(index=False)
        .encode("utf-8-sig"),
        file_name="hdbscan_filtered_outliers.csv",
        mime="text/csv",
    )


def _render_membership(
    bundle,
) -> None:
    """Hiển thị các bài có top-2 gần nhau."""

    st.subheader(
        "Full membership và bài chồng lấn"
    )

    documents = bundle.document_topics

    required_columns = {
        "second_topic_id",
        "second_membership_strength",
        "top_two_share_gap",
        "is_multi_topic",
    }

    if not required_columns.issubset(
        documents.columns
    ):
        st.info(
            "N/A — run này chưa xuất full "
            "membership vector."
        )
        return

    split_options = [
        "Tất cả",
        *documents["split"].dropna().unique().tolist(),
    ]

    selected_split = st.selectbox(
        "Split",
        split_options,
        key="hdbscan_membership_split",
    )

    gap_max = st.slider(
        "Top-2 gap tối đa",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.01,
        help=(
            "Gap càng nhỏ thì hai topic đứng đầu "
            "càng gần nhau."
        ),
    )

    multi_topic_only = st.checkbox(
        "Chỉ hiện bài được đánh dấu multi-topic",
        value=True,
        key="hdbscan_multi_only",
    )

    query = st.text_input(
        "Tìm bài trong full membership",
        key="hdbscan_membership_query",
    )

    split_value = (
        None
        if selected_split == "Tất cả"
        else selected_split
    )

    filtered = filter_documents(
        bundle,
        split=split_value,
        multi_topic_only=multi_topic_only,
        gap_max=gap_max,
        query=query,
    )

    filtered = filtered.sort_values(
        "top_two_share_gap",
        ascending=True,
        na_position="last",
    )

    display_columns = [
        column
        for column in (
            "article_id",
            "split",
            "title",
            "topic_id",
            "membership_strength",
            "soft_primary_topic_id",
            "soft_primary_membership_strength",
            "second_topic_id",
            "second_membership_strength",
            "top_two_share_gap",
            "is_multi_topic",
            "is_outlier",
            "category_gold",
        )
        if column in filtered.columns
    ]

    st.caption(
        f"{len(filtered):,} bài phù hợp."
    )

    st.dataframe(
        filtered[display_columns],
        hide_index=True,
        width="stretch",
    )

    st.caption(
        "`is_multi_topic` là heuristic với "
        "`top_two_share_gap <= 0.10`; "
        "membership strength không phải xác suất "
        "đã hiệu chuẩn."
    )


def _render_projection(
    bundle,
    artifact_dir: str,
) -> None:
    """Vẽ bản đồ UMAP 2D với tooltip đầy đủ."""

    st.subheader("Bản đồ HDBSCAN 2D")

    projection = load_hdbscan_projection(
        bundle,
        artifact_dir,
    )

    if projection.empty:
        st.info(
            "N/A — run này chưa có projection_2d.csv."
        )
        return

    split_values = (
        projection["split"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_splits = st.multiselect(
        "Split hiển thị",
        split_values,
        default=split_values,
        key="hdbscan_projection_splits",
    )

    points = projection.loc[
        projection["split"].isin(selected_splits)
    ].copy()

    if points.empty:
        st.info("Không có điểm phù hợp với bộ lọc.")
        return

    alt.data_transformers.disable_max_rows()

    tooltip = [
        alt.Tooltip(
            "article_id:N",
            title="Article ID",
        ),
        alt.Tooltip(
            "title:N",
            title="Tiêu đề",
        ),
        alt.Tooltip(
            "source:N",
            title="Nguồn",
        ),
        alt.Tooltip(
            "published_at:T",
            title="Ngày đăng",
        ),
        alt.Tooltip(
            "split:N",
            title="Split",
        ),
        alt.Tooltip(
            "topic_label:N",
            title="Topic",
        ),
        alt.Tooltip(
            "membership_strength:Q",
            title="Membership",
            format=".4f",
        ),
        alt.Tooltip(
            "top_two_share_gap:Q",
            title="Top-2 gap",
            format=".4f",
        ),
    ]

    assigned = points.loc[
        ~points["is_outlier"]
    ].copy()

    outliers = points.loc[
        points["is_outlier"]
    ].copy()

    layers = []

    if not assigned.empty:
        layers.append(
            alt.Chart(assigned)
            .mark_circle(
                size=45,
                opacity=0.65,
            )
            .encode(
                x=alt.X(
                    "x:Q",
                    title="UMAP x",
                ),
                y=alt.Y(
                    "y:Q",
                    title="UMAP y",
                ),
                color=alt.Color(
                    "topic_label:N",
                    title="Topic",
                    legend=None,
                    scale=alt.Scale(
                        scheme="turbo",
                    ),
                ),
                shape=alt.Shape(
                    "split:N",
                    title="Split",
                ),
                tooltip=tooltip,
            )
        )

    if not outliers.empty:
        layers.append(
            alt.Chart(outliers)
            .mark_circle(
                size=45,
                opacity=0.55,
                color=OUTLIER_COLOR,
            )
            .encode(
                x=alt.X(
                    "x:Q",
                    title="UMAP x",
                ),
                y=alt.Y(
                    "y:Q",
                    title="UMAP y",
                ),
                shape=alt.Shape(
                    "split:N",
                    title="Split",
                ),
                tooltip=tooltip,
            )
        )

    chart = (
        alt.layer(*layers)
        .properties(height=650)
        .interactive()
    )

    st.altair_chart(
        chart,
        width="stretch",
    )

    st.caption(
        "Màu biểu thị topic, ký hiệu biểu thị split "
        "và outlier có màu xám. UMAP 2D chỉ phục vụ "
        "trực quan hóa, không phải metric chất lượng."
    )


def _render_stability(
    artifact_dir: str,
) -> None:
    """Hiển thị stability theo seed."""

    st.subheader("Stability theo seed")

    stability = load_hdbscan_stability(
        artifact_dir
    )

    if stability.empty:
        st.info(
            "N/A — run này chưa có stability summary."
        )
        return

    st.dataframe(
        stability,
        hide_index=True,
        width="stretch",
    )

    score_columns = [
        column
        for column in (
            "ari_between_runs",
            "keyword_jaccard",
        )
        if column in stability.columns
    ]

    if score_columns:
        st.line_chart(
            stability.set_index("seed")[
                score_columns
            ]
        )

    rate_columns = [
        column
        for column in (
            "coverage",
            "outlier_rate",
        )
        if column in stability.columns
    ]

    if rate_columns:
        st.line_chart(
            stability.set_index("seed")[
                rate_columns
            ]
        )

    st.caption(
        "ARI giữa các run đo độ ổn định assignment. "
        "Keyword Jaccard đo mức giống nhau của các bộ "
        "từ khóa sau khi ghép topic."
    )


def _render_representative_documents(
    artifact_dir: str,
) -> None:
    """Hiển thị bài đại diện của các topic."""

    st.subheader("Bài đại diện")

    representatives = load_optional_table(
        artifact_dir,
        "representative_docs.csv",
    )

    if representatives.empty:
        st.info(
            "N/A — run này chưa có bài đại diện."
        )
        return

    topic_ids = sorted(
        representatives["topic_id"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    selected_topic = st.selectbox(
        "Topic",
        topic_ids,
        key="hdbscan_representative_topic",
    )

    selected = representatives.loc[
        representatives["topic_id"]
        .eq(selected_topic)
    ].sort_values("rank")

    st.dataframe(
        selected,
        hide_index=True,
        width="stretch",
    )


def _render_human_evaluation(
    artifact_dir: str,
) -> None:
    """Hiển thị human evaluation khi có kết quả."""

    st.subheader("Human evaluation")

    reviews = load_hdbscan_human_reviews(
        artifact_dir
    )

    if reviews.empty:
        st.info(
            "N/A — đang chờ kết quả đánh giá "
            "độc lập từ R1 và R2."
        )
        st.caption(
            "Không suy ra hoặc tự động điền các nhãn "
            "outlier_correct, topic_consistent và "
            "is_boundary_or_mixed."
        )
        return

    first_row = st.columns(3)

    first_row[0].metric(
        "Số phiếu",
        f"{len(reviews):,}",
    )

    reviewer_count = (
        reviews["reviewer"].nunique()
        if "reviewer" in reviews.columns
        else None
    )

    first_row[1].metric(
        "Người đánh giá",
        (
            f"{reviewer_count:,}"
            if reviewer_count is not None
            else "N/A"
        ),
    )

    completed_count = None

    if "review_status" in reviews.columns:
        completed_count = int(
            reviews["review_status"]
            .astype(str)
            .str.lower()
            .eq("completed")
            .sum()
        )

    first_row[2].metric(
        "Phiếu hoàn thành",
        (
            f"{completed_count:,}"
            if completed_count is not None
            else "N/A"
        ),
    )

    st.dataframe(
        reviews,
        hide_index=True,
        width="stretch",
    )

def _render_experiment_comparison(
    artifact_dir: str,
) -> None:
    """So sánh MiniLM/MPNet và EOM/Leaf trên validation."""

    st.subheader("So sánh cấu hình HDBSCAN")

    comparison = (
        load_hdbscan_experiment_comparison(
            artifact_dir
        )
    )

    if comparison.empty:
        st.info(
            "N/A — chưa tìm thấy dữ liệu so sánh "
            "MiniLM–MPNet và EOM–Leaf."
        )
        return

    st.caption(
        "Các metric trong bảng thuộc validation. "
        "UMAP và HDBSCAN được fit trên train; "
        "validation chỉ được transform và dự đoán."
    )

    display_columns = [
        "configuration",
        "embedding_run",
        "experiment",
        "min_cluster_size",
        "min_samples",
        "topic_count",
        "train_outlier_rate",
        "validation_coverage",
        "validation_outlier_rate",
        "validation_ari",
        "validation_nmi",
    ]

    st.dataframe(
        comparison[display_columns],
        hide_index=True,
        width="stretch",
        column_config={
            "configuration": "Cấu hình",
            "embedding_run": "Embedding run",
            "experiment": "Experiment",
            "min_cluster_size": (
                "Min cluster size"
            ),
            "min_samples": "Min samples",
            "topic_count": "Số topic",
            "train_outlier_rate": (
                st.column_config.NumberColumn(
                    "Train outlier",
                    format="%.2f",
                )
            ),
            "validation_coverage": (
                st.column_config.NumberColumn(
                    "Validation coverage",
                    format="%.2f",
                )
            ),
            "validation_outlier_rate": (
                st.column_config.NumberColumn(
                    "Validation outlier",
                    format="%.2f",
                )
            ),
            "validation_ari": (
                st.column_config.NumberColumn(
                    "Validation ARI",
                    format="%.4f",
                )
            ),
            "validation_nmi": (
                st.column_config.NumberColumn(
                    "Validation NMI",
                    format="%.4f",
                )
            ),
        },
    )

    st.subheader("Coverage và outlier")

    rate_chart_data = comparison[
        [
            "configuration",
            "validation_coverage",
            "validation_outlier_rate",
        ]
    ].set_index("configuration")

    st.bar_chart(rate_chart_data)

    st.subheader("ARI và NMI")

    metric_chart_data = comparison[
        [
            "configuration",
            "validation_ari",
            "validation_nmi",
        ]
    ].set_index("configuration")

    st.bar_chart(metric_chart_data)

    st.caption(
        "MiniLM–MPNet và EOM–Leaf được so sánh "
        "trên cùng validation split. Không sử dụng "
        "test để chọn lại cấu hình."
    )

def _render_hard_cases(
    bundle,
    artifact_dir: str,
) -> None:
    """Hiển thị hard cases thủ công và bài không ổn định theo seed."""

    st.subheader("Hard cases")

    reviewed_cases = load_hdbscan_error_cases(
        artifact_dir
    )

    transition_cases = (
        load_hdbscan_transition_hard_cases(
            bundle,
            artifact_dir,
        )
    )

    sections = st.tabs(
        [
            "Error analysis đã duyệt",
            "Không ổn định theo seed",
        ]
    )

    with sections[0]:
        if reviewed_cases.empty:
            st.info(
                "N/A — chưa có bảng hard cases "
                "đã phân tích."
            )
        else:
            error_groups = sorted(
                reviewed_cases["error_group"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_groups = st.multiselect(
                "Nhóm lỗi",
                error_groups,
                default=error_groups,
                key="hdbscan_error_groups",
            )

            split_values = sorted(
                reviewed_cases["split"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_splits = st.multiselect(
                "Split hard cases",
                split_values,
                default=split_values,
                key="hdbscan_error_splits",
            )

            filtered = reviewed_cases.loc[
                reviewed_cases[
                    "error_group"
                ].isin(selected_groups)
                & reviewed_cases[
                    "split"
                ].isin(selected_splits)
            ].copy()

            st.caption(
                f"{len(filtered):,} hard cases "
                "phù hợp."
            )

            overview_columns = [
                column
                for column in (
                    "case_id",
                    "split",
                    "embedding_run",
                    "experiment",
                    "article_id",
                    "cluster_id",
                    "error_group",
                    "review_label",
                    "title",
                )
                if column in filtered.columns
            ]

            st.dataframe(
                filtered[overview_columns],
                hide_index=True,
                width="stretch",
            )

            if not filtered.empty:
                case_ids = filtered[
                    "case_id"
                ].astype(str).tolist()

                selected_case_id = st.selectbox(
                    "Xem chi tiết hard case",
                    case_ids,
                    key="hdbscan_selected_hard_case",
                )

                selected = filtered.loc[
                    filtered["case_id"]
                    .astype(str)
                    .eq(selected_case_id)
                ].iloc[0]

                st.markdown(
                    f"### {selected_case_id}: "
                    f"{selected.get('title', '')}"
                )

                st.markdown(
                    f"""
- **Article ID:** `{selected.get("article_id", "")}`
- **Split:** `{selected.get("split", "")}`
- **Embedding:** `{selected.get("embedding_run", "")}`
- **Experiment:** `{selected.get("experiment", "")}`
- **Topic:** `{selected.get("cluster_id", "")}`
- **Nhóm lỗi:** `{selected.get("error_group", "")}`
"""
                )

                st.markdown("**Bằng chứng:**")
                st.write(
                    selected.get("evidence", "")
                )

                st.markdown("**Diễn giải:**")
                st.write(
                    selected.get(
                        "interpretation",
                        "",
                    )
                )

    with sections[1]:
        if transition_cases.empty:
            st.info(
                "N/A — chưa có dữ liệu chuyển trạng "
                "thái giữa các seed."
            )
        else:
            maximum_transition = int(
                transition_cases[
                    "transition_count"
                ].max()
            )

            minimum_transition = st.slider(
                "Số lần chuyển trạng thái tối thiểu",
                min_value=1,
                max_value=maximum_transition,
                value=min(
                    10,
                    maximum_transition,
                ),
                step=1,
            )

            split_options = [
                "Tất cả",
                *sorted(
                    transition_cases["split"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                ),
            ]

            selected_split = st.selectbox(
                "Split stability",
                split_options,
                key="hdbscan_transition_split",
            )

            unstable = transition_cases.loc[
                transition_cases[
                    "transition_count"
                ].ge(minimum_transition)
            ].copy()

            if selected_split != "Tất cả":
                unstable = unstable.loc[
                    unstable["split"].eq(
                        selected_split
                    )
                ]

            unstable = unstable.sort_values(
                "transition_count",
                ascending=False,
            )

            st.caption(
                f"{len(unstable):,} bài thay đổi "
                "giữa cluster và outlier."
            )

            display_columns = [
                column
                for column in (
                    "article_id",
                    "split",
                    "title",
                    "topic_id",
                    "transition_count",
                    "compared_seed_pairs",
                    "instability_rate",
                    "noise_to_cluster",
                    "cluster_to_noise",
                    "membership_strength",
                    "top_two_share_gap",
                    "is_multi_topic",
                    "is_outlier",
                    "category_gold",
                )
                if column in unstable.columns
            ]

            st.dataframe(
                unstable[display_columns],
                hide_index=True,
                width="stretch",
                column_config={
                    "instability_rate": (
                        st.column_config.NumberColumn(
                            "Instability rate",
                            format="%.2f",
                        )
                    ),
                },
            )

            st.caption(
                "Hard case stability là bài thay đổi "
                "giữa cluster và nhãn outlier -1 khi "
                "thay UMAP seed. Giá trị cao thể hiện "
                "assignment nhạy với seed."
            )

def render_hdbscan_details(
    bundle,
    artifact_dir: str,
) -> None:
    """Hiển thị dashboard riêng cho HDBSCAN."""

    tabs = st.tabs(
        [
            "Chất lượng và runtime",
            "Coverage và outlier",
            "Full membership",
            "Bản đồ 2D",
            "So sánh cấu hình",
            "Stability",
            "Hard cases",
            "Bài đại diện",
            "Human evaluation",
        ]
    )

    with tabs[0]:
        _render_quality_and_runtime(
            bundle,
            artifact_dir,
        )

    with tabs[1]:
        _render_coverage_and_outliers(bundle)

    with tabs[2]:
        _render_membership(bundle)

    with tabs[3]:
        _render_projection(
            bundle,
            artifact_dir,
        )

    with tabs[4]:
        _render_experiment_comparison(
            artifact_dir
        )

    with tabs[5]:
        _render_stability(artifact_dir)

    with tabs[6]:
        _render_hard_cases(
            bundle,
            artifact_dir,
        )

    with tabs[7]:
        _render_representative_documents(
            artifact_dir
        )

    with tabs[8]:
        _render_human_evaluation(
            artifact_dir
        )