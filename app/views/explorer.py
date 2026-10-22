"""Trang 2: Khám phá chủ đề (bản đồ và chi tiết, xu hướng, đổi tên và gộp)."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from app.data_service import (
    enrich_documents, get_document_table, get_documents_for_topic, get_keywords_for_topic,
    get_representatives_for_topic, get_topic_table, get_topic_trend, load_projection,
    load_representatives,
)
from app.ui import charts
from app.ui.components import (
    build_topic_labels, data_table, doc_row, fmt_number, keyword_pills, kpi_row, label_of,
    note, page_header, table_download,
)
from app.ui.context import AnalysisContext


def render(ctx: AnalysisContext) -> None:
    page_header(
        "Khám phá chủ đề",
        "Chọn một chủ đề để xem vị trí trên bản đồ, từ khóa, bài tiêu biểu và xu hướng.",
        {"Run": ctx.run.label, "Split": "Tất cả có sẵn" if ctx.split == "all" else ctx.split},
        eyebrow="Phân tích",
    )
    topics = get_topic_table(ctx.bundle, split=ctx.split)
    if topics.empty:
        st.info("Không có topic trên tập đang chọn.")
        return
    labels = build_topic_labels(ctx.bundle, ctx.run.run_id)
    map_tab, trend_tab, rename_tab = st.tabs(["Bản đồ và chi tiết", "Xu hướng thời gian", "Đổi tên và gộp"])
    with map_tab:
        _render_map_and_detail(ctx, topics, labels)
    with trend_tab:
        _render_trend(ctx, topics, labels)
    with rename_tab:
        _render_mapping(ctx, topics, labels)


def _render_map_and_detail(ctx: AnalysisContext, topics: pd.DataFrame, labels: dict[int, str]) -> None:
    ids = topics["topic_id"].astype(int).tolist()
    selected = st.selectbox(
        "Chủ đề đang xem", ids, format_func=lambda v: label_of(v, labels), key="explorer_topic",
    )
    map_col, detail_col = st.columns([6, 4])
    with map_col:
        _render_map(ctx, ids, selected, labels)
    with detail_col:
        _render_detail(ctx, topics, selected, labels)


def _render_map(ctx: AnalysisContext, ids: list[int], selected: int, labels: dict[int, str]) -> None:
    st.subheader("Bản đồ 2D")
    projection = load_projection(str(ctx.run.path))
    if projection.empty:
        st.info("Run này chưa xuất tọa độ 2D. Bản đồ sẽ hiện khi có projection_2d.csv theo article_id.")
        return
    if ctx.split != "all" and "split" in projection:
        projection = projection.loc[projection["split"] == ctx.split]
    available = {int(v) for v in projection["topic_id"].dropna().astype(int).unique() if v != -1}
    choices = [t for t in ids if t in available]
    default = list(dict.fromkeys(([selected] if selected in available else []) + choices[:8]))
    shown_ids = st.multiselect(
        "Chủ đề hiển thị trên bản đồ", choices, default=default,
        format_func=lambda v: label_of(v, labels), key=f"map_topics_{ctx.run.run_id}",
    )
    show_outliers = st.checkbox("Hiển thị outlier", value=False, key=f"map_outliers_{ctx.run.run_id}")
    keep = projection["topic_id"].isin(shown_ids)
    if show_outliers:
        keep |= projection["topic_id"] == -1
    points = projection.loc[keep]
    if points.empty:
        st.info("Chọn ít nhất một chủ đề có điểm.")
        return
    points = enrich_documents(points).copy()
    figure = charts.projection_scatter(
        points, labels, shown_ids, selected if selected in shown_ids else None, show_outliers,
    )
    st.plotly_chart(figure, width="stretch", config=charts.PLOT_CONFIG)
    note("Khoảng cách trên phép chiếu 2D chỉ hỗ trợ trực quan; không dùng riêng biểu đồ này để kết luận chất lượng mô hình.")


def _render_detail(ctx: AnalysisContext, topics: pd.DataFrame, selected: int, labels: dict[int, str]) -> None:
    run, bundle, split = ctx.run, ctx.bundle, ctx.split
    row = topics.loc[topics["topic_id"] == selected].iloc[0]
    st.subheader(label_of(selected, labels))
    kpi_row([
        ("Số bài", fmt_number(int(row["document_count"])), None),
        ("Tỷ lệ", f"{float(row['percentage']):.1%}", None),
    ])
    st.write("")
    keywords = get_keywords_for_topic(bundle, selected)
    keyword_pills(keywords["keyword"].head(10).astype(str).tolist())
    score_column = "weight" if "weight" in keywords.columns else ("score" if "score" in keywords.columns else None)
    if score_column and not keywords.empty:
        scored = keywords.head(10).copy()
        scored[score_column] = pd.to_numeric(scored[score_column], errors="coerce")
        st.plotly_chart(
            charts.keyword_bar(scored, score_column, "Điểm từ khóa"),
            width="stretch", config=charts.PLOT_CONFIG,
        )

    representatives = load_representatives(str(run.path))
    if not representatives.empty:
        st.markdown("**Bài đại diện đã xuất**")
        reps = get_representatives_for_topic(str(run.path), selected, split)
        if reps.empty:
            st.info("Run có bài đại diện nhưng chưa có bài ở split này.")
        else:
            for item in reps.head(5).to_dict("records"):
                doc_row(str(item.get("title", "")), f"{item.get('article_id', '')} · {item.get('split', '')}")
    else:
        st.markdown("**Bài gợi ý để xem**")
        st.caption("Sắp theo độ gần tâm cụm hoặc điểm gán topic; đây chưa phải danh sách representative_docs được duyệt.")
        examples = enrich_documents(get_documents_for_topic(bundle, selected, split)).head(5)
        for item in examples.to_dict("records"):
            doc_row(str(item.get("title", "")), f"{item.get('article_id', '')} · {item.get('source', '')}")

    documents = enrich_documents(get_documents_for_topic(bundle, selected, split))
    columns = ("article_id", "title", "source", "published_at", "split", "topic_probability", "url")
    shown = documents[[c for c in columns if c in documents]]
    with st.expander(f"Tất cả bài thuộc chủ đề ({len(shown):,})"):
        data_table(shown.head(300), height=320)
        st.caption(f"Hiển thị tối đa 300/{len(shown):,} bài; tải CSV để xem toàn bộ.")
        table_download(shown, f"{run.run_id}_{split}_topic_{selected}.csv", "Tải toàn bộ bài CSV")


def _render_trend(ctx: AnalysisContext, topics: pd.DataFrame, labels: dict[int, str]) -> None:
    documents = enrich_documents(get_document_table(ctx.bundle, ctx.split))
    if "published_at" not in documents or documents["published_at"].notna().sum() == 0:
        st.info("Chưa có ngày xuất bản hợp lệ cho run này.")
        return
    ids = topics["topic_id"].astype(int).tolist()
    current = st.session_state.get("explorer_topic")
    default = list(dict.fromkeys(([current] if current in ids else []) + ids[:3]))
    chosen = st.multiselect(
        "Chủ đề cần so sánh", ids, default=default,
        format_func=lambda v: label_of(v, labels), key=f"trend_topics_{ctx.run.run_id}",
    )
    frequency = st.radio("Gộp theo", ["Ngày", "Tuần"], horizontal=True, key="trend_frequency")
    code = "D" if frequency == "Ngày" else "W"
    series = {label_of(t, labels): get_topic_trend(documents, t, code) for t in chosen}
    st.plotly_chart(charts.trend_line(series), width="stretch", config=charts.PLOT_CONFIG)
    note("Đây là số bài trong corpus đã thu thập theo ngày xuất bản, không phải số lượng tin tức trên toàn thị trường.")
    frames = [f.assign(topic=name) for name, f in series.items() if not f.empty]
    if frames:
        table_download(pd.concat(frames, ignore_index=True), f"{ctx.run.run_id}_{ctx.split}_trend.csv", "Tải chuỗi thời gian CSV")


def _render_mapping(ctx: AnalysisContext, topics: pd.DataFrame, labels: dict[int, str]) -> None:
    run, bundle = ctx.run, ctx.bundle
    note("Tên và nhóm hiển thị được lưu trong phiên làm việc; artifact gốc không bị sửa. Tên mới được áp dụng cho mọi trang.")
    choices = topics["topic_id"].astype(int).tolist()
    key = f"mapping_{run.run_id}"
    if key not in st.session_state:
        st.session_state[key] = {}
    uploaded = st.file_uploader("Nạp mapping JSON đã lưu", type="json", key="mapping_upload")
    if uploaded is not None and st.button("Áp dụng mapping đã nạp", key="mapping_apply"):
        try:
            imported = json.load(uploaded)
            if imported.get("run_id") != run.run_id:
                raise ValueError("Run ID trong file không khớp run đang chọn")
            valid_ids = {str(value) for value in choices}
            entries = imported.get("mapping", {})
            if not isinstance(entries, dict) or not set(entries).issubset(valid_ids):
                raise ValueError("Mapping chứa topic không hợp lệ")
            if any(
                not isinstance(value, dict)
                or not isinstance(value.get("display_name", ""), str)
                or not isinstance(value.get("display_group_id"), (str, type(None)))
                for value in entries.values()
            ):
                raise ValueError("Nội dung mapping không đúng định dạng")
            st.session_state[key] = entries
            st.success("Đã nạp mapping cho run này.")
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            st.error(f"Không nạp được mapping: {error}")
    raw_names = dict(zip(bundle.topic_summary["topic_id"], bundle.topic_summary["topic_name"]))
    chosen = st.selectbox("Chủ đề cần đổi tên", choices, format_func=lambda v: label_of(v, labels), key="mapping_topic")
    current = st.session_state[key].get(str(chosen), {})
    with st.form("mapping_form"):
        new_name = st.text_input("Tên hiển thị", value=current.get("display_name", raw_names.get(chosen, "")))
        group = st.text_input("Nhóm hiển thị (tùy chọn)", value=current.get("display_group_id", "") or "")
        saved = st.form_submit_button("Lưu mapping trong phiên")
    if saved:
        st.session_state[key][str(chosen)] = {
            "display_name": new_name.strip() or raw_names.get(chosen, ""),
            "display_group_id": group.strip() or None,
        }
        st.toast("Đã lưu mapping hiển thị. Các topic cùng group chỉ được nhóm trên giao diện.")
        st.rerun()
    mapping = st.session_state[key]
    if mapping:
        display = pd.DataFrame([{"topic_id": int(topic_id), **value} for topic_id, value in mapping.items()])
        data_table(display)
        merged = topics.copy()
        merged["display_name"] = merged.apply(
            lambda r: mapping.get(str(int(r["topic_id"])), {}).get("display_name", r["topic_name"]), axis=1,
        )
        merged["display_group_id"] = merged["topic_id"].map(
            lambda v: mapping.get(str(int(v)), {}).get("display_group_id") or f"T{int(v)}"
        )
        groups = merged.groupby("display_group_id", as_index=False).agg(
            topic_ids=("topic_id", lambda values: ", ".join(f"T{v}" for v in values)),
            document_count=("document_count", "sum"),
        )
        st.subheader("Bảng nhóm hiển thị")
        data_table(groups)
        st.caption("Nhóm chỉ cộng số bài để trình bày; topic assignment và model gốc vẫn giữ nguyên.")
    payload = {"run_id": run.run_id, "schema_version": bundle.manifest.schema_version, "mapping": mapping}
    st.download_button(
        "Tải mapping JSON",
        json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        f"{run.run_id}_display_mapping.json", mime="application/json",
    )
