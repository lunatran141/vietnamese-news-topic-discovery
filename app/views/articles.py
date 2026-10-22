"""Trang 3: Kho bài viết (tìm kiếm, trường hợp cần xem xét, phân loại bài mới)."""

from __future__ import annotations

import json
import os

import pandas as pd
import requests
import streamlit as st

from app.data_service import enrich_documents, get_document_table, get_model_overview
from app.ui.components import (
    build_topic_labels, data_table, kpi_row, label_of, note, page_header, table_download,
)
from app.ui.context import AnalysisContext

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
REVIEW_TYPES = ["Outlier", "Topic quá rộng", "Bài đa chủ đề", "Topic chồng lấn"]


def render(ctx: AnalysisContext) -> None:
    page_header(
        "Kho bài viết",
        "Tìm bài, rà soát các trường hợp cần xem xét và thử phân loại bài mới.",
        {"Run": ctx.run.label, "Split": "Tất cả có sẵn" if ctx.split == "all" else ctx.split},
        eyebrow="Phân tích",
    )
    labels = build_topic_labels(ctx.bundle, ctx.run.run_id)
    search_tab, review_tab, new_tab = st.tabs(["Tìm kiếm và lọc", "Trường hợp cần xem xét", "Phân loại bài mới"])
    with search_tab:
        _render_search(ctx, labels)
    with review_tab:
        _render_review(ctx, labels)
    with new_tab:
        _render_new_article()


def _filter_articles(ctx: AnalysisContext, labels: dict[int, str]) -> pd.DataFrame:
    documents = enrich_documents(get_document_table(ctx.bundle, ctx.split))
    left, middle, right = st.columns(3)
    query = left.text_input("Tìm tiêu đề hoặc article ID", key="search_query").strip()
    if query:
        title = documents["title"].fillna("").astype(str) if "title" in documents else pd.Series("", index=documents.index)
        documents = documents.loc[
            title.str.contains(query, case=False, regex=False)
            | documents["article_id"].astype(str).str.contains(query, case=False, regex=False)
        ]
    if "source" in documents:
        options = sorted(documents["source"].dropna().astype(str).unique())
        chosen = middle.multiselect("Nguồn báo", options, key="search_sources")
        if chosen:
            documents = documents.loc[documents["source"].isin(chosen)]
    topics = sorted(documents["topic_id"].dropna().astype(int).unique())
    chosen_topics = right.multiselect(
        "Chủ đề", topics, format_func=lambda v: label_of(v, labels), key="search_topics",
    )
    if chosen_topics:
        documents = documents.loc[documents["topic_id"].isin(chosen_topics)]
    if "published_at" in documents and documents["published_at"].notna().any():
        date = pd.to_datetime(
            documents["published_at"], errors="coerce", utc=True
        ).dt.tz_convert("Asia/Ho_Chi_Minh").dt.date
        available = date.dropna()
        if not available.empty:
            range_value = st.date_input("Khoảng ngày xuất bản", (available.min(), available.max()), key="search_dates")
            if isinstance(range_value, tuple) and len(range_value) == 2:
                documents = documents.loc[date.between(*range_value)]
    return documents


def _render_search(ctx: AnalysisContext, labels: dict[int, str]) -> None:
    note("Tìm trong các bài đã có artifact; bộ lọc nguồn và ngày lấy từ metadata corpus gốc.")
    documents = _filter_articles(ctx, labels)
    kpi_row([("Bài phù hợp", f"{len(documents):,}", None)])
    st.write("")
    columns = [c for c in ("article_id", "title", "topic_id", "source", "published_at", "split", "url") if c in documents]
    data_table(documents[columns].head(500), height=420)
    st.caption("Hiển thị tối đa 500 bài.")
    table_download(documents[columns], f"{ctx.run.run_id}_{ctx.split}_search.csv", "Tải kết quả CSV")


def _render_review(ctx: AnalysisContext, labels: dict[int, str]) -> None:
    documents = enrich_documents(get_document_table(ctx.bundle, ctx.split))
    selected = st.radio("Loại trường hợp", REVIEW_TYPES, horizontal=True, key="review_type")
    if selected == "Outlier":
        overview = get_model_overview(ctx.bundle, ctx.split, artifact_dir=ctx.run.path)
        if not overview["supports_outlier"]:
            st.info(
                "Outlier không áp dụng cho run này vì thuật toán buộc mọi "
                "bài vào một topic. Chỉ BERTopic/HDBSCAN hiện xuất nhãn -1."
            )
            return
        cases = documents.loc[documents["is_outlier"].fillna(False).astype(bool)]
        note("Các bài có topic -1 theo artifact; cần đánh giá thủ công trước khi gọi là lỗi.")
    elif selected == "Bài đa chủ đề":
        if "is_multi_topic" not in documents:
            st.info("Run này chưa xuất dấu hiệu bài đa chủ đề.")
            return
        cases = documents.loc[documents["is_multi_topic"].fillna(False).astype(bool)]
        note("Artifact của mô hình đánh dấu bài có nhiều topic đủ trọng số; đây là trường hợp cần xem xét, không tự động là lỗi.")
    elif selected == "Topic chồng lấn":
        if "top_two_share_gap" not in documents:
            st.info("Run này chưa xuất khoảng cách giữa hai topic mạnh nhất.")
            return
        gap = pd.to_numeric(documents["top_two_share_gap"], errors="coerce")
        cases = documents.loc[gap.le(0.1)]
        note("Hai topic mạnh nhất có tỷ trọng gần nhau (chênh lệch ≤ 0,1); cần đọc bài để xác định có chồng lấn thực sự hay không.")
    else:
        counts = documents.loc[documents["topic_id"] != -1, "topic_id"].value_counts()
        if counts.empty:
            st.info("Không có topic để phân tích.")
            return
        broad = counts.loc[counts >= counts.quantile(0.9)].index
        cases = documents.loc[documents["topic_id"].isin(broad)]
        note("Các topic thuộc nhóm 10% lớn nhất theo số bài. Kích thước lớn là tín hiệu để kiểm tra, chưa chứng minh topic sai.")
    kpi_row([("Số bài trong nhóm đang xem", f"{len(cases):,}", None)])
    st.write("")
    columns = [c for c in (
        "article_id", "title", "topic_id", "split", "source", "published_at", "topic_probability",
        "topic_share", "soft_primary_topic_id", "soft_primary_membership_strength",
        "secondary_topic_id", "secondary_topic_share", "second_topic_id", "second_topic_share",
        "second_membership_strength", "top_two_share_gap", "is_multi_topic",
    ) if c in cases]
    data_table(cases[columns].head(300), height=420)
    table_download(cases[columns], f"{ctx.run.run_id}_{ctx.split}_error_cases.csv", "Tải danh sách CSV")


def _render_new_article() -> None:
    note("Trang này dùng API nội bộ và model đã lưu. Không huấn luyện lại khi gửi bài.")
    with st.form("new_article_form"):
        title = st.text_input("Tiêu đề")
        content = st.text_area("Nội dung bài báo", height=220)
        submitted = st.form_submit_button("Phân tích bài báo")
    if not submitted:
        st.info("API hiện có `/health`; chức năng dự đoán chỉ sẵn sàng khi endpoint `/predict` cùng model inference được bàn giao.")
        return
    if not title.strip() and not content.strip():
        st.warning("Nhập tiêu đề hoặc nội dung bài báo.")
        return
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"title": title.strip(), "content": content.strip()},
            timeout=60,
        )
    except requests.RequestException as error:
        st.error(f"Không gọi được API: {error}")
        return
    if response.status_code in (404, 501, 503):
        st.warning("API dự đoán chưa được triển khai cho model đã lưu. Cần hoàn thành P10A trước khi demo bài mới.")
        return
    try:
        response.raise_for_status()
        result = response.json()
    except (requests.RequestException, ValueError) as error:
        st.error(f"API trả về lỗi: {error}")
        return
    st.json(result)
    st.download_button(
        "Tải kết quả JSON",
        json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
        "new_article_prediction.json", mime="application/json",
    )

