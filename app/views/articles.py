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
INFERENCE_MODELS = {
    "K-Means": "kmeans",
    "LDA": "lda",
    "NMF": "nmf",
    "HDBSCAN": "hdbscan",
    "BERTopic": "bertopic",
}


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


def _as_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).ne(0)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "1.0", "yes", "y", "t"})


def _is_outlier_mask(documents: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=documents.index)
    if "topic_id" in documents:
        mask = mask | pd.to_numeric(documents["topic_id"], errors="coerce").eq(-1)
    if "is_outlier" in documents:
        mask = mask | _as_bool_series(documents["is_outlier"])
    return mask


def _select_review_cases(documents: pd.DataFrame, selected: str) -> pd.DataFrame:
    outlier_mask = _is_outlier_mask(documents)
    if selected == "Outlier":
        return documents.loc[outlier_mask]
    non_outlier = documents.loc[~outlier_mask]
    if selected == "Bài đa chủ đề":
        if "is_multi_topic" not in non_outlier:
            return non_outlier.iloc[0:0]
        return non_outlier.loc[_as_bool_series(non_outlier["is_multi_topic"])]
    if selected == "Topic chồng lấn":
        if "top_two_share_gap" not in non_outlier:
            return non_outlier.iloc[0:0]
        gap = pd.to_numeric(non_outlier["top_two_share_gap"], errors="coerce")
        return non_outlier.loc[gap.le(0.1)]
    counts = non_outlier["topic_id"].value_counts() if "topic_id" in non_outlier else pd.Series(dtype=int)
    if counts.empty:
        return non_outlier.iloc[0:0]
    broad = counts.loc[counts >= counts.quantile(0.9)].index
    return non_outlier.loc[non_outlier["topic_id"].isin(broad)]


def _prepare_review_table(cases: pd.DataFrame, labels: dict[int, str]) -> pd.DataFrame:
    df = cases.copy()
    if "topic_id" in df:
        df["topic_label"] = df["topic_id"].map(
            lambda v: label_of(int(v), labels) if pd.notna(v) else ""
        )
    candidate_columns = [
        "article_id",
        "title",
        "topic_id",
        "topic_label",
        "category_gold",
        "split",
        "source",
        "published_at",
        "topic_probability",
        "topic_share",
        "distance_to_center",
        "soft_primary_topic_id",
        "soft_primary_membership_strength",
        "second_topic_id",
        "second_topic_probability",
        "second_topic_share",
        "second_membership_strength",
        "secondary_topic_id",
        "secondary_topic_share",
        "top_two_share_gap",
        "is_multi_topic",
    ]
    has_second_id = "second_topic_id" in df and df["second_topic_id"].notna().any()
    has_second_share = "second_topic_share" in df and df["second_topic_share"].notna().any()
    core_columns = {"article_id", "title", "topic_id", "topic_label"}
    selected_columns: list[str] = []
    for col in candidate_columns:
        if col not in df:
            continue
        if col == "secondary_topic_id" and has_second_id:
            continue
        if col == "secondary_topic_share" and has_second_share:
            continue
        if len(df) > 0 and col not in core_columns and not df[col].notna().any():
            continue
        selected_columns.append(col)
    return df[selected_columns]


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
        cases = _select_review_cases(documents, selected)
        note("Các bài có topic -1 theo artifact; cần đánh giá thủ công trước khi gọi là lỗi.")
    elif selected == "Bài đa chủ đề":
        if "is_multi_topic" not in documents:
            st.info("Run này chưa xuất dấu hiệu bài đa chủ đề.")
            return
        cases = _select_review_cases(documents, selected)
        note("Artifact của mô hình đánh dấu bài có nhiều topic đủ trọng số; đây là trường hợp cần xem xét, không tự động là lỗi.")
    elif selected == "Topic chồng lấn":
        if "top_two_share_gap" not in documents:
            st.info("Run này chưa xuất khoảng cách giữa hai topic mạnh nhất.")
            return
        cases = _select_review_cases(documents, selected)
        note("Hai topic mạnh nhất có tỷ trọng gần nhau (chênh lệch ≤ 0,1); cần đọc bài để xác định có chồng lấn thực sự hay không.")
    else:
        cases = _select_review_cases(documents, selected)
        if cases.empty:
            st.info("Không có topic để phân tích.")
            return
        note("Các topic thuộc nhóm 10% lớn nhất theo số bài. Kích thước lớn là tín hiệu để kiểm tra, chưa chứng minh topic sai.")
    left, right = st.columns(2)
    query = left.text_input("Tìm tiêu đề hoặc article ID", key="review_query").strip()
    available_topics = sorted(cases["topic_id"].dropna().astype(int).unique()) if "topic_id" in cases else []
    chosen_topics = right.multiselect(
        "Lọc theo chủ đề", available_topics, format_func=lambda v: label_of(v, labels), key="review_topics",
    )
    if query:
        title = cases["title"].fillna("").astype(str) if "title" in cases else pd.Series("", index=cases.index)
        cases = cases.loc[
            title.str.contains(query, case=False, regex=False)
            | cases["article_id"].astype(str).str.contains(query, case=False, regex=False)
        ]
    if chosen_topics:
        cases = cases.loc[cases["topic_id"].isin(chosen_topics)]
    kpi_row([("Số bài trong nhóm đang xem", f"{len(cases):,}", None)])
    st.write("")
    review_table = _prepare_review_table(cases, labels)
    data_table(review_table.head(300), height=420)
    st.caption("Hiển thị tối đa 300 bài trên bảng; file CSV tải về gồm toàn bộ danh sách đang lọc.")
    table_download(review_table, f"{ctx.run.run_id}_{ctx.split}_error_cases.csv", "Tải danh sách CSV")


def _render_new_article() -> None:
    note("Trang này dùng API nội bộ và model đã lưu. Không huấn luyện lại khi gửi bài.")
    with st.form("new_article_form"):
        model_label = st.selectbox("Mô hình dự đoán", list(INFERENCE_MODELS))
        title = st.text_input("Tiêu đề")
        content = st.text_area("Nội dung bài báo", height=220)
        submitted = st.form_submit_button("Phân tích bài báo")
    if not submitted:
        st.info(
            "Chọn mô hình và nhập tiêu đề hoặc nội dung để phân tích. "
            "Lần dự đoán đầu có thể lâu hơn do API cần nạp model vào bộ nhớ."
        )
        return
    if not title.strip() and not content.strip():
        st.warning("Nhập tiêu đề hoặc nội dung bài báo.")
        return
    try:
        with st.spinner("Đang tải model và phân tích bài báo..."):
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json={"title": title.strip(), "content": content.strip(), "model": INFERENCE_MODELS[model_label]},
                timeout=180,
            )
    except requests.RequestException as error:
        st.error(f"Không gọi được API: {error}")
        return
    if response.status_code in (404, 501, 503):
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = None
        st.warning(
            detail
            or "API dự đoán chưa sẵn sàng cho model đã lưu."
        )
        return
    try:
        response.raise_for_status()
        result = response.json()
    except (requests.RequestException, ValueError) as error:
        st.error(f"API trả về lỗi: {error}")
        return
    topic_id = result.get("topic_id")
    confidence = result.get("confidence")
    confidence_label = (
        f"{float(confidence):.1%}" if confidence is not None else "Không có"
    )
    kpi_row(
        [
            ("Topic", f"T{topic_id}", None),
            ("Mức gắn topic", confidence_label, None),
            (
                "Thời gian dự đoán",
                f"{float(result.get('inference_seconds', 0)):.3f} giây",
                None,
            ),
        ]
    )
    if result.get("is_outlier"):
        st.warning("Bài báo được xác định là outlier và chưa có topic đủ rõ.")
    else:
        st.success(f"Chủ đề dự đoán: {result.get('topic_name', f'Topic {topic_id}')}")
    st.caption(
        "Mức gắn topic (nếu có) là điểm tương đối của mô hình, không phải "
        "xác suất dự đoán đúng đã được hiệu chỉnh. K-Means không xuất điểm này. "
        f"Model: {result.get('model', 'BERTopic')} · "
        f"Run: {result.get('run_id', 'không xác định')} · "
        f"Tiền xử lý: {result.get('preprocessing_variant', 'không xác định')}."
    )

    keywords = result.get("keywords") or []
    if keywords:
        st.markdown("**Từ khóa giải thích:** " + " · ".join(map(str, keywords)))

    representatives = result.get("representative_articles") or []
    if representatives:
        st.markdown("**Bài báo đại diện của topic:**")
        representative_table = pd.DataFrame(representatives).rename(
            columns={
                "article_id": "Mã bài",
                "title": "Tiêu đề",
                "split": "Tập dữ liệu",
            }
        )
        data_table(representative_table, height=190)

    with st.expander("Xem JSON đầy đủ"):
        st.json(result)
    st.download_button(
        "Tải kết quả JSON",
        json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
        "new_article_prediction.json", mime="application/json",
    )
