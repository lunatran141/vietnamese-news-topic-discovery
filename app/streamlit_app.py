"""Giao diện khám phá kết quả của năm phương pháp topic discovery."""

from __future__ import annotations

import json
import os
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
import yaml

from app.data_service import (
    ArtifactRun,
    discover_artifact_runs,
    enrich_documents,
    get_available_splits,
    get_document_table,
    get_documents_for_topic,
    get_keywords_for_topic,
    get_model_overview,
    get_representatives_for_topic,
    get_topic_table,
    get_topic_trend,
    load_dashboard_bundle,
    load_projection,
    load_representatives,
)
from app.nmf_dashboard import render_nmf_details
from app.hdbscan_dashboard import render_hdbscan_details

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
PASTEL_COLORS = [
    "#A8DADC", "#F4ACB7", "#CDB4DB", "#BDE0FE", "#FFC8DD",
    "#B8E0D2", "#D8B4F8", "#FFD6A5", "#AED9C7", "#F5D6A8",
    "#B6C8E3", "#E8C1D7", "#C4D9B4", "#B8C0E0", "#F1C6B7",
]
PAGES = [
    "Tổng quan", "Khám phá topic", "Tìm kiếm & lọc", "Bản đồ 2D",
    "Xu hướng", "So sánh mô hình", "Thí nghiệm", "Phân tích lỗi",
    "Đổi tên & gộp", "Bài báo mới", "Phương pháp & đóng góp",
]
CONTRIBUTIONS = [
    ("TV1 · Khôi", "Dữ liệu, TF-IDF, KMeans, preprocessing và input ablation", "kmeans"),
    ("TV2 · Tiến", "NMF, chọn số topic và phân tích bài đa chủ đề", "nmf"),
    ("TV3 · Tuấn", "LDA, stability nhiều seed và so sánh LDA–NMF", "lda"),
    ("TV4 · Vân", "Sentence embeddings, HDBSCAN và outlier analysis", "hdbscan"),
    ("TV5 · Ngân", "BERTopic, artifact chung và tích hợp Streamlit", "bertopic"),
]
FIGURES = {
    "kmeans": [
        ("Chọn K", "results/kmeans/figures/kmeans_elbow_inertia.png"),
        ("Ablation P0–P3", "results/kmeans/figures/m03_ablation_metrics_comparison.png"),
        ("Title-only và Title+Content", "results/kmeans/figures/m03a_metrics_comparison.png"),
    ],
    "nmf": [
        ("Chọn số topic", "results/nmf/nmf_p3_select_review_seed42_v1/topic_count_selection.png"),
    ],
    "lda": [],
    "hdbscan": [
        ("Số cụm", "results/hdbscan/figures/m11_cluster_count.png"),
        ("Outlier trên validation", "results/hdbscan/figures/m11_validation_noise_coverage.png"),
        ("ARI và NMI", "results/hdbscan/figures/m11_validation_ari_nmi.png"),
    ],
    "bertopic": [
        ("So sánh cấu hình", "results/bertopic_visualization/01_experiment_dashboard.png"),
        ("Title only và Title + Content", "results/bertopic_visualization/09_bertopic_title_content_ablation.png"),
        ("Từ khóa c-TF-IDF", "results/bertopic_visualization/06_e1_ctfidf_keywords.png"),
        ("Đánh giá cụm", "results/bertopic_visualization/08_e1_cluster_evaluation.png"),
    ],
}


st.set_page_config(
    page_title="Vietnamese News Topic Discovery", page_icon="📰", layout="wide"
)


def fmt_number(value: int | float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.3f}"


def fmt_percent(value: float | None) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{value:.1%}"


def topic_label(topic_id: int, names: dict[int, str]) -> str:
    return f"T{topic_id}: {names.get(topic_id, 'Chưa có tên')}"


def topic_names(bundle) -> dict[int, str]:
    return dict(zip(bundle.topic_summary["topic_id"], bundle.topic_summary["topic_name"]))


def table_download(frame: pd.DataFrame, filename: str, label: str) -> None:
    st.download_button(
        label,
        frame.to_csv(index=False).encode("utf-8-sig"),
        filename,
        mime="text/csv",
    )


def sidebar(runs: list[ArtifactRun]) -> tuple[str, ArtifactRun, object, str]:
    with st.sidebar:
        st.title("📰 Topic Discovery")
        st.caption("Khám phá kết quả từ 5 phương pháp trên cùng corpus tin tức.")
        page = st.radio("Trang", PAGES, label_visibility="collapsed")
        st.divider()
        preferred = next((i for i, run in enumerate(runs) if run.method == "bertopic"), 0)
        run = st.selectbox("Mô hình / run", runs, index=preferred, format_func=lambda x: x.label)
        try:
            bundle = load_dashboard_bundle(str(run.path))
        except (FileNotFoundError, ValueError) as error:
            st.error(f"Artifact của run này không hợp lệ: {error}")
            st.stop()
        splits = get_available_splits(bundle)
        split = st.selectbox("Tập dữ liệu", ["all", *splits], format_func=lambda x: "Tất cả có sẵn" if x == "all" else x)
        st.caption("Các mô hình có thể xuất những split khác nhau; bộ lọc này áp dụng cho trang đang xem.")
        st.divider()
        if st.button("Kiểm tra API", width="stretch"):
            try:
                response = requests.get(f"{API_BASE_URL}/health", timeout=5)
                response.raise_for_status()
                st.success("API đang hoạt động")
            except requests.RequestException:
                st.warning("Chưa kết nối được API; các trang artifact vẫn hoạt động.")
    return page, run, bundle, split


def render_header(page: str, run: ArtifactRun, split: str) -> None:
    st.title(page)
    st.caption(f"{run.label}  ·  Split: {split}  ·  Schema artifact 1.0")


def render_topic_chart(topics: pd.DataFrame) -> None:
    top = topics.head(15)
    if top.empty:
        st.info("Không có topic trên tập đang chọn.")
        return
    figure, axis = plt.subplots(figsize=(11, 4.5))
    labels = [f"T{int(value)}" for value in top["topic_id"]]
    bars = axis.bar(
        labels, top["document_count"],
        color=[PASTEL_COLORS[i % len(PASTEL_COLORS)] for i in range(len(top))],
    )
    axis.bar_label(bars, padding=2, fontsize=8)
    axis.set_ylabel("Số bài")
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def render_overview(run: ArtifactRun, bundle, split: str) -> None:
    overview = get_model_overview(bundle, split)
    topics = get_topic_table(bundle, split=split)
    cards = st.columns(5)
    for column, title, value in zip(
        cards,
        ("Số topic", "Số bài", "Đã gán topic", "Outlier", "Tỷ lệ outlier"),
        (
            fmt_number(overview["topic_count"]), fmt_number(overview["document_count"]),
            fmt_number(overview["assigned_document_count"]),
            fmt_number(overview["outlier_document_count"]), fmt_percent(overview["outlier_rate"]),
        ),
    ):
        column.metric(title, value)
    left, right = st.columns([1.7, 1])
    with left:
        st.subheader("Phân bố topic")
        render_topic_chart(topics)
    with right:
        st.subheader("Cấu hình run")
        st.write(f"**Thuật toán:** {overview['algorithm']}")
        st.write(f"**Biểu diễn:** {overview['representation']}")
        st.write(f"**Run ID:** `{bundle.model_summary.run_id}`")
        st.write(f"**Tổng runtime đã ghi nhận:** {fmt_number(overview['runtime_seconds'])} giây")
        st.write(f"**Fit time:** {fmt_number(overview['fit_seconds'])} giây")
        st.write(f"**Representation time:** {fmt_number(overview['representation_seconds'])} giây")
        if overview["runtime_scope"]:
            st.caption(f"Phạm vi runtime: {overview['runtime_scope']}")
        st.write(f"**Topic diversity được báo cáo:** {fmt_number(overview['topic_diversity'])}")
        if not overview["supports_outlier"]:
            st.caption(
                "Mô hình này buộc mọi bài vào một topic nên chỉ số outlier "
                "không áp dụng; N/A không có nghĩa là 0%."
            )
        st.caption("Metric run có thể được đo trên tập khác với bộ lọc đang xem. Số bài, topic và outlier phía trên được tính trên split đã chọn.")
    st.subheader("Danh sách topic")
    columns = ["topic_id", "topic_name", "document_count", "percentage", "top_keywords"]
    display = topics[columns].copy()
    display["percentage"] = (display["percentage"] * 100).round(2)
    st.dataframe(display, width="stretch", hide_index=True)
    table_download(display, f"{run.run_id}_{split}_topics.csv", "Tải bảng topic CSV")
    if run.method == "nmf":
        st.divider()
        st.subheader("Phân tích chi tiết NMF")
        render_nmf_details(bundle, str(run.path))
    if run.method == "hdbscan":
        st.divider()
        st.subheader("Phân tích chi tiết HDBSCAN")
        render_hdbscan_details(bundle, str(run.path))


def render_topic_detail(run: ArtifactRun, bundle, split: str) -> None:
    topics = get_topic_table(bundle, split=split)
    if topics.empty:
        st.info("Không có topic trên tập đang chọn.")
        return
    names = topic_names(bundle)
    ids = topics["topic_id"].astype(int).tolist()
    selected = st.selectbox("Chọn topic", ids, format_func=lambda value: topic_label(value, names))
    row = topics.loc[topics["topic_id"] == selected].iloc[0]
    st.metric("Số bài trên split đang xem", int(row["document_count"]))
    keywords = get_keywords_for_topic(bundle, selected)
    st.subheader("Từ khóa đại diện")
    st.write(" · ".join(keywords["keyword"].head(12).astype(str)) or "Chưa có từ khóa")
    with st.expander("Điểm từ khóa"):
        st.dataframe(keywords, width="stretch", hide_index=True)

    representatives = load_representatives(str(run.path))
    if not representatives.empty:
        reps = get_representatives_for_topic(str(run.path), selected, split)
        st.subheader("Bài đại diện đã xuất")
        if reps.empty:
            st.info("Run có bài đại diện nhưng chưa có bài ở split này.")
        else:
            st.dataframe(reps[[c for c in ("rank", "title", "article_id", "split") if c in reps]], width="stretch", hide_index=True)
    else:
        st.subheader("Bài gợi ý để xem")
        st.caption("Sắp theo độ gần tâm cụm hoặc điểm gán topic khi artifact có điểm; đây chưa phải danh sách representative_docs được duyệt.")
        examples = enrich_documents(get_documents_for_topic(bundle, selected, split)).head(5)
        st.dataframe(examples[[c for c in ("title", "article_id", "split", "topic_probability") if c in examples]], width="stretch", hide_index=True)

    st.subheader("Các bài thuộc topic")
    documents = enrich_documents(get_documents_for_topic(bundle, selected, split))
    shown = documents[[c for c in ("article_id", "title", "source", "published_at", "split", "topic_probability", "url") if c in documents]]
    st.dataframe(shown.head(300), width="stretch", hide_index=True)
    st.caption(f"Hiển thị tối đa 300/{len(shown):,} bài; tải CSV để xem toàn bộ.")
    table_download(shown, f"{run.run_id}_{split}_topic_{selected}.csv", "Tải toàn bộ bài CSV")


def filter_documents(bundle, split: str) -> pd.DataFrame:
    documents = enrich_documents(get_document_table(bundle, split))
    left, middle, right = st.columns(3)
    query = left.text_input("Tìm tiêu đề hoặc article ID").strip()
    if query:
        title = documents["title"].fillna("").astype(str) if "title" in documents else pd.Series("", index=documents.index)
        documents = documents.loc[
            title.str.contains(query, case=False, regex=False)
            | documents["article_id"].astype(str).str.contains(query, case=False, regex=False)
        ]
    if "source" in documents:
        options = sorted(documents["source"].dropna().astype(str).unique())
        chosen = middle.multiselect("Nguồn báo", options)
        if chosen:
            documents = documents.loc[documents["source"].isin(chosen)]
    topics = sorted(documents["topic_id"].dropna().astype(int).unique())
    chosen_topics = right.multiselect("Topic", topics, format_func=lambda value: f"T{value}")
    if chosen_topics:
        documents = documents.loc[documents["topic_id"].isin(chosen_topics)]
    if "published_at" in documents and documents["published_at"].notna().any():
        date = pd.to_datetime(
            documents["published_at"], errors="coerce", utc=True
        ).dt.tz_convert("Asia/Ho_Chi_Minh").dt.date
        available = date.dropna()
        if not available.empty:
            range_value = st.date_input("Khoảng ngày xuất bản", (available.min(), available.max()))
            if isinstance(range_value, tuple) and len(range_value) == 2:
                documents = documents.loc[date.between(*range_value)]
    return documents


def render_search(run: ArtifactRun, bundle, split: str) -> None:
    st.caption("Tìm trong các bài đã có artifact; bộ lọc nguồn và ngày lấy từ metadata corpus gốc.")
    documents = filter_documents(bundle, split)
    st.metric("Bài phù hợp", len(documents))
    columns = [c for c in ("article_id", "title", "topic_id", "source", "published_at", "split", "url") if c in documents]
    st.dataframe(documents[columns].head(500), width="stretch", hide_index=True)
    st.caption("Hiển thị tối đa 500 bài.")
    table_download(documents[columns], f"{run.run_id}_{split}_search.csv", "Tải kết quả CSV")


def render_map(run: ArtifactRun, bundle, split: str) -> None:
    projection = load_projection(str(run.path))
    if projection.empty:
        st.info("Run này chưa xuất tọa độ 2D. Bản đồ sẽ hiện khi có projection_2d.csv theo article_id.")
        return
    if split != "all" and "split" in projection:
        projection = projection.loc[projection["split"] == split]
    choices = sorted(value for value in projection["topic_id"].dropna().astype(int).unique() if value != -1)
    selected = st.multiselect("Lọc topic", choices, default=choices[:8], format_func=lambda value: f"T{value}")
    shown = projection.loc[projection["topic_id"].isin(selected)]
    if st.checkbox("Hiển thị outlier", value=False):
        shown = pd.concat([shown, projection.loc[projection["topic_id"] == -1]], ignore_index=True)
    if shown.empty:
        st.info("Chọn ít nhất một topic có điểm.")
        return
    shown = enrich_documents(shown).copy()
    shown["topic_label"] = shown["topic_id"].map(
        lambda value: "Outlier" if value == -1 else f"T{int(value)}"
    )
    domain = [f"T{value}" for value in selected]
    colors = [PASTEL_COLORS[i % len(PASTEL_COLORS)] for i in range(len(domain))]
    if (shown["topic_id"] == -1).any():
        domain.append("Outlier")
        colors.append("#667085")
    tooltip = ["article_id:N", "topic_label:N"]
    if "title" in shown:
        tooltip.insert(1, "title:N")
    if "split" in shown:
        tooltip.append("split:N")
    alt.data_transformers.disable_max_rows()
    chart = (
        alt.Chart(shown)
        .mark_circle(size=43, opacity=0.55)
        .encode(
            x=alt.X("x:Q", title="Trục 1"),
            y=alt.Y("y:Q", title="Trục 2"),
            color=alt.Color(
                "topic_label:N", title="Topic",
                scale=alt.Scale(domain=domain, range=colors),
            ),
            tooltip=tooltip,
        )
        .properties(height=570)
        .interactive()
    )
    st.altair_chart(chart, width="stretch")
    st.caption("Khoảng cách trên phép chiếu 2D chỉ hỗ trợ trực quan; không dùng riêng biểu đồ này để kết luận chất lượng mô hình.")


def render_trend(run: ArtifactRun, bundle, split: str) -> None:
    documents = enrich_documents(get_document_table(bundle, split))
    if "published_at" not in documents or documents["published_at"].notna().sum() == 0:
        st.info("Chưa có ngày xuất bản hợp lệ cho run này.")
        return
    names = topic_names(bundle)
    choices = get_topic_table(bundle, split=split)["topic_id"].astype(int).tolist()
    if not choices:
        st.info("Chưa có topic ở split này.")
        return
    selected = st.selectbox("Topic", choices, format_func=lambda value: topic_label(value, names))
    frequency = st.radio("Gộp theo", ["Ngày", "Tuần"], horizontal=True)
    trend = get_topic_trend(documents, selected, "D" if frequency == "Ngày" else "W")
    if trend.empty:
        st.info("Topic này chưa có bài có ngày xuất bản hợp lệ.")
        return
    st.line_chart(trend.set_index("period")["count"], x_label="Thời gian", y_label="Số bài")
    st.caption("Đây là số bài trong corpus đã thu thập theo ngày xuất bản, không phải số lượng tin tức trên toàn thị trường.")
    table_download(trend, f"{run.run_id}_{split}_topic_{selected}_trend.csv", "Tải chuỗi thời gian CSV")


def comparison_table(runs: list[ArtifactRun], split: str) -> pd.DataFrame:
    rows = []
    for run in runs:
        bundle = load_dashboard_bundle(str(run.path))
        available = get_available_splits(bundle)
        if split != "all" and split not in available:
            continue
        overview = get_model_overview(bundle, split)
        rows.append({
            "Mô hình": bundle.model_summary.algorithm,
            "Run": bundle.model_summary.run_id,
            "Split": split if split != "all" else ", ".join(available),
            "Số bài": overview["document_count"],
            "Số topic": overview["topic_count"],
            "Outlier áp dụng": "Có" if overview["supports_outlier"] else "Không",
            "Outlier (%)": (
                round(100 * overview["outlier_rate"], 2)
                if overview["outlier_rate"] is not None
                else None
            ),
            "Topic diversity (run)": overview["topic_diversity"],
            "ARI (run)": overview["adjusted_rand_index"],
            "NMI (run)": overview["normalized_mutual_info"],
            "Coherence (run)": overview["coherence"],
            "Fit (giây, run)": overview["fit_seconds"],
            "Runtime pipeline (giây, run)": overview["runtime_seconds"],
            "Biểu diễn": overview["representation"],
        })
    return pd.DataFrame(rows)


def render_comparison(runs: list[ArtifactRun], current: ArtifactRun) -> None:
    st.caption("Chọn cùng một split để đối chiếu số bài và số topic. Metric run vẫn giữ đúng phạm vi mà nhóm tác giả đã báo cáo.")
    split = st.selectbox("Split dùng để so sánh", ["train", "val", "test", "all"])
    table = comparison_table(runs, split)
    if table.empty:
        st.info("Không có run nào xuất split đã chọn.")
        return
    st.dataframe(table, width="stretch", hide_index=True)
    table_download(table, f"model_comparison_{split}.csv", "Tải bảng so sánh CSV")
    if split == "all":
        st.warning("Mỗi mô hình có phạm vi split khác nhau; hàng 'all' dùng để khám phá, không dùng làm kết luận so sánh định lượng.")
    st.info("ARI/NMI dùng category_gold rộng; coherence, topic diversity, runtime và outlier có định nghĩa/phạm vi khác nhau giữa các phương pháp. Ô N/A nghĩa là run chưa xuất metric, không phải giá trị bằng 0.")
    st.subheader("Xem hai run cạnh nhau")
    options = [run for run in runs if split == "all" or split in get_available_splits(load_dashboard_bundle(str(run.path)))]
    if len(options) < 2:
        return
    first = st.selectbox("Run A", options, format_func=lambda run: run.label, index=options.index(current) if current in options else 0)
    second = st.selectbox("Run B", options, format_func=lambda run: run.label, index=1 if len(options) > 1 else 0)
    cols = st.columns(2)
    for column, run in zip(cols, (first, second)):
        bundle = load_dashboard_bundle(str(run.path))
        with column:
            st.markdown(f"**{run.label}**")
            st.dataframe(get_topic_table(bundle, split=split).head(8)[["topic_id", "topic_name", "document_count"]], width="stretch", hide_index=True)
            with st.expander("Cấu hình run"):
                st.json(bundle.model_summary.config)


def render_errors(run: ArtifactRun, bundle, split: str) -> None:
    documents = enrich_documents(get_document_table(bundle, split))
    types = ["Outlier", "Topic quá rộng", "Bài đa chủ đề", "Topic chồng lấn"]
    selected = st.radio("Loại trường hợp", types, horizontal=True)
    if selected == "Outlier":
        overview = get_model_overview(bundle, split)
        if not overview["supports_outlier"]:
            st.info(
                "Outlier không áp dụng cho run này vì thuật toán buộc mọi "
                "bài vào một topic. Chỉ BERTopic/HDBSCAN hiện xuất nhãn -1."
            )
            return
        cases = documents.loc[documents["is_outlier"].fillna(False).astype(bool)]
        st.caption("Các bài có topic -1 theo artifact; cần đánh giá thủ công trước khi gọi là lỗi.")
    elif selected == "Bài đa chủ đề":
        if "is_multi_topic" not in documents:
            st.info("Run này chưa xuất dấu hiệu bài đa chủ đề.")
            return
        cases = documents.loc[documents["is_multi_topic"].fillna(False).astype(bool)]
        st.caption("Artifact của mô hình đánh dấu bài có nhiều topic đủ trọng số; đây là trường hợp cần xem xét, không tự động là lỗi.")
    elif selected == "Topic chồng lấn":
        if "top_two_share_gap" not in documents:
            st.info("Run này chưa xuất khoảng cách giữa hai topic mạnh nhất.")
            return
        gap = pd.to_numeric(documents["top_two_share_gap"], errors="coerce")
        cases = documents.loc[gap.le(0.1)]
        st.caption("Hai topic mạnh nhất có tỷ trọng gần nhau (chênh lệch ≤ 0,1); cần đọc bài để xác định có chồng lấn thực sự hay không.")
    else:
        counts = documents.loc[documents["topic_id"] != -1, "topic_id"].value_counts()
        if counts.empty:
            st.info("Không có topic để phân tích.")
            return
        threshold = counts.quantile(0.9)
        broad = counts.loc[counts >= threshold].index
        cases = documents.loc[documents["topic_id"].isin(broad)]
        st.caption("Các topic thuộc nhóm 10% lớn nhất theo số bài. Kích thước lớn là tín hiệu để kiểm tra, chưa chứng minh topic sai.")
    st.metric("Số bài trong nhóm đang xem", len(cases))
    columns = [c for c in (
        "article_id", "title", "topic_id", "split", "source",
        "published_at", "topic_probability", "topic_share",
        "soft_primary_topic_id", "soft_primary_membership_strength",
        "secondary_topic_id", "secondary_topic_share", "second_topic_id",
        "second_topic_share", "second_membership_strength",
        "top_two_share_gap", "is_multi_topic",
    ) if c in cases]
    st.dataframe(cases[columns].head(300), width="stretch", hide_index=True)
    table_download(cases[columns], f"{run.run_id}_{split}_error_cases.csv", "Tải danh sách CSV")


def render_experiments(run: ArtifactRun) -> None:
    st.caption("Hình và bảng kết quả đã được nhóm xuất trong repo. Các phép đo có protocol riêng cho từng phương pháp.")
    for title, relative_path in FIGURES.get(run.method, []):
        path = Path(__file__).resolve().parents[1] / relative_path
        if path.is_file():
            st.subheader(title)
            st.image(str(path), width="stretch")
    tables = {
        "kmeans": ["results/kmeans/kmeans_k_search_metrics.csv", "results/kmeans/m03_preprocessing_ablation_metrics.csv", "results/kmeans/m03a_title_vs_content_metrics.csv"],
        "nmf": [
            "results/nmf/nmf_p3_select_review_seed42_v1/topic_count_metrics.csv",
            "results/nmf/nmf_p3_k10_seed42_dashboard_v2/split_metrics.csv",
            "results/nmf/nmf_p3_k10_seed42_dashboard_v2/runtime_by_stage.csv",
            "results/nmf/nmf_p3_k10_seed42_dashboard_v2/multi_topic_cases.csv",
        ],
        "lda": ["results/lda/m08/topic_number_results.csv", "results/lda/m09/stability_metrics.csv", "results/lda/m09/runtime_by_seed.csv"],
        "hdbscan": [
            "results/hdbscan/m11_summary.csv",
            "results/hdbscan/validation/m11_validation_metrics.csv",
            "results/hdbscan/outlier_analysis/m12_outlier_rates.csv",
            "results/hdbscan/stability/minilm_title_content_eom_15_5_seeds42_51/stability_summary.csv",
            "results/hdbscan/stability/minilm_title_content_eom_15_5_seeds42_51/pairwise_keyword_jaccard.csv",
            "results/hdbscan/membership/minilm_title_content_eom_15_5/membership_summary.csv",
        ],
        "bertopic": ["results/bertopic_experiments/summary.csv"],
    }
    for relative_path in tables.get(run.method, []):
        path = Path(__file__).resolve().parents[1] / relative_path
        if not path.is_file():
            continue
        with st.expander(path.stem.replace("_", " ").title()):
            frame = pd.read_csv(path, encoding="utf-8-sig")
            if run.method == "bertopic" and path.name == "summary.csv":
                display_columns = [
                    "run_id", "input_text_label", "topic_count", "train_outlier_rate",
                    "validation_outlier_rate", "topic_diversity",
                    "validation_adjusted_rand_index",
                    "validation_normalized_mutual_info",
                    "validation_npmi_coherence",
                    "test_outlier_rate", "test_adjusted_rand_index",
                    "test_normalized_mutual_info", "test_npmi_coherence",
                ]
                labels = {
                    "run_id": "Run",
                    "input_text_label": "Đầu vào",
                    "topic_count": "Số topic",
                    "train_outlier_rate": "Outlier train",
                    "validation_outlier_rate": "Outlier validation",
                    "topic_diversity": "Diversity",
                    "validation_adjusted_rand_index": "ARI validation",
                    "validation_normalized_mutual_info": "NMI validation",
                    "validation_npmi_coherence": "NPMI validation",
                    "test_outlier_rate": "Outlier test",
                    "test_adjusted_rand_index": "ARI test",
                    "test_normalized_mutual_info": "NMI test",
                    "test_npmi_coherence": "NPMI test",
                }
                compact = frame[
                    [column for column in display_columns if column in frame]
                ].rename(columns=labels)
                for percentage_column in (
                    "Outlier train",
                    "Outlier validation",
                    "Outlier test",
                ):
                    if percentage_column in compact:
                        compact[percentage_column] = (
                            100 * compact[percentage_column]
                        )
                st.markdown("**Chất lượng và khả năng bao phủ**")
                st.dataframe(
                    compact,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Outlier train": st.column_config.NumberColumn(format="%.1f%%"),
                        "Outlier validation": st.column_config.NumberColumn(format="%.1f%%"),
                        "Diversity": st.column_config.NumberColumn(format="%.3f"),
                        "ARI validation": st.column_config.NumberColumn(format="%.3f"),
                        "NMI validation": st.column_config.NumberColumn(format="%.3f"),
                        "NPMI validation": st.column_config.NumberColumn(format="%.3f"),
                        "Outlier test": st.column_config.NumberColumn(format="%.1f%%"),
                        "ARI test": st.column_config.NumberColumn(format="%.3f"),
                        "NMI test": st.column_config.NumberColumn(format="%.3f"),
                        "NPMI test": st.column_config.NumberColumn(format="%.3f"),
                    },
                )
                timing_columns = [
                    "run_id", "model_load_seconds",
                    "representation_seconds", "model_init_seconds",
                    "fit_seconds", "validation_predict_seconds",
                    "test_predict_seconds",
                    "evaluation_seconds", "runtime_seconds",
                ]
                timing_labels = {
                    "run_id": "Run",
                    "model_load_seconds": "Load model",
                    "representation_seconds": "Embedding",
                    "model_init_seconds": "Khởi tạo",
                    "fit_seconds": "Fit",
                    "validation_predict_seconds": "Predict validation",
                    "test_predict_seconds": "Predict test",
                    "evaluation_seconds": "Evaluation",
                    "runtime_seconds": "Runtime pipeline",
                }
                timings = frame[
                    [column for column in timing_columns if column in frame]
                ].rename(columns=timing_labels)
                st.markdown("**Thời gian theo giai đoạn (giây)**")
                st.dataframe(timings, width="stretch", hide_index=True)
                st.caption(
                    "N/A nghĩa là run cũ chưa được đo lại theo protocol mới. "
                    "Không được diễn giải N/A là 0 giây."
                )
            else:
                st.dataframe(frame.head(100), width="stretch", hide_index=True)
            st.caption(f"Hiển thị tối đa 100/{len(frame):,} hàng.")
            table_download(frame, path.name, "Tải bảng CSV")
    if run.method == "bertopic":
        config_path = Path(__file__).resolve().parents[1] / "configs/bertopic_experiments.yaml"
        with config_path.open("r", encoding="utf-8") as file:
            experiment_config = yaml.safe_load(file)
        config_rows = []
        for item in experiment_config["runs"]:
            config_rows.append(
                {
                    "Run": item["run_id"],
                    "Đầu vào": item.get("input_text_label", "title_content"),
                    "Embedding": item["embedding"]["model_name"],
                    "UMAP neighbors": item["umap"]["n_neighbors"],
                    "UMAP components": item["umap"]["n_components"],
                    "UMAP min_dist": item["umap"]["min_dist"],
                    "UMAP metric": item["umap"]["metric"],
                    "HDBSCAN min cluster": item["hdbscan"]["min_cluster_size"],
                    "HDBSCAN min samples": item["hdbscan"]["min_samples"],
                    "Selection": item["hdbscan"]["cluster_selection_method"],
                }
            )
        st.subheader("Cấu hình UMAP và HDBSCAN")
        st.dataframe(pd.DataFrame(config_rows), width="stretch", hide_index=True)
        st.caption(
            "E2 chỉ đổi UMAP n_neighbors; E3 chỉ đổi HDBSCAN "
            "min_cluster_size; E4 chỉ đổi đầu vào sang title-only. "
            "E1 được chọn bằng validation trước khi mở test."
        )

        bundle = load_dashboard_bundle(str(run.path))
        topics = get_topic_table(bundle).head(15)
        if not topics.empty:
            names = topic_names(bundle)
            topic_id = st.selectbox(
                "Xem từ khóa c-TF-IDF của topic",
                topics["topic_id"].astype(int).tolist(),
                format_func=lambda value: topic_label(value, names),
                key="experiment_ctfidf_topic",
            )
            keywords = get_keywords_for_topic(bundle, topic_id).head(15)
            if not keywords.empty:
                score_column = (
                    "weight" if "weight" in keywords.columns else "score"
                )
                keywords[score_column] = pd.to_numeric(
                    keywords[score_column],
                    errors="coerce",
                )
                keyword_chart = (
                    alt.Chart(keywords)
                    .mark_bar(color="#CDB4DB")
                    .encode(
                        x=alt.X(
                            f"{score_column}:Q",
                            title="Điểm c-TF-IDF",
                            scale=alt.Scale(zero=True),
                        ),
                        y=alt.Y("keyword:N", sort="-x", title="Từ khóa"),
                        tooltip=[
                            "keyword:N",
                            alt.Tooltip(
                                f"{score_column}:Q",
                                title="Điểm c-TF-IDF",
                                format=".4f",
                            ),
                        ],
                    )
                    .properties(height=380)
                )
                st.altair_chart(keyword_chart, width="stretch")
                st.dataframe(keywords, width="stretch", hide_index=True)
                st.caption(
                    "c-TF-IDF cao nghĩa là từ đặc trưng cho topic so với các "
                    "topic khác; không phải xác suất xuất hiện của từ."
                )
    if not FIGURES.get(run.method) and not tables.get(run.method):
        st.info("Chưa có hình hoặc bảng thí nghiệm được liên kết cho run này.")


def render_mapping(run: ArtifactRun, bundle, split: str) -> None:
    st.caption("Tên và nhóm hiển thị được lưu trong phiên làm việc; artifact gốc không bị sửa.")
    names = topic_names(bundle)
    choices = get_topic_table(bundle, split=split)["topic_id"].astype(int).tolist()
    if not choices:
        st.info("Không có topic để chỉnh.")
        return
    key = f"mapping_{run.run_id}"
    if key not in st.session_state:
        st.session_state[key] = {}
    uploaded = st.file_uploader("Nạp mapping JSON đã lưu", type="json")
    if uploaded is not None and st.button("Áp dụng mapping đã nạp"):
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
    chosen = st.selectbox("Topic", choices, format_func=lambda value: topic_label(value, names))
    current = st.session_state[key].get(str(chosen), {})
    with st.form("mapping_form"):
        new_name = st.text_input("Tên hiển thị", value=current.get("display_name", names.get(chosen, "")))
        group = st.text_input("Nhóm hiển thị (tùy chọn)", value=current.get("display_group_id", ""))
        saved = st.form_submit_button("Lưu mapping trong phiên")
    if saved:
        st.session_state[key][str(chosen)] = {
            "display_name": new_name.strip() or names.get(chosen, ""),
            "display_group_id": group.strip() or None,
        }
        st.success("Đã lưu mapping hiển thị. Các topic cùng group chỉ được nhóm trên giao diện.")
    mapping = st.session_state[key]
    if mapping:
        display = pd.DataFrame([{"topic_id": int(topic_id), **value} for topic_id, value in mapping.items()])
        st.dataframe(display, width="stretch", hide_index=True)
        topics = get_topic_table(bundle, split=split).copy()
        topics["display_name"] = topics.apply(
            lambda row: mapping.get(str(int(row["topic_id"])), {}).get(
                "display_name", row["topic_name"]
            ),
            axis=1,
        )
        topics["display_group_id"] = topics["topic_id"].map(
            lambda value: mapping.get(str(int(value)), {}).get("display_group_id")
            or f"T{int(value)}"
        )
        groups = topics.groupby("display_group_id", as_index=False).agg(
            topic_ids=("topic_id", lambda values: ", ".join(f"T{value}" for value in values)),
            document_count=("document_count", "sum"),
        )
        st.subheader("Bảng nhóm hiển thị")
        st.dataframe(groups, width="stretch", hide_index=True)
        st.caption("Nhóm chỉ cộng số bài để trình bày; topic assignment và model gốc vẫn giữ nguyên.")
    payload = {"run_id": run.run_id, "schema_version": bundle.manifest.schema_version, "mapping": mapping}
    st.download_button("Tải mapping JSON", json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"), f"{run.run_id}_display_mapping.json", mime="application/json")


def render_new_article() -> None:
    st.caption("Trang này dùng API nội bộ và model đã lưu. Không huấn luyện lại khi gửi bài.")
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
    st.download_button("Tải kết quả JSON", json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"), "new_article_prediction.json", mime="application/json")


def render_contributions(runs: list[ArtifactRun]) -> None:
    st.caption("Bằng chứng được lấy từ các run artifact hiện có; phần demo của mỗi người nên kèm PR, hình và một case thật.")
    available = {run.method: run for run in runs}
    for member, description, method in CONTRIBUTIONS:
        with st.expander(member, expanded=(method == "bertopic")):
            st.write(description)
            run = available.get(method)
            if run:
                bundle = load_dashboard_bundle(str(run.path))
                st.write(f"**Artifact:** `{run.path.relative_to(Path(__file__).resolve().parents[1])}`")
                st.write(f"**Run:** `{bundle.model_summary.run_id}` · **Số topic báo cáo:** {bundle.model_summary.topic_count}")
            else:
                st.info("Chưa có artifact chuẩn cho phương pháp này.")
    st.divider()
    st.subheader("Tiến độ sản phẩm")
    st.write("Các trang đọc artifact có thể dùng ngay. Bài báo mới cần API `/predict` và model inference; Event Detection và Trending/Emerging cần artifact/logic riêng trước khi đưa thành kết quả nghiên cứu.")


def main() -> None:
    runs = discover_artifact_runs()
    if not runs:
        st.error("Không tìm thấy manifest trong artifacts/common.")
        st.stop()
    page, run, bundle, split = sidebar(runs)
    render_header(page, run, split)
    if page == "Tổng quan":
        render_overview(run, bundle, split)
    elif page == "Khám phá topic":
        render_topic_detail(run, bundle, split)
    elif page == "Tìm kiếm & lọc":
        render_search(run, bundle, split)
    elif page == "Bản đồ 2D":
        render_map(run, bundle, split)
    elif page == "Xu hướng":
        render_trend(run, bundle, split)
    elif page == "So sánh mô hình":
        render_comparison(runs, run)
    elif page == "Thí nghiệm":
        render_experiments(run)
    elif page == "Phân tích lỗi":
        render_errors(run, bundle, split)
    elif page == "Đổi tên & gộp":
        render_mapping(run, bundle, split)
    elif page == "Bài báo mới":
        render_new_article()
    else:
        render_contributions(runs)

if __name__ == "__main__":
    main()
