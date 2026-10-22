"""Trang 4: Tổ chức tự động bài báo mới (từ URL, văn bản hoặc file) thành nhóm sự kiện và chủ đề."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.article_organizer import (
    SAMPLE_ARTICLES,
    assign_topics_to_articles,
    cluster_articles_into_events,
    fetch_articles_from_urls,
    find_related_corpus_articles,
    get_csv_template_bytes,
    parse_raw_text_entries,
    parse_uploaded_file,
)
from app.ui import charts
from app.ui.components import (
    build_topic_labels,
    data_table,
    doc_row,
    fmt_number,
    fmt_percent,
    keyword_pills,
    kpi_row,
    note,
    page_header,
    table_download,
)
from app.ui.context import AnalysisContext

ORGANIZER_MODES = [
    "Tự động gom Nhóm Sự kiện (Dynamic Events)",
    "Tổ chức theo Chủ đề của Mô hình (Model Topics)",
]


def render(ctx: AnalysisContext) -> None:
    run, bundle = ctx.run, ctx.bundle
    labels = build_topic_labels(bundle, run.run_id)

    page_header(
        "Tổ chức bài báo mới",
        "Dán URL, nhập tiêu đề và nội dung hoặc tải file để tự động phân nhóm sự kiện và gán chủ đề.",
        {"Mô hình đang dùng": run.label},
        eyebrow="Phân tích",
    )

    if "organizer_articles" not in st.session_state:
        st.session_state["organizer_articles"] = [dict(item) for item in SAMPLE_ARTICLES]
        st.session_state["organizer_source_note"] = "Bộ 7 bài báo mẫu đa lĩnh vực"

    _render_input_panel()

    raw_articles: list[dict[str, str]] = st.session_state.get("organizer_articles", [])
    if not raw_articles:
        st.info(
            "Chưa có bài báo nào trong danh sách. Hãy dán URL, nhập nội dung, tải file lên "
            "hoặc bấm nút nạp bộ dữ liệu mẫu ở trên để bắt đầu."
        )
        return

    # Thực thi gán chủ đề theo mô hình + gom cụm sự kiện động
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.8, 1.2, 0.8])
    with ctrl_col1:
        org_mode = st.radio(
            "Phương thức tổ chức chính",
            ORGANIZER_MODES,
            horizontal=True,
            key="organizer_mode",
        )
    with ctrl_col2:
        sim_threshold = st.slider(
            "Ngưỡng tương đồng gom sự kiện",
            min_value=0.05,
            max_value=0.60,
            value=0.14,
            step=0.02,
            key="organizer_sim_threshold",
            help="Ngưỡng càng cao thì các bài phải càng giống nhau mới được xếp chung vào một nhóm sự kiện.",
        )
    with ctrl_col3:
        st.write("")
        if st.button("Xóa danh sách bài", width="stretch", key="organizer_clear_btn"):
            st.session_state["organizer_articles"] = []
            st.session_state["organizer_source_note"] = ""
            st.rerun()

    source_note = st.session_state.get("organizer_source_note", "")
    if source_note:
        st.caption(f"Nguồn dữ liệu đang nạp: {source_note} ({len(raw_articles)} bài viết)")

    assigned_df = assign_topics_to_articles(raw_articles, bundle, run, labels)
    organized_df, event_summary = cluster_articles_into_events(
        assigned_df,
        similarity_threshold=float(sim_threshold),
    )

    active_topics_found = int(organized_df.loc[organized_df["topic_id"] != -1, "topic_id"].nunique())
    event_groups_found = int(len(event_summary))
    mean_conf = float(organized_df["confidence"].mean()) if not organized_df.empty else 0.0

    kpi_row([
        ("Số bài báo đầu vào", fmt_number(len(organized_df)), None),
        ("Nhóm sự kiện phát hiện", fmt_number(event_groups_found), "Gom cụm tương đồng TF-IDF"),
        ("Chủ đề mô hình khớp", fmt_number(active_topics_found), run.label),
        ("Độ tin cậy TB", fmt_percent(mean_conf), "Xác suất / điểm tương đồng"),
    ])
    st.write("")

    by_event = org_mode == ORGANIZER_MODES[0]
    _render_visual_summary(organized_df, event_summary, labels, by_event=by_event)

    st.subheader("Các nhóm đã tổ chức")
    if by_event:
        _render_event_groups(organized_df, event_summary, bundle, labels)
    else:
        _render_topic_groups(organized_df, bundle, labels)

    st.subheader("Bảng chi tiết toàn bộ bài báo")
    display_cols = [
        "title",
        "event_label",
        "topic_label",
        "confidence",
        "matched_keywords",
        "top_3_topics",
        "source",
        "published_at",
        "url",
    ]
    table_view = organized_df[display_cols].rename(
        columns={
            "title": "Tiêu đề",
            "event_label": "Nhóm sự kiện",
            "topic_label": "Chủ đề mô hình",
            "confidence": "Độ tin cậy",
            "matched_keywords": "Từ khóa khớp",
            "top_3_topics": "Top-3 chủ đề gần nhất",
            "source": "Nguồn",
            "published_at": "Ngày đăng",
            "url": "URL",
        }
    )
    data_table(
        table_view,
        {
            "Độ tin cậy": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    table_download(
        table_view,
        f"organized_articles_{run.run_id}.csv",
        "Tải kết quả phân nhóm CSV",
    )


def _render_input_panel() -> None:
    tab_url, tab_text, tab_file = st.tabs(
        ["Dán URL bài báo", "Nhập Tiêu đề & Nội dung", "Tải lên File bài báo"]
    )

    with tab_url:
        note(
            "Dán một hoặc nhiều đường dẫn (mỗi dòng 1 URL) từ VnExpress, Tuổi Trẻ, VietNamNet "
            "hoặc các trang báo điện tử tiếng Việt. Hệ thống sẽ tự động trích xuất tiêu đề và nội dung."
        )
        urls_text = st.text_area(
            "Danh sách URL bài báo (mỗi dòng một URL)",
            height=110,
            placeholder="https://vnexpress.net/...\nhttps://tuoitre.vn/...\nhttps://vietnamnet.vn/...",
            key="organizer_urls_input",
        )
        u_col1, u_col2 = st.columns([1.2, 1])
        with u_col1:
            if st.button("Tải URL và Tổ chức nhóm", type="primary", width="stretch", key="btn_fetch_urls"):
                if not urls_text.strip():
                    st.warning("Vui lòng dán ít nhất một đường dẫn URL.")
                else:
                    with st.spinner("Đang tải và trích xuất nội dung bài báo..."):
                        fetched, errors = fetch_articles_from_urls(urls_text)
                    for err in errors:
                        st.warning(err)
                    if fetched:
                        st.session_state["organizer_articles"] = fetched
                        st.session_state["organizer_source_note"] = f"Trích xuất từ {len(fetched)} URL"
                        st.success(f"Đã tải và trích xuất thành công {len(fetched)} bài báo.")
        with u_col2:
            if st.button("Nạp 7 bài báo mẫu đa sự kiện", width="stretch", key="btn_sample_from_url"):
                st.session_state["organizer_articles"] = [dict(item) for item in SAMPLE_ARTICLES]
                st.session_state["organizer_source_note"] = "Bộ 7 bài báo mẫu đa lĩnh vực"
                st.rerun()

    with tab_text:
        note(
            "Nhập nhanh một bài báo ở cột trái, hoặc dán nhiều bài báo cùng lúc ở cột phải "
            "(mỗi bài cách nhau bởi dòng `---` hoặc một dòng trống; dòng đầu tiên của mỗi đoạn là tiêu đề)."
        )
        t_left, t_right = st.columns([1, 1])
        with t_left:
            single_title = st.text_input(
                "Tiêu đề bài báo",
                placeholder="Ví dụ: Ngân hàng Nhà nước điều chỉnh lãi suất...",
                key="org_single_title",
            )
            single_content = st.text_area(
                "Nội dung bài báo",
                height=120,
                placeholder="Dán nội dung hoặc tóm tắt bài báo vào đây...",
                key="org_single_content",
            )
        with t_right:
            batch_text = st.text_area(
                "Hoặc dán nhiều bài báo (ngăn cách bằng dòng `---`)",
                height=188,
                placeholder=(
                    "Tiêu đề bài 1\nNội dung bài báo thứ nhất...\n---\n"
                    "Tiêu đề bài 2\nNội dung bài báo thứ hai..."
                ),
                key="org_batch_text",
            )

        b_col1, b_col2 = st.columns([1, 1])
        with b_col1:
            if st.button(
                "Phân tích & Tổ chức văn bản vừa nhập (Thay thế danh sách)",
                type="primary",
                width="stretch",
                key="btn_replace_text",
            ):
                parsed = parse_raw_text_entries(single_title, single_content, batch_text)
                if not parsed:
                    st.warning("Vui lòng nhập tiêu đề/nội dung hoặc đoạn văn bản.")
                else:
                    st.session_state["organizer_articles"] = parsed
                    st.session_state["organizer_source_note"] = f"Nhập văn bản trực tiếp ({len(parsed)} bài)"
                    st.rerun()
        with b_col2:
            if st.button(
                "Thêm vào danh sách bài hiện tại",
                width="stretch",
                key="btn_append_text",
            ):
                parsed = parse_raw_text_entries(single_title, single_content, batch_text)
                if not parsed:
                    st.warning("Vui lòng nhập tiêu đề/nội dung trước khi thêm.")
                else:
                    current = list(st.session_state.get("organizer_articles", []))
                    current.extend(parsed)
                    st.session_state["organizer_articles"] = current
                    st.session_state["organizer_source_note"] = f"Danh sách kết hợp ({len(current)} bài)"
                    st.rerun()

    with tab_file:
        note(
            "Hỗ trợ file `.csv` (có cột `title`, `content` hoặc `text`), `.json`, `.jsonl` hoặc `.txt`. "
            "Bạn có thể tải file CSV mẫu bên dưới để xem định dạng chuẩn."
        )
        f_col1, f_col2 = st.columns([1.6, 1])
        with f_col1:
            uploaded = st.file_uploader(
                "Chọn file danh sách bài báo (.csv, .json, .jsonl, .txt)",
                type=["csv", "json", "jsonl", "txt"],
                key="organizer_file_uploader",
            )
            if uploaded is not None:
                if st.button("Đọc file và Tổ chức nhóm", type="primary", width="stretch", key="btn_process_file"):
                    try:
                        parsed_file = parse_uploaded_file(uploaded.name, uploaded.getvalue())
                        if not parsed_file:
                            st.warning("Không tìm thấy bài báo hợp lệ trong file đã tải lên.")
                        else:
                            st.session_state["organizer_articles"] = parsed_file
                            st.session_state["organizer_source_note"] = f"File: {uploaded.name}"
                            st.success(f"Đã nạp {len(parsed_file)} bài báo từ {uploaded.name}.")
                    except Exception as exc:
                        st.error(f"Không đọc được file {uploaded.name}: {exc}")
        with f_col2:
            st.write("")
            st.download_button(
                "Tải file CSV mẫu (Template)",
                data=get_csv_template_bytes(),
                file_name="sample_articles_template.csv",
                mime="text/csv",
                width="stretch",
                key="btn_download_template",
            )


def _render_visual_summary(
    organized_df: pd.DataFrame,
    event_summary: pd.DataFrame,
    topic_labels: dict[int, str],
    *,
    by_event: bool,
) -> None:
    left, right = st.columns([1.15, 1.25])

    if by_event:
        bar_df = pd.DataFrame(
            {
                "topic_id": event_summary["event_id"],
                "document_count": event_summary["document_count"],
            }
        )
        bar_labels = dict(zip(event_summary["event_id"].astype(int), event_summary["event_label"].astype(str)))
        scatter_df = organized_df.copy()
        scatter_df["topic_id"] = scatter_df["event_id"].astype(int)
        scatter_labels = bar_labels
        scatter_ids = sorted(int(v) for v in scatter_df["topic_id"].unique())
    else:
        counts = organized_df["topic_id"].value_counts()
        bar_df = pd.DataFrame(
            {
                "topic_id": counts.index.astype(int),
                "document_count": counts.values.astype(int),
            }
        )
        bar_labels = {
            int(tid): ("Outlier" if int(tid) == -1 else topic_labels.get(int(tid), f"T{int(tid)}"))
            for tid in bar_df["topic_id"]
        }
        scatter_df = organized_df.copy()
        scatter_df["topic_id"] = scatter_df["topic_id"].astype(int)
        scatter_labels = bar_labels
        scatter_ids = sorted(int(v) for v in scatter_df["topic_id"].unique() if int(v) != -1)

    with left:
        st.subheader("Quy mô các nhóm đã tổ chức")
        st.plotly_chart(
            charts.topic_distribution_bar(bar_df, bar_labels, top_n=15),
            width="stretch",
            config=charts.PLOT_CONFIG,
        )
    with right:
        st.subheader("Bản đồ phân bố tương đồng (2D)")
        st.plotly_chart(
            charts.projection_scatter(
                scatter_df,
                scatter_labels,
                scatter_ids,
                selected=None,
                show_outliers=True,
            ),
            width="stretch",
            config=charts.PLOT_CONFIG,
        )


def _render_event_groups(
    organized_df: pd.DataFrame,
    event_summary: pd.DataFrame,
    bundle,
    labels: dict[int, str],
) -> None:
    note(
        "Mỗi nhóm sự kiện dưới đây quy tụ các bài báo có nội dung sát nhau nhất, tự động trích xuất "
        "từ khóa sự kiện, chọn bài viết tâm điểm và đối chiếu với các bài liên quan trong kho 9,274 bài gốc."
    )
    for idx, row in enumerate(event_summary.to_dict("records")):
        eid = int(row["event_id"])
        elabel = str(row["event_label"])
        doc_count = int(row["document_count"])
        share = float(row["percentage"])
        kws = [w.strip() for w in str(row["top_keywords"]).split(",") if w.strip()]
        rep_title = str(row["representative_title"])

        members = organized_df.loc[organized_df["event_id"] == eid].sort_values(
            ["is_centroid", "event_similarity"], ascending=[False, False]
        )
        dominant_topic = members["topic_label"].mode().iloc[0] if not members.empty else "Chưa rõ"

        header_text = (
            f"{elabel} — {doc_count} bài ({share:.1%}) · Chủ đề chính: {dominant_topic}"
        )
        with st.expander(header_text, expanded=(idx < 2)):
            keyword_pills(kws)
            st.caption(f"Bài viết tâm điểm của nhóm: {rep_title}")

            c_members, c_related = st.columns([1.3, 1])
            with c_members:
                st.markdown("**Bài báo thuộc nhóm sự kiện này**")
                for m_row in members.to_dict("records"):
                    badge = "Tâm điểm nhóm" if m_row.get("is_centroid") else "Thành viên"
                    meta = [
                        badge,
                        f"Chủ đề: {m_row['topic_label']} ({m_row['confidence']:.1%})",
                        f"Tương đồng nhóm: {m_row['event_similarity']:.2f}",
                    ]
                    if m_row.get("source"):
                        meta.append(str(m_row["source"]))
                    if m_row.get("published_at"):
                        meta.append(str(m_row["published_at"])[:10])
                    doc_row(str(m_row["title"]), " · ".join(meta))

            with c_related:
                st.markdown("**Bài liên quan trong kho 9,274 bài gốc**")
                related = find_related_corpus_articles(
                    members["processed_text"].tolist(),
                    bundle,
                    labels,
                    top_k=4,
                )
                if related.empty:
                    st.caption("Không tìm thấy bài tương đồng vượt ngưỡng trong kho gốc.")
                else:
                    for r_row in related.to_dict("records"):
                        r_meta = [
                            str(r_row.get("topic_label", "")),
                            f"Tương đồng: {float(r_row.get('similarity', 0.0)):.2f}",
                            str(r_row.get("source") or "N/A"),
                            str(r_row.get("published_at") or "N/A")[:10],
                        ]
                        doc_row(str(r_row.get("title") or r_row.get("article_id")), " · ".join(r_meta))


def _render_topic_groups(
    organized_df: pd.DataFrame,
    bundle,
    labels: dict[int, str],
) -> None:
    note(
        "Các bài báo đầu vào được gom theo Chủ đề của mô hình đang chọn ở Sidebar, "
        "kèm từ khóa khớp và gợi ý bài viết cùng chủ đề trong kho dữ liệu gốc."
    )
    grouped = (
        organized_df.groupby(["topic_id", "topic_label"], as_index=False)
        .agg(
            document_count=("article_id", "count"),
            mean_confidence=("confidence", "mean"),
        )
        .sort_values("document_count", ascending=False)
    )

    total_docs = len(organized_df)
    for idx, g_row in enumerate(grouped.to_dict("records")):
        tid = int(g_row["topic_id"])
        tlabel = str(g_row["topic_label"])
        cnt = int(g_row["document_count"])
        share = cnt / total_docs if total_docs else 0.0
        mean_c = float(g_row["mean_confidence"])

        members = organized_df.loc[organized_df["topic_id"] == tid].sort_values(
            "confidence", ascending=False
        )
        sub_kw = bundle.topic_keywords.loc[bundle.topic_keywords["topic_id"] == tid].head(8)
        kws = [str(k).replace("_", " ") for k in sub_kw["keyword"]] if not sub_kw.empty else []

        header_text = (
            f"{tlabel} — {cnt} bài ({share:.1%}) · Độ tin cậy TB: {mean_c:.1%}"
        )
        with st.expander(header_text, expanded=(idx < 2)):
            if kws:
                keyword_pills(kws)

            c_members, c_related = st.columns([1.3, 1])
            with c_members:
                st.markdown("**Bài báo đầu vào thuộc chủ đề này**")
                for m_row in members.to_dict("records"):
                    meta = [
                        f"Độ tin cậy: {m_row['confidence']:.1%}",
                        f"Sự kiện: {m_row['event_label']}",
                    ]
                    if m_row.get("matched_keywords"):
                        meta.append(f"Khớp: {m_row['matched_keywords']}")
                    if m_row.get("source"):
                        meta.append(str(m_row["source"]))
                    doc_row(str(m_row["title"]), " · ".join(meta))

            with c_related:
                st.markdown("**Bài cùng sự kiện trong kho 9,274 bài gốc**")
                related = find_related_corpus_articles(
                    members["processed_text"].tolist(),
                    bundle,
                    labels,
                    top_k=4,
                )
                if related.empty:
                    st.caption("Không tìm thấy bài tương đồng trong kho gốc.")
                else:
                    for r_row in related.to_dict("records"):
                        r_meta = [
                            str(r_row.get("topic_label", "")),
                            f"Tương đồng: {float(r_row.get('similarity', 0.0)):.2f}",
                            str(r_row.get("source") or "N/A"),
                            str(r_row.get("published_at") or "N/A")[:10],
                        ]
                        doc_row(str(r_row.get("title") or r_row.get("article_id")), " · ".join(r_meta))
