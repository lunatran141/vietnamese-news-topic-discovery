"""LDA deep dive (read-only): common artifacts + experiment CSVs.

Compatible with the team's app/data_service.py and Streamlit layout.
No model re-training, no manufactured per-document 15-topic probabilities.
"""
from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import streamlit as st

from app.data_service import get_split_metrics

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "lda"


def _csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.warning(f"Không đọc được {path.name}: {exc}")
        return pd.DataFrame()


def _show(df: pd.DataFrame, *, limit=200, download=None) -> None:
    if df.empty:
        st.info("N/A — chưa có dữ liệu cho mục này.")
        return
    st.dataframe(df.head(limit), hide_index=True, width="stretch")
    if len(df) > limit:
        st.caption(f"Hiển thị {limit:,}/{len(df):,} dòng.")
    if download:
        st.download_button("Tải CSV", df.to_csv(index=False).encode("utf-8-sig"),
                           download, mime="text/csv")


def _metric(value, digits=4):
    try:
        return "N/A" if value is None or pd.isna(value) else f"{float(value):,.{digits}f}"
    except (ValueError, TypeError):
        return "N/A"


def _names(bundle) -> dict[int, str]:
    topics = bundle.topic_summary
    return {int(r.topic_id): str(r.topic_name)
            for r in topics.itertuples(index=False)
            if pd.notna(r.topic_id) and int(r.topic_id) >= 0}


def _split_filter(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if "split" not in df.columns:
        return df
    present = set(df["split"].dropna().astype(str))
    values = ["Tất cả"] + [v for v in ("train", "val", "test") if v in present]
    selected = st.selectbox("Split", values, key=key)
    return df if selected == "Tất cả" else df.loc[df["split"].astype(str).eq(selected)].copy()


def _quality(bundle):
    st.subheader("Chất lượng và runtime")
    docs, summary = bundle.document_topics, bundle.model_summary
    m = summary.metrics or {}
    a, b, c, d = st.columns(4)
    a.metric("Số bài báo", f"{len(docs):,}")
    b.metric("Số topic", str(summary.topic_count))
    c.metric("Test NPMI", _metric(m.get("test_npmi", m.get("npmi_coherence"))))
    d.metric("Topic diversity", _metric(m.get("topic_diversity")))
    st.markdown("**Đánh giá validation và test**")
    _show(get_split_metrics(bundle))
    st.caption("Validation: fit train; test: fit train+validation theo protocol của LDA Final.")
    stages = [("Biểu diễn BoW", "representation_seconds"), ("Fit LDA", "fit_seconds"),
              ("Suy diễn validation", "validation_inference_seconds"),
              ("Suy diễn test", "test_inference_seconds"), ("Evaluation", "evaluation_seconds"),
              ("Pipeline runtime", "runtime_seconds")]
    extras = getattr(summary, "model_extra", None) or {}
    rows = [{"Công đoạn": label, "Giây": getattr(summary, field, None) or extras.get(field)}
            for label, field in stages]
    st.markdown("**Thời gian theo công đoạn (nếu metadata có ghi)**")
    _show(pd.DataFrame(rows))
    st.caption("N/A nghĩa là metadata chưa khai báo công đoạn đó; không thay bằng 0.")


def _topics(bundle):
    st.subheader("Phân bố topic và từ khóa")
    summary = bundle.topic_summary.copy()
    if summary.empty:
        st.info("Chưa có topic_summary.csv")
        return
    summary = summary.loc[pd.to_numeric(summary["topic_id"], errors="coerce").ge(0)]
    if "document_count" in summary:
        chart = summary[["topic_id", "document_count"]].copy()
        chart["Topic"] = chart["topic_id"].map(lambda v: f"T{int(v)}")
        st.bar_chart(chart.set_index("Topic")["document_count"])
    _show(summary, download="lda_topic_summary.csv")
    names = _names(bundle)
    if not names:
        return
    topic = st.selectbox("Topic cần phân tích", sorted(names),
                         format_func=lambda v: f"T{v} · {names[v]}", key="lda_topic_select")
    words = bundle.topic_keywords
    selected = words.loc[pd.to_numeric(words["topic_id"], errors="coerce").eq(topic)].copy()
    if "rank" in selected:
        selected = selected.sort_values("rank")
    if "weight" in selected and "keyword" in selected:
        preview = selected.head(15).copy()
        preview["weight"] = pd.to_numeric(preview["weight"], errors="coerce")
        st.bar_chart(preview.set_index("keyword")["weight"])
    _show(selected, download=f"lda_topic_{topic}_keywords.csv")
    docs = bundle.document_topics
    matches = docs.loc[pd.to_numeric(docs["topic_id"], errors="coerce").eq(topic)].copy()
    if "topic_probability" in matches:
        matches["topic_probability"] = pd.to_numeric(matches["topic_probability"], errors="coerce")
        matches = matches.sort_values("topic_probability", ascending=False)
    st.markdown("**Bài có xác suất topic chính cao nhất**")
    _show(matches[[c for c in ("article_id", "split", "title", "topic_probability", "category_gold")
                   if c in matches]], limit=20, download=f"lda_topic_{topic}_documents.csv")
    st.caption("Các tên topic hiện tại tạo từ những từ khóa đứng đầu, chưa phải nhãn chủ đề đã được người đánh giá duyệt.")


def _overlap(bundle):
    st.subheader("Xác suất topic và bài chồng lấn")
    docs = bundle.document_topics.copy()
    docs = _split_filter(docs, "lda_membership_split")
    if "top_two_share_gap" not in docs:
        st.info("N/A — document_topics.csv thiếu top_two_share_gap.")
        return
    gap = st.slider("Top-2 gap tối đa", 0.0, 1.0, 0.10, 0.01, key="lda_membership_gap")
    docs["top_two_share_gap"] = pd.to_numeric(docs["top_two_share_gap"], errors="coerce")
    docs = docs.loc[docs["top_two_share_gap"].le(gap)].copy()
    if "is_multi_topic" in docs:
        multi = st.checkbox("Chỉ bài được đánh dấu đa chủ đề", value=False, key="lda_multi_only")
        if multi:
            docs = docs.loc[docs["is_multi_topic"].astype(str).str.lower().isin(["true", "1"])].copy()
    query = st.text_input("Tìm article_id hoặc tiêu đề", key="lda_membership_query")
    if query:
        mask = docs["article_id"].astype(str).str.contains(query, regex=False, case=False)
        if "title" in docs:
            mask |= docs["title"].fillna("").astype(str).str.contains(query, regex=False, case=False)
        docs = docs.loc[mask].copy()
    docs = docs.sort_values("top_two_share_gap")
    st.caption(f"{len(docs):,} bài đáp ứng điều kiện.")
    cols = [c for c in ("article_id", "split", "title", "topic_id", "topic_probability",
                        "second_topic_id", "second_topic_probability", "top_two_share_gap",
                        "topic_entropy", "is_multi_topic", "category_gold") if c in docs]
    _show(docs[cols], download="lda_overlap_documents.csv")
    st.caption("LDA cung cấp phân phối xác suất. Common artifact chỉ lưu topic chính, topic phụ và các chỉ số tổng hợp; đây không phải full 15-topic vector.")


def _projection(bundle, artifact_dir: str):
    st.subheader("Bản đồ topic 2D")
    projection = _csv(Path(artifact_dir) / "projection_2d.csv")
    if projection.empty or not {"x", "y", "topic_id"}.issubset(projection.columns):
        st.info("N/A — chưa có projection_2d.csv hợp lệ.")
        return
    projection = _split_filter(projection, "lda_projection_split")
    if projection.empty:
        st.info("Không có điểm trong split được chọn.")
        return
    projection["Topic"] = projection["topic_id"].astype(str)
    st.scatter_chart(projection, x="x", y="y", color="Topic")
    st.caption("PCA của phân phối topic; không dùng khoảng cách 2D để kết luận độ tương đồng ngữ nghĩa.")


def _config(bundle):
    st.subheader("So sánh cấu hình và lựa chọn K")
    table = _csv(RESULTS / "m08" / "topic_number_results.csv")
    if table.empty:
        st.info("N/A — chưa có kết quả M08.")
    else:
        if "k" in table:
            for metric in ("coherence_cv", "coherence_npmi", "topic_diversity", "runtime_seconds"):
                if metric in table:
                    st.markdown(f"**{metric} theo K**")
                    plot = table[["k", metric]].copy()
                    plot["k"] = pd.to_numeric(plot["k"], errors="coerce")
                    plot[metric] = pd.to_numeric(plot[metric], errors="coerce")
                    st.line_chart(plot.dropna().set_index("k"))
        _show(table, download="lda_k_selection.csv")
    st.caption("K=15 là lựa chọn của run final; metric theo K thuộc thí nghiệm M08, không phải phép đo lại trên test.")
    with st.expander("Cấu hình run đang chọn"):
        st.json(bundle.model_summary.config)


def _stability(bundle):
    st.subheader("Độ ổn định giữa các random seed")
    table = _csv(RESULTS / "m09" / "stability_metrics.csv")
    if table.empty:
        table = _csv(RESULTS / "m09" / "pairwise_stability.csv")
    if table.empty:
        st.info("N/A — chưa có thí nghiệm M09.")
        return
    if "candidate_seed" in table:
        display = table.loc[table["candidate_seed"].ne(table.get("reference_seed", -1))].copy()
        seed_column = "candidate_seed"
    else:
        display, seed_column = table.copy(), "seed_b"
    cols = [c for c in ("mean_matched_jaccard", "mean_document_cosine") if c in display]
    if cols and seed_column in display:
        st.bar_chart(display.set_index(seed_column)[cols])
    _show(table, download="lda_stability.csv")
    m = bundle.model_summary.metrics or {}
    st.caption(f"Matched Jaccard trung bình: {_metric(m.get('stability_mean'))}; "
               f"độ lệch chuẩn: {_metric(m.get('stability_std'))}. "
               "So khớp topic giữa seed là cần thiết vì ID topic có thể hoán vị.")


def _hard_cases(bundle):
    st.subheader("Hard cases — các bài có assignment khó")
    docs = _split_filter(bundle.document_topics.copy(), "lda_hard_split")
    options = [c for c in ("Độ tự tin thấp", "Top-2 gần nhau", "Entropy cao")]
    criterion = st.selectbox("Tiêu chí", options, key="lda_hard_criterion")
    col = {"Độ tự tin thấp": "topic_probability", "Top-2 gần nhau": "top_two_share_gap",
           "Entropy cao": "topic_entropy"}[criterion]
    if col not in docs:
        st.info(f"N/A — thiếu {col}.")
        return
    docs[col] = pd.to_numeric(docs[col], errors="coerce")
    docs = docs.sort_values(col, ascending=(criterion != "Entropy cao"))
    cols = [c for c in ("article_id", "split", "title", "topic_id", "topic_probability",
                        "second_topic_id", "second_topic_probability", "top_two_share_gap",
                        "topic_entropy", "category_gold") if c in docs]
    _show(docs[cols], limit=50, download="lda_hard_cases.csv")
    st.caption("Đây là hard cases dựa trên độ bất định của assignment, không phải lỗi gán nhãn đã được con người xác nhận.")


def _representatives(bundle, artifact_dir: str):
    st.subheader("Bài báo đại diện")
    path = Path(artifact_dir) / "representative_docs.csv"
    reps = _csv(path)
    if reps.empty:
        reps = _csv(RESULTS / "m08" / "representative_docs.csv")
    if reps.empty:
        docs = bundle.document_topics.copy()
        if "topic_probability" not in docs:
            st.info("N/A — chưa có dữ liệu bài đại diện.")
            return
        docs["topic_probability"] = pd.to_numeric(docs["topic_probability"], errors="coerce")
        reps = (docs.sort_values("topic_probability", ascending=False)
                .groupby("topic_id", sort=True).head(5))
        st.caption("Bài gợi ý được suy ra từ xác suất topic chính, chưa phải danh sách đã được duyệt.")
    else:
        st.caption(f"Đọc danh sách bài đại diện từ {path.name if path.is_file() else 'results/lda/m08/representative_docs.csv'}.")
    if "topic_id" in reps:
        names = _names(bundle)
        topics = sorted(pd.to_numeric(reps["topic_id"], errors="coerce").dropna().astype(int).unique())
        topic = st.selectbox("Lọc topic", ["Tất cả"] + topics,
                             format_func=lambda x: "Tất cả" if x == "Tất cả" else f"T{x} · {names.get(x, '')}",
                             key="lda_reps_topic")
        if topic != "Tất cả":
            reps = reps.loc[pd.to_numeric(reps["topic_id"], errors="coerce").eq(topic)]
    _show(reps, download="lda_representatives.csv")


def _human_eval(bundle):
    st.subheader("Đánh giá bởi con người")
    summary_path = ROOT / "results" / "human_eval" / "lda_nmf" / "summary.json"
    if not summary_path.is_file():
        st.info("N/A — chưa có summary đánh giá người cho LDA/NMF.")
        return
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        st.warning(f"Không đọc được human evaluation: {exc}")
        return
    run_id = bundle.model_summary.run_id
    linked = payload.get("source_runs", {}).get("LDA")
    st.caption(f"LDA run trong phiếu: {linked or 'N/A'}; run đang xem: {run_id}.")
    if linked != run_id:
        st.info("Bộ chấm không dành cho run LDA đang chọn.")
    elif payload.get("status") != "complete":
        st.info("Đánh giá người chưa hoàn thành đủ phiếu độc lập.")
    else:
        for key in ("by_model", "agreement"):
            value = payload.get(key)
            if value is not None:
                st.markdown(f"**{key}**")
                _show(pd.DataFrame(value) if isinstance(value, (list, dict)) else pd.DataFrame())
    st.caption("Không dùng coherence/NPMI thay cho kết quả đánh giá độc lập của người chấm.")


def render_lda_details(bundle, artifact_dir: str) -> None:
    """Entry point, same pattern as render_hdbscan_details and render_nmf_details."""
    st.subheader("Phân tích chi tiết LDA")
    tabs = st.tabs([
        "Chất lượng và runtime", "Phân bố và từ khóa", "Xác suất & top-2",
        "Bản đồ 2D", "So sánh cấu hình", "Stability", "Hard cases",
        "Bài đại diện", "Human evaluation",
    ])
    with tabs[0]: _quality(bundle)
    with tabs[1]: _topics(bundle)
    with tabs[2]: _overlap(bundle)
    with tabs[3]: _projection(bundle, artifact_dir)
    with tabs[4]: _config(bundle)
    with tabs[5]: _stability(bundle)
    with tabs[6]: _hard_cases(bundle)
    with tabs[7]: _representatives(bundle, artifact_dir)
    with tabs[8]: _human_eval(bundle)
