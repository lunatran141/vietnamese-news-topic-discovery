"""Các màn hình phân tích NMF từ dữ liệu đã xuất."""

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
import json

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from app.ui import charts
from app.ui.components import clean_topic_name

from app.data_service import (
    discover_model_runs, filter_documents, get_article_distribution,
    get_split_metrics, load_dashboard_bundle, load_optional_table,
)


def _render_topic_visuals(bundle) -> None:
    """Compute split counts from the selected run, rather than another run's figures."""
    st.subheader("Phân bố chủ đề và trọng số từ khóa")
    docs = bundle.document_topics
    splits = [s for s in ("train", "val", "test") if s in set(docs["split"])]
    if not splits:
        st.info("Chưa có dữ liệu split để vẽ phân bố chủ đề.")
        return
    split = st.selectbox("Split phân bố chủ đề", splits, key="nmf_topic_visual_split")
    counts = docs.loc[docs["split"].eq(split), "topic_id"].value_counts()
    topics = bundle.topic_summary.copy()
    topics["document_count"] = topics["topic_id"].map(counts).fillna(0).astype(int)
    topics = topics.sort_values("document_count", ascending=False)
    labels = {int(r.topic_id): f"T{int(r.topic_id)} · {clean_topic_name(r.topic_name, getattr(r, 'top_keywords', None))}"
              for r in topics.itertuples()}
    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.topic_distribution_bar(topics, labels),
                        width="stretch", config=charts.PLOT_CONFIG)
        st.caption(f"Đếm topic chính trên {len(docs.loc[docs['split'].eq(split)]):,} bài thuộc {split}.")
    with right:
        ids = sorted(labels)
        if not ids:
            st.info("Chưa có chủ đề để hiển thị.")
            return
        topic = st.selectbox("Chọn topic xem từ khóa", ids,
                             format_func=lambda t: labels[t], key="nmf_keyword_visual_topic")
        words = bundle.topic_keywords
        words = words.loc[words["topic_id"].eq(topic)].sort_values("rank").head(10).copy()
        if "weight" in words:
            words["weight"] = pd.to_numeric(words["weight"], errors="coerce")
            st.plotly_chart(charts.keyword_bar(words, "weight", "Trọng số từ khóa trong H"),
                            width="stretch", config=charts.PLOT_CONFIG)
        else:
            st.info("Run này chưa xuất trọng số từ khóa.")
        st.caption("Trọng số H dùng để xếp hạng từ trong từng topic, không phải xác suất. Tên topic là gợi ý, chưa phải nhãn đã duyệt.")


def _render_overlap_visuals(docs: pd.DataFrame, threshold: float) -> None:
    if docs.empty or "second_topic_share" not in docs:
        return
    shares = pd.to_numeric(docs["second_topic_share"], errors="coerce").dropna()
    figure = go.Figure(go.Histogram(x=shares, xbins=dict(start=0, end=1, size=0.025),
                                    marker_color="#397A96"))
    figure.add_vline(x=threshold, line_dash="dash", line_color="#B65D38",
                     annotation_text=f"Ngưỡng {threshold:.2f}")
    figure.update_layout(template="plotly_white", height=280,
                         xaxis_title="Share của topic đứng thứ hai", yaxis_title="Số bài",
                         xaxis_range=[0, 1], margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(figure, width="stretch", config=charts.PLOT_CONFIG)
    st.caption("Phân bố trên các bài đang lọc. Vượt ngưỡng chỉ tạo ứng viên đa chủ đề; cần đọc bài để xác nhận.")


def render_nmf_details(bundle, artifact_dir: str) -> None:
    metrics = bundle.model_summary.metrics
    analysis = bundle.model_summary.config.get("analysis", {})
    tabs = st.tabs(["Chất lượng và thời gian", "Bài báo và top-2", "Bài đại diện và bản đồ", "NMF–LDA", "Đánh giá bằng người"])

    with tabs[0]:
        st.subheader("Kết quả theo split")
        split_metrics = get_split_metrics(bundle)
        st.dataframe(split_metrics, hide_index=True, width="stretch")
        available = [c for c in ("ari", "nmi", "npmi", "topic_diversity")
                     if c in split_metrics and split_metrics[c].notna().any()]
        if available:
            st.bar_chart(split_metrics.set_index("split")[available])
        st.caption("Validation: fit trên train. Test: fit trên train+validation. ARI/NMI đối chiếu chuyên mục báo; reconstruction error càng thấp càng tốt, các metric còn lại càng cao càng tốt.")
        timings = pd.DataFrame([
            {"Công đoạn": name, "Giây": metrics.get(name)} for name in (
                "representation_seconds", "fit_seconds", "validation_transform_seconds",
                "test_transform_seconds", "evaluation_seconds", "runtime_seconds",
            )
        ])
        st.dataframe(timings, hide_index=True, width="stretch")
        st.caption("Lượt đo mới với K đã khóa: tổng gồm cả bước kiểm tra validation và fit final. Transform gồm TF-IDF và suy diễn NMF; evaluation chỉ gồm tính metric. Không gồm đọc dữ liệu, ghi file hoặc giao diện.")
        if bundle.model_summary.runtime_seconds is None:
            st.info("Run này chưa có runtime được đo. Chọn run dashboard để xem lượt đo mới.")

    with tabs[1]:
        st.caption("Topic share là trọng số NMF chuẩn hóa theo từng bài, không phải xác suất đã hiệu chỉnh.")
        st.caption("Các bài ở đây dùng mô hình final fit trên train+validation. Dòng val thuộc development; metric validation độc lập ở tab trước dùng một mô hình chỉ fit trên train.")
        split = st.selectbox("Tập bài báo", ["val", "train", "test"], key="nmf_split")
        threshold = float(analysis.get("secondary_topic_min_share", 0.20))
        multi_only = st.checkbox(f"Chỉ ứng viên đa chủ đề: topic phụ có share ≥ {threshold:.2f}")
        close_only = st.checkbox("Lọc thêm bài có hai topic gần nhau")
        gap = st.slider("Gap tối đa giữa hai topic", 0.0, 1.0,
                        float(analysis.get("top_two_gap_filter", 0.10)), 0.01)
        query = st.text_input("Tìm tiêu đề hoặc mã bài")
        docs = filter_documents(bundle, split, multi_only, gap if close_only else None, query)
        st.caption(f"{len(docs):,} bài phù hợp. Bộ lọc gap không thay đổi nhãn đa chủ đề đã xuất.")
        _render_overlap_visuals(docs, threshold)
        columns = [name for name in (
            "article_id", "title", "topic_id", "topic_share", "second_topic_id",
            "second_topic_share", "top_two_share_gap", "topic_entropy",
            "normalized_topic_entropy", "is_multi_topic", "category_gold",
        ) if name in docs]
        st.dataframe(docs[columns], hide_index=True, width="stretch")
        st.download_button("Tải danh sách đang lọc", docs[columns].to_csv(index=False).encode("utf-8-sig"),
                           file_name="nmf_filtered_documents.csv", mime="text/csv")
        if not docs.empty:
            indexed = docs.set_index("article_id")
            article = st.selectbox("Xem phân phối topic của bài", docs["article_id"].tolist(),
                format_func=lambda key: f"{key} · {str(indexed.loc[key].get('title', ''))[:100]}")
            distribution = get_article_distribution(bundle, article)
            if not distribution.empty:
                st.bar_chart(distribution.set_index("topic_id")["topic_share"])
                st.dataframe(distribution, hide_index=True)
            topic = int(indexed.loc[article]["topic_id"])
            keywords = bundle.topic_keywords.loc[bundle.topic_keywords["topic_id"].eq(topic)].sort_values("rank")
            st.write("Từ khóa topic chính: " + ", ".join(keywords["keyword"].astype(str)))

    with tabs[2]:
        _render_topic_visuals(bundle)
        reps = load_optional_table(artifact_dir, "representative_docs.csv")
        st.subheader("Bài đại diện")
        if reps.empty:
            st.info("Run này chưa có bài đại diện.")
        else:
            st.dataframe(reps, hide_index=True, width="stretch")
            st.caption("Đọc cột split để biết nguồn bài đại diện. Run dashboard lấy đại diện từ train+validation.")
        projection = load_optional_table(artifact_dir, "projection_2d.csv")
        if not projection.empty:
            projection_split = st.selectbox("Split trên bản đồ", ["val", "train", "test"], key="nmf_projection_split")
            points = projection.loc[projection["split"].eq(projection_split)].copy()
            points["topic_id"] = points["topic_id"].astype(str)
            st.scatter_chart(points, x="x", y="y", color="topic_id")
            st.caption("PCA fit trên topic shares của train+validation, chỉ để trực quan hóa; hai chiều không phải metric chất lượng hay không gian embedding.")

    with tabs[3]:
        rows = []
        candidates = discover_model_runs()
        lda_paths = [path for path in candidates.values() if Path(path).parent.name.lower() == "lda"]
        chosen = [bundle]
        if lda_paths:
            lda_path = st.selectbox("Run LDA để đối chiếu", lda_paths)
            chosen.append(load_dashboard_bundle(lda_path))
        for model in chosen:
            table = get_split_metrics(model)
            for row in table.to_dict("records"):
                if row["split"] == "test":
                    rows.append({"Mô hình": model.model_summary.algorithm,
                                 "Run": model.model_summary.run_id,
                                 "K": model.model_summary.topic_count, **row})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("So sánh hai pipeline: NMF dùng P3/TF-IDF, LDA final dùng P2/Bag-of-Words. NPMI là document-level, top-10, trên văn bản test của từng pipeline; preprocessing/từ vựng khác nhau. Không coi đây là thí nghiệm cô lập thuật toán. Reconstruction error chỉ áp dụng NMF; không so share với xác suất LDA.")

    with tabs[4]:
        path = Path("results/human_eval/lda_nmf/summary.json")
        if path.is_file():
            summary = json.loads(path.read_text(encoding="utf-8"))
            matches = summary.get("source_runs", {}).get("NMF") == bundle.model_summary.run_id
            st.caption("Nguồn bộ đánh giá: " + ", ".join(
                f"{model}: {run}" for model, run in summary.get("source_runs", {}).items()))
            if not matches:
                st.info("Run đang chọn chưa có bộ đánh giá tương ứng.")
            elif summary["status"] == "complete":
                st.dataframe(pd.DataFrame(summary["by_model"]), hide_index=True)
                st.dataframe(pd.DataFrame(summary["agreement"]), hide_index=True)
                st.caption("Kết quả thuộc các run được ghi trong metadata của bộ đánh giá, không tự động áp dụng cho run khác.")
            else:
                st.info("Chưa đánh giá đầy đủ: cần hai người chấm độc lập hoàn thành phiếu.")
        else:
            st.info("Chưa đánh giá. Chưa có kết quả chấm của người đánh giá.")
        st.write("Chấm coherence từ khóa, khả năng đặt tên và độ phù hợp bài–topic từ 1 đến 5; ghi topic quá rộng, khó đặt tên và mã người chấm.")
        st.caption("Phiếu giữ mù tên mô hình. Điểm không được suy ra từ metric tự động.")
