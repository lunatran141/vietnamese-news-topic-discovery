"""Các màn hình phân tích NMF từ dữ liệu đã xuất."""

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
import json

import pandas as pd
import streamlit as st

from app.data_service import (
    discover_model_runs, filter_documents, get_article_distribution,
    get_split_metrics, load_dashboard_bundle, load_optional_table,
)


def render_nmf_details(bundle, artifact_dir: str) -> None:
    metrics = bundle.model_summary.metrics
    analysis = bundle.model_summary.config.get("analysis", {})
    tabs = st.tabs(["Chất lượng và thời gian", "Bài báo và top-2", "Bài đại diện và bản đồ", "NMF–LDA", "Đánh giá bằng người"])

    with tabs[0]:
        st.subheader("Kết quả theo split")
        st.dataframe(get_split_metrics(bundle), hide_index=True, width="stretch")
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
        multi_only = st.checkbox(f"Chỉ bài đa chủ đề: topic phụ có share ≥ {threshold:.2f}")
        close_only = st.checkbox("Lọc thêm bài có hai topic gần nhau")
        gap = st.slider("Gap tối đa giữa hai topic", 0.0, 1.0,
                        float(analysis.get("top_two_gap_filter", 0.10)), 0.01)
        query = st.text_input("Tìm tiêu đề hoặc mã bài")
        docs = filter_documents(bundle, split, multi_only, gap if close_only else None, query)
        st.caption(f"{len(docs):,} bài phù hợp. Bộ lọc gap không thay đổi nhãn đa chủ đề đã xuất.")
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
