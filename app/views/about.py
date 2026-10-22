"""Trang 5: Dự án (phương pháp, đóng góp, tiến độ)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.data_service import load_dashboard_bundle
from app.ui.components import note, page_header
from app.ui.context import AnalysisContext

REPO_ROOT = Path(__file__).resolve().parents[2]

CONTRIBUTIONS = [
    ("TV1 · Khôi", "Dữ liệu, TF-IDF, KMeans, preprocessing và input ablation", "kmeans"),
    ("TV2 · Tiến", "NMF, chọn số topic và phân tích bài đa chủ đề", "nmf"),
    ("TV3 · Tuấn", "LDA, stability nhiều seed và so sánh LDA–NMF", "lda"),
    ("TV4 · Vân", "Sentence embeddings, HDBSCAN và outlier analysis", "hdbscan"),
    ("TV5 · Ngân", "BERTopic, artifact chung và tích hợp Streamlit", "bertopic"),
]


def render(ctx: AnalysisContext) -> None:
    page_header(
        "Dự án",
        "Phương pháp, đóng góp của từng thành viên và tiến độ sản phẩm.",
        eyebrow="Dự án",
    )
    note(
        "Bằng chứng được lấy từ các run artifact hiện có; phần demo của mỗi người "
        "nên kèm PR, hình và một case thật."
    )
    available = {run.method: run for run in ctx.runs}
    for member, description, method in CONTRIBUTIONS:
        with st.expander(member, expanded=(method == "bertopic")):
            st.write(description)
            run = available.get(method)
            if run:
                bundle = load_dashboard_bundle(str(run.path))
                st.write(f"**Artifact:** `{run.path.relative_to(REPO_ROOT)}`")
                st.write(f"**Run:** `{bundle.model_summary.run_id}` · **Số topic báo cáo:** {bundle.model_summary.topic_count}")
            else:
                st.info("Chưa có artifact chuẩn cho phương pháp này.")
    st.subheader("Tiến độ sản phẩm")
    st.write(
        "Các trang đọc artifact có thể dùng ngay. Phân loại bài mới cần API `/predict` và model "
        "inference; Event Detection và Trending/Emerging cần artifact/logic riêng trước khi đưa "
        "thành kết quả nghiên cứu."
    )

