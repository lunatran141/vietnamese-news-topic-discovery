"""Trang 5: Dự án (phương pháp, đóng góp, tiến độ)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
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
    architecture_tab, members_tab = st.tabs(
        ["Kiến trúc hệ thống & API", "Phương pháp & đóng góp"]
    )

    with architecture_tab:
        _render_system_architecture()

    with members_tab:
        _render_contributions(ctx)


def _render_contributions(ctx: AnalysisContext) -> None:
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
    st.subheader("Trạng thái tích hợp")
    st.success(
        "Dashboard hiển thị kết quả của 5 phương pháp. Trang Phân loại bài mới "
        "cho phép chọn từng mô hình để dự đoán qua API nội bộ."
    )


def _render_system_architecture() -> None:
    st.subheader("Luồng suy diễn bài báo mới")
    st.markdown(
        "**Bài báo mới** → **Chọn mô hình trên Streamlit** → "
        "**API xử lý bằng mô hình đã lưu** → **Chủ đề, từ khóa và bài đại diện**"
    )

    st.subheader("Vai trò của từng tầng")
    layers = pd.DataFrame([
        {
            "Tầng": "Streamlit",
            "Thành phần": "app/views/articles.py",
            "Vai trò": "Nhận tiêu đề/nội dung, gọi API và trình bày kết quả giải thích.",
        },
        {
            "Tầng": "Internal API",
            "Thành phần": "src/api/main.py",
            "Vai trò": "Kiểm tra request, điều phối service và chuẩn hóa mã lỗi HTTP.",
        },
        {
            "Tầng": "Inference service",
            "Thành phần": "src/api/inference_service.py",
            "Vai trò": "Áp dụng tiền xử lý phù hợp và dự đoán bằng mô hình đã chọn.",
        },
        {
            "Tầng": "Artifact/model",
            "Thành phần": "Artifact của 5 phương pháp",
            "Vai trò": "Cung cấp mô hình, nhãn topic, từ khóa và bài đại diện.",
        },
    ])
    st.dataframe(layers, width="stretch", hide_index=True)

    st.subheader("Endpoint nội bộ")
    endpoints = pd.DataFrame([
        {
            "Phương thức": "GET",
            "Endpoint": "/health",
            "Đầu vào": "Không có",
            "Đầu ra": '{"status": "ok"}',
            "Mục đích": "Kiểm tra API đang hoạt động.",
        },
        {
            "Phương thức": "GET",
            "Endpoint": "/topics",
            "Đầu vào": "model",
            "Đầu ra": "Danh sách topic, tên và từ khóa",
            "Mục đích": "Xem các chủ đề của mô hình được chọn.",
        },
        {
            "Phương thức": "POST",
            "Endpoint": "/predict",
            "Đầu vào": "title, content, model",
            "Đầu ra": "topic, confidence, keywords, representative_articles, runtime",
            "Mục đích": "Phân loại bài mới bằng một trong 5 mô hình.",
        },
    ])
    st.dataframe(endpoints, width="stretch", hide_index=True)

    with st.expander("Quy trình xử lý chi tiết"):
        st.markdown(
            "1. Người dùng chọn mô hình và nhập tiêu đề hoặc nội dung.\n"
            "2. API kiểm tra dữ liệu và áp dụng cách tiền xử lý tương ứng.\n"
            "3. Mô hình đã lưu gán topic cho bài mới.\n"
            "4. Giao diện hiển thị tên chủ đề, từ khóa và bài đại diện."
        )

    st.subheader("Giới hạn triển khai")
    st.markdown(
        "- Lần gọi đầu có thể chậm do model được nạp vào bộ nhớ.\n"
        "- Điểm gắn topic (nếu có) không phải xác suất dự đoán đúng.\n"
        "- BERTopic và HDBSCAN có thể trả về bài chưa thuộc cụm nào; "
        "K-Means, NMF và LDA luôn gán một topic."
    )

