"""Dashboard khám phá kết quả của năm phương pháp topic discovery (router)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Đảm bảo thư mục gốc dự án luôn có trong sys.path khi chạy bằng streamlit run
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import requests
import streamlit as st

from app.data_service import (
    ArtifactRun, discover_artifact_runs, get_available_splits, load_dashboard_bundle,
)
from app.ui.context import AnalysisContext
from app.ui.theme import inject_css, register_altair_theme
from app.views import about, articles, evaluation, explorer, organizer, overview

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Vietnamese News Topic Discovery", layout="wide")
inject_css()
register_altair_theme()


def analysis_sidebar(runs: list[ArtifactRun]) -> AnalysisContext:
    """Khối chọn ngữ cảnh phân tích; áp dụng cho các trang theo-một-run."""
    with st.sidebar:
        st.markdown('<div class="ui-eyebrow">Ngữ cảnh phân tích</div>', unsafe_allow_html=True)
        preferred = next((i for i, run in enumerate(runs) if run.method == "bertopic"), 0)
        run = st.selectbox("Mô hình và run", runs, index=preferred, format_func=lambda x: x.label)
        try:
            bundle = load_dashboard_bundle(str(run.path))
        except (FileNotFoundError, ValueError) as error:
            st.error(f"Artifact của run này không hợp lệ: {error}")
            st.stop()
        splits = get_available_splits(bundle)
        split = st.selectbox(
            "Tập dữ liệu", ["all", *splits],
            format_func=lambda x: "Tất cả có sẵn" if x == "all" else x,
        )
        st.caption(
            "Các mô hình có thể xuất split khác nhau. Bộ chọn này áp dụng cho các trang phân tích; "
            "trang so sánh mô hình tự chọn split riêng."
        )
        with st.expander("Kết nối hệ thống"):
            if st.button("Kiểm tra API", width="stretch"):
                try:
                    requests.get(f"{API_BASE_URL}/health", timeout=5).raise_for_status()
                    st.success("API đang hoạt động")
                except requests.RequestException:
                    st.warning("Chưa kết nối được API; các trang artifact vẫn hoạt động.")
    return AnalysisContext(runs=runs, run=run, bundle=bundle, split=split)


def main() -> None:
    runs = discover_artifact_runs()
    if not runs:
        st.error("Không tìm thấy manifest trong artifacts/common.")
        st.stop()

    def page_overview() -> None:
        overview.render(ctx)

    def page_explorer() -> None:
        explorer.render(ctx)

    def page_articles() -> None:
        articles.render(ctx)

    def page_organizer() -> None:
        organizer.render(ctx)

    def page_evaluation() -> None:
        evaluation.render(ctx)

    def page_about() -> None:
        about.render(ctx)

    navigation = st.navigation({
        "Phân tích": [
            st.Page(page_overview, title="Tổng quan", url_path="tong-quan", default=True),
            st.Page(page_explorer, title="Khám phá chủ đề", url_path="kham-pha-chu-de"),
            st.Page(page_articles, title="Kho bài viết", url_path="kho-bai-viet"),
            st.Page(page_organizer, title="Tổ chức bài báo mới", url_path="to-chuc-bai-bao"),
        ],
        "Đánh giá": [st.Page(page_evaluation, title="Đánh giá và so sánh", url_path="danh-gia")],
        "Dự án": [st.Page(page_about, title="Dự án", url_path="du-an")],
    })
    ctx = analysis_sidebar(runs)
    navigation.run()


if __name__ == "__main__":
    main()
