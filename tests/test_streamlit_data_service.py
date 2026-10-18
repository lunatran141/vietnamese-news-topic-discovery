"""Kiểm thử lớp dữ liệu dành cho Streamlit."""

from app.data_service import (
    get_documents_for_topic,
    get_keywords_for_topic,
    get_model_overview,
    get_topic_table,
    load_dashboard_bundle,
)


ARTIFACT_DIR = (
    "artifacts/common/bertopic/"
    "e1_vietnamese_sbert"
)


def test_load_dashboard_bundle() -> None:
    """Streamlit đọc được artifact BERTopic thật."""
    bundle = load_dashboard_bundle(ARTIFACT_DIR)

    assert bundle.model_summary.run_id == (
        "e1_vietnamese_sbert"
    )
    assert len(bundle.document_topics) == 7884
    assert len(bundle.topic_keywords) == 960


def test_get_model_overview() -> None:
    """Dashboard có đủ số liệu tổng quan."""
    bundle = load_dashboard_bundle(ARTIFACT_DIR)

    overview = get_model_overview(bundle)

    assert overview["algorithm"] == "BERTopic"
    assert overview["topic_count"] == 96
    assert overview["document_count"] == 7884
    assert (
        overview["assigned_document_count"]
        + overview["outlier_document_count"]
        == overview["document_count"]
    )


def test_get_topic_table_excludes_outlier() -> None:
    """Bảng topic mặc định không hiển thị topic -1."""
    bundle = load_dashboard_bundle(ARTIFACT_DIR)

    topics = get_topic_table(bundle)

    assert -1 not in topics["topic_id"].tolist()
    assert topics["document_count"].is_monotonic_decreasing


def test_get_topic_details() -> None:
    """Có thể lấy bài báo và từ khóa theo topic."""
    bundle = load_dashboard_bundle(ARTIFACT_DIR)

    topic_id = int(
        get_topic_table(bundle).iloc[0]["topic_id"]
    )

    documents = get_documents_for_topic(
        bundle,
        topic_id,
    )
    keywords = get_keywords_for_topic(
        bundle,
        topic_id,
    )

    assert not documents.empty
    assert not keywords.empty
    assert documents["topic_id"].eq(topic_id).all()
    assert keywords["topic_id"].eq(topic_id).all()