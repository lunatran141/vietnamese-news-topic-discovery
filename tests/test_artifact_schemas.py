"""Kiểm thử schema artifact chung."""

import pytest
from pydantic import ValidationError

from src.artifacts.schemas import (
    ArtifactManifest,
    DocumentTopic,
    ModelSummary,
    TopicKeyword,
    TopicSummary,
)


def test_document_topic_accepts_valid_record() -> None:
    """Chấp nhận kết quả gán topic hợp lệ."""
    record = DocumentTopic(
        article_id="TTR_001",
        split="train",
        topic_id=2,
        topic_probability=0.91,
        is_outlier=False,
        title="Bài báo thử nghiệm",
    )

    assert record.article_id == "TTR_001"
    assert record.topic_id == 2
    assert record.topic_probability == pytest.approx(0.91)


def test_document_topic_rejects_invalid_probability() -> None:
    """Xác suất gán topic phải nằm trong đoạn 0 đến 1."""
    with pytest.raises(ValidationError):
        DocumentTopic(
            article_id="TTR_001",
            split="train",
            topic_id=2,
            topic_probability=1.5,
        )


def test_topic_keyword_requires_positive_rank() -> None:
    """Thứ hạng từ khóa phải bắt đầu từ 1."""
    with pytest.raises(ValidationError):
        TopicKeyword(
            topic_id=0,
            rank=0,
            keyword="kinh tế",
            weight=0.25,
        )


def test_outlier_topic_is_detected() -> None:
    """Topic -1 được xem là outlier."""
    topic = TopicSummary(
        topic_id=-1,
        topic_name="Outlier",
        document_count=12,
        percentage=0.12,
        top_keywords=[],
    )

    assert topic.is_outlier is True


def test_model_summary_accepts_model_metrics() -> None:
    """Schema hỗ trợ metric khác nhau giữa các mô hình."""
    summary = ModelSummary(
        model_name="BERTopic E1",
        run_id="e1_vietnamese_sbert",
        algorithm="BERTopic",
        representation="Vietnamese SBERT",
        topic_count=15,
        outlier_rate=0.35,
        runtime_seconds=911.0,
        metrics={
            "topic_diversity": 0.681,
            "coherence": None,
        },
        config={
            "umap_n_neighbors": 15,
            "hdbscan_min_cluster_size": 15,
        },
    )

    assert summary.topic_count == 15
    assert summary.metrics["topic_diversity"] == pytest.approx(
        0.681
    )


def test_artifact_manifest_declares_required_files() -> None:
    """Manifest phải chỉ rõ các file dùng bởi API và Streamlit."""
    manifest = ArtifactManifest(
        model_name="BERTopic E1",
        run_id="e1_vietnamese_sbert",
        model_summary_path="model_summary.json",
        topic_summary_path="topic_summary.csv",
        document_topics_path="document_topics.csv",
        topic_keywords_path="topic_keywords.csv",
        config_path="config_used.yaml",
    )

    assert manifest.schema_version == "1.0"
    assert manifest.document_topics_path == "document_topics.csv"