"""Kiểm thử Internal Inference API mà không tải model nặng."""

from __future__ import annotations

import pytest
import numpy as np
from fastapi import HTTPException
from pydantic import ValidationError

from src.api.inference_service import BERTopicInferenceService, MultiModelInferenceService, preprocess_article
from src.api.main import health_check, list_topics, predict_article
from src.api.schemas import (
    PredictRequest,
    PredictResponse,
    RepresentativeArticle,
)


class FakeInferenceService:
    def predict(self, title: str, content: str, model: str = "bertopic") -> PredictResponse:
        assert model == "bertopic"
        assert "giá vàng" in preprocess_article(title, content)
        return PredictResponse(
            model="BERTopic",
            run_id="e1_vietnamese_sbert",
            topic_id=7,
            topic_name="Thị trường tài chính",
            is_outlier=False,
            confidence=0.82,
            keywords=["giá vàng", "thị trường"],
            representative_articles=[
                RepresentativeArticle(
                    article_id="TTR_demo",
                    title="Giá vàng tăng",
                    split="train",
                )
            ],
            inference_seconds=0.01,
            preprocessing_variant="p1_normalized",
            inference_mode="saved_model_transform",
        )


class FakeTopicModel:
    """Chỉ có transform; test sẽ lỗi nếu code cố fit lại model."""

    def transform(self, documents: list[str]):
        assert documents and "giá vàng" in documents[0]
        return np.asarray([7]), np.asarray([[0.2, 0.78, 0.02]])


def test_health() -> None:
    assert health_check() == {"status": "ok"}


def test_predict_returns_standardized_response() -> None:
    result = predict_article(
        PredictRequest(
            title="Giá vàng hôm nay",
            content="Giá vàng trong nước tiếp tục tăng.",
        ),
        FakeInferenceService(),
    )
    assert result.run_id == "e1_vietnamese_sbert"
    assert result.topic_id == 7
    assert result.confidence == 0.82
    assert result.keywords == ["giá vàng", "thị trường"]


def test_predict_rejects_empty_article() -> None:
    with pytest.raises(HTTPException) as error:
        predict_article(
            PredictRequest(title="   ", content=""),
            FakeInferenceService(),
        )
    assert error.value.status_code == 422
    assert "không được đồng thời để trống" in error.value.detail


def test_request_length_is_validated() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(title="x" * 1_001, content="nội dung")


def test_service_uses_saved_transform_and_explanation_artifacts() -> None:
    result = BERTopicInferenceService(model=FakeTopicModel()).predict(
        "Giá vàng hôm nay",
        "Giá vàng trong nước tiếp tục tăng.",
    )
    assert result.topic_id == 7
    assert result.confidence == pytest.approx(0.78)
    assert result.topic_name
    assert result.keywords
    assert result.preprocessing_variant == "p1_normalized"


def test_invalid_model_is_rejected() -> None:
    with pytest.raises(HTTPException) as error:
        predict_article(PredictRequest(title="Tin mới", model="unknown"), MultiModelInferenceService())
    assert error.value.status_code == 422


def test_topics_reads_common_artifact() -> None:
    result = list_topics("kmeans", MultiModelInferenceService())
    assert result["run_id"] == "kmeans_baseline"
    assert result["topics"]
    assert result["topics"][0]["keywords"]

