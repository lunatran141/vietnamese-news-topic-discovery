"""Kiểu dữ liệu trao đổi của Internal Inference API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Bài báo mới cần gán topic."""

    title: str = Field(default="", max_length=1_000)
    content: str = Field(default="", max_length=100_000)
    model: str = Field(default="bertopic", description="bertopic, hdbscan, kmeans, nmf hoặc lda")


class RepresentativeArticle(BaseModel):
    """Bài báo đại diện giúp giải thích kết quả dự đoán."""

    article_id: str
    title: str
    split: str | None = None


class PredictResponse(BaseModel):
    """Kết quả dự đoán thống nhất cho giao diện Streamlit."""

    model: str
    run_id: str
    topic_id: int
    topic_name: str
    is_outlier: bool
    confidence: float | None = None
    keywords: list[str] = Field(default_factory=list)
    representative_articles: list[RepresentativeArticle] = Field(
        default_factory=list
    )
    inference_seconds: float
    preprocessing_variant: str
    inference_mode: str

