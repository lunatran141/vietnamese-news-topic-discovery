"""Schema dữ liệu chung cho các mô hình khám phá chủ đề."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


OUTLIER_TOPIC_ID = -1


class TopicKeyword(BaseModel):
    """Một từ khóa đại diện cho topic."""

    model_config = ConfigDict(extra="forbid")

    topic_id: int
    rank: int = Field(ge=1)
    keyword: str = Field(min_length=1)
    weight: float = Field(ge=0.0)


class TopicSummary(BaseModel):
    """Thông tin tổng hợp của một topic."""

    model_config = ConfigDict(extra="forbid")

    topic_id: int
    topic_name: str = Field(min_length=1)
    document_count: int = Field(ge=0)
    percentage: float = Field(ge=0.0, le=1.0)
    top_keywords: list[str] = Field(default_factory=list)

    @property
    def is_outlier(self) -> bool:
        """Cho biết đây có phải topic nhiễu hay không."""
        return self.topic_id == OUTLIER_TOPIC_ID


class DocumentTopic(BaseModel):
    """Kết quả gán một bài báo vào topic."""

    model_config = ConfigDict(extra="allow")

    article_id: str = Field(min_length=1)
    split: str = Field(min_length=1)
    topic_id: int
    topic_probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    is_outlier: bool = False
    title: str | None = None
    source: str | None = None
    published_at: str | None = None

    @field_validator("is_outlier")
    @classmethod
    def validate_outlier_flag(
        cls,
        value: bool,
    ) -> bool:
        """Giữ trường outlier ở dạng boolean."""
        return bool(value)


class ModelSummary(BaseModel):
    """Metadata và metric tổng quát của một lần chạy mô hình."""

    model_config = ConfigDict(extra="allow")

    model_name: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    algorithm: str = Field(min_length=1)
    representation: str = Field(min_length=1)
    topic_count: int = Field(ge=0)
    outlier_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    runtime_seconds: float | None = Field(
        default=None,
        ge=0.0,
    )
    metrics: dict[str, float | int | None] = Field(
        default_factory=dict
    )
    config: dict[str, Any] = Field(default_factory=dict)


class ArtifactManifest(BaseModel):
    """Danh mục file đầu ra của một mô hình."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    model_name: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    model_summary_path: str
    topic_summary_path: str
    document_topics_path: str
    topic_keywords_path: str
    config_path: str