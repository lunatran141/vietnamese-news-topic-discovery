"""Ngữ cảnh phân tích dùng chung giữa router và các trang."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.data_service import ArtifactRun


@dataclass(frozen=True)
class AnalysisContext:
    """Run, bundle và split đang được chọn ở sidebar."""

    runs: list[ArtifactRun]
    run: ArtifactRun
    bundle: Any
    split: str

