import numpy as np
import pytest

from src.evaluation.bertopic_metrics import (
    calculate_outlier_rate,
    calculate_topic_diversity,
)


def test_calculate_outlier_rate() -> None:
    """Tỷ lệ outlier bằng số topic -1 chia tổng số bài."""
    topics = [-1, 0, 0, 1]

    outlier_rate = calculate_outlier_rate(topics)

    assert outlier_rate == pytest.approx(0.25)


def test_outlier_rate_rejects_empty_input() -> None:
    """Không tính outlier rate khi danh sách topic rỗng."""
    with pytest.raises(ValueError, match="rỗng"):
        calculate_outlier_rate([])


def test_calculate_topic_diversity() -> None:
    """Topic diversity đo tỷ lệ keyword không bị lặp."""
    topic_words = {
        -1: [("nhiễu", 0.5)],
        0: [("kinh tế", 0.8), ("ngân hàng", 0.6)],
        1: [("thể thao", 0.9), ("ngân hàng", 0.5)],
    }

    diversity = calculate_topic_diversity(
        topic_words,
        top_n=2,
    )

    assert diversity == pytest.approx(0.75)


def test_topic_diversity_ignores_outlier_topic() -> None:
    """Topic -1 không được tính vào topic diversity."""
    topic_words = {
        -1: [("nhiễu", 0.5)],
        0: [("kinh tế", 0.8), ("ngân hàng", 0.6)],
    }

    diversity = calculate_topic_diversity(
        topic_words,
        top_n=2,
    )

    assert diversity == pytest.approx(1.0)

def test_outlier_rate_accepts_numpy_array() -> None:
    """Metric phải nhận được output NumPy của BERTopic."""
    topics = np.array([-1, 0, 1, -1])

    outlier_rate = calculate_outlier_rate(topics)

    assert outlier_rate == pytest.approx(0.5)