import numpy as np
import pandas as pd
import pytest

from src.evaluation.bertopic_artifacts import (
    build_document_topics,
    build_embedding_projection,
    build_topic_keywords,
    build_topic_summary,
    prepare_topic_diagnostics,
    prepare_topic_probabilities,
)

class FakeTopicModel:
    """Mô phỏng từ khóa topic để không cần tải model thật."""

    def get_topics(
        self,
    ) -> dict[int, list[tuple[str, float]]]:
        return {
            -1: [("nhiễu", 0.1)],
            0: [
                ("ngân_hàng", 0.9),
                ("lãi_suất", 0.8),
                ("tín_dụng", 0.7),
            ],
            1: [
                ("bóng_đá", 0.95),
                ("đội_tuyển", 0.85),
                ("trận_đấu", 0.75),
            ],
        }


def make_dataset() -> pd.DataFrame:
    """Tạo dataset nhỏ dùng để kiểm tra bảng kết quả."""
    return pd.DataFrame(
        {
            "article_id": ["A001", "A002", "A003"],
            "title": [
                "Ngân hàng giảm lãi suất",
                "Đội tuyển chuẩn bị thi đấu",
                "Tin tổng hợp",
            ],
            "category_gold": [
                "Kinh doanh",
                "Thể thao",
                "Đời sống",
            ],
        }
    )


def test_prepare_topic_probabilities_from_matrix() -> None:
    """Kiểm tra ma trận xác suất được rút thành một giá trị mỗi bài."""
    probabilities = np.array(
        [
            [0.8, 0.2],
            [0.3, 0.7],
        ]
    )

    result = prepare_topic_probabilities(
        probabilities,
        expected_length=2,
    )

    assert result.tolist() == [0.8, 0.7]


def test_prepare_topic_diagnostics_from_membership_matrix() -> None:
    """Ma trận membership phải xuất gap, entropy và cờ đa chủ đề."""
    result = prepare_topic_diagnostics(
        np.array([[0.51, 0.49], [0.90, 0.10]]),
        expected_length=2,
    )

    assert result["second_topic_probability"].tolist() == [0.49, 0.10]
    assert result["top_two_share_gap"].tolist() == pytest.approx([0.02, 0.80])
    assert result["is_multi_topic"].tolist() == [True, False]


def test_build_document_topics() -> None:
    """Kiểm tra mỗi bài được gắn đúng topic và trạng thái outlier."""
    result = build_document_topics(
        make_dataset(),
        topics=[0, 1, -1],
        probabilities=[0.9, 0.8, 0.2],
        split="train",
    )

    assert len(result) == 3
    assert result["article_id"].tolist() == [
        "A001",
        "A002",
        "A003",
    ]
    assert result["topic"].tolist() == [0, 1, -1]
    assert result["is_outlier"].tolist() == [
        False,
        False,
        True,
    ]
    assert result["split"].eq("train").all()


def test_document_topics_rejects_wrong_length() -> None:
    """Kiểm tra số topic phải khớp số bài."""
    with pytest.raises(
        ValueError,
        match="Số topic",
    ):
        build_document_topics(
            make_dataset(),
            topics=[0, 1],
            probabilities=None,
            split="train",
        )


def test_build_topic_keywords_ignores_outlier() -> None:
    """Kiểm tra bảng từ khóa không chứa topic nhiễu."""
    result = build_topic_keywords(
        FakeTopicModel()
    )

    assert set(result["topic"]) == {0, 1}
    assert result.iloc[0]["word"] == "ngân_hàng"
    assert result.iloc[0]["rank"] == 1
    assert result.iloc[0]["ctfidf_score"] == 0.9


def test_build_topic_summary() -> None:
    """Kiểm tra bảng tổng hợp topic có đủ thông tin chính."""
    document_topics = build_document_topics(
        make_dataset(),
        topics=[0, 1, -1],
        probabilities=[0.9, 0.8, 0.2],
        split="train",
    )
    topic_keywords = build_topic_keywords(
        FakeTopicModel()
    )

    result = build_topic_summary(
        document_topics,
        topic_keywords,
    )

    assert len(result) == 2
    assert result["document_count"].sum() == 2
    assert set(result["dominant_category"]) == {
        "Kinh doanh",
        "Thể thao",
    }
    assert result["category_purity"].eq(1.0).all()

def test_build_embedding_projection() -> None:
    """Tọa độ 2D phải giữ đúng bài báo, topic và split."""
    article_ids = pd.Series(
        ["A001", "A002", "A003"]
    )
    topics = [0, -1, 1]
    embeddings_2d = np.array(
        [
            [0.10, 0.20],
            [0.30, 0.40],
            [0.50, 0.60],
        ]
    )

    result = build_embedding_projection(
        article_ids=article_ids,
        split="train",
        topics=topics,
        embeddings_2d=embeddings_2d,
    )

    assert list(result.columns) == [
        "article_id",
        "split",
        "topic",
        "x",
        "y",
        "is_outlier",
    ]
    assert len(result) == 3
    assert result["article_id"].tolist() == [
        "A001",
        "A002",
        "A003",
    ]
    assert result["topic"].tolist() == [0, -1, 1]
    assert result["is_outlier"].tolist() == [
        False,
        True,
        False,
    ]
    assert result["split"].eq("train").all()

def test_embedding_projection_rejects_non_2d_data() -> None:
    """Tọa độ trực quan hóa phải có đúng hai chiều."""
    article_ids = pd.Series(["A001", "A002"])
    embeddings_3d = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
    )

    with pytest.raises(
        ValueError,
        match="đúng hai cột",
    ):
        build_embedding_projection(
            article_ids=article_ids,
            split="train",
            topics=[0, 1],
            embeddings_2d=embeddings_3d,
        )
