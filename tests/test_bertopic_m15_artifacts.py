"""Kiểm thử quá trình tạo artifact BERTopic cho M15."""

import pandas as pd

from experiments.bertopic.build_m15_artifacts import (
    build_display_mapping,
    build_representative_documents,
    build_topic_keywords,
    make_suggested_label,
)


def make_keyword_table() -> pd.DataFrame:
    """Tạo bảng keyword giả lập cho unit test."""
    return pd.DataFrame(
        {
            "topic": [-1, 0, 0, 0, 1, 1],
            "rank": [1, 1, 2, 3, 1, 2],
            "word": [
                "nhiễu",
                "ngân hàng",
                "lãi suất",
                "tín dụng",
                "bóng đá",
                "đội tuyển",
            ],
            "ctfidf_score": [
                0.1,
                0.9,
                0.8,
                0.7,
                0.95,
                0.85,
            ],
        }
    )


def make_document_topics() -> pd.DataFrame:
    """Tạo bảng document-topic giả lập."""
    return pd.DataFrame(
        {
            "article_id": [
                "A001",
                "A002",
                "A003",
                "A004",
            ],
            "split": [
                "train",
                "train",
                "val",
                "train",
            ],
            "topic": [0, 0, 1, -1],
            "topic_probability": [
                0.75,
                0.95,
                0.80,
                0.20,
            ],
            "is_outlier": [
                False,
                False,
                False,
                True,
            ],
            "title": [
                "Ngân hàng điều chỉnh lãi suất",
                "Tăng trưởng tín dụng",
                "Đội tuyển chuẩn bị thi đấu",
                "Tin chưa xác định",
            ],
            "category_gold": [
                "Kinh doanh",
                "Kinh doanh",
                "Thể thao",
                "Đời sống",
            ],
        }
    )


def test_topic_keywords_exclude_outlier() -> None:
    """Topic -1 không được đưa vào keyword thông thường."""
    result = build_topic_keywords(
        make_keyword_table()
    )

    assert "-1" not in result
    assert set(result) == {"0", "1"}
    assert result["0"][0]["word"] == "ngân hàng"
    assert result["0"][0]["rank"] == 1


def test_representatives_exclude_outlier() -> None:
    """Bài outlier không được chọn làm bài đại diện."""
    result = build_representative_documents(
        make_document_topics()
    )

    assert "-1" not in result
    assert set(result) == {"0", "1"}


def test_representatives_sorted_by_probability() -> None:
    """Bài có xác suất cao hơn phải đứng trước."""
    result = build_representative_documents(
        make_document_topics()
    )

    assert result["0"][0]["article_id"] == "A002"
    assert result["0"][1]["article_id"] == "A001"
    assert (
        result["0"][0]["topic_probability"]
        >= result["0"][1]["topic_probability"]
    )


def test_suggested_label_uses_top_three_words() -> None:
    """Nhãn tạm được tạo từ tối đa ba keyword đầu."""
    keywords = [
        {"word": "ngân_hàng"},
        {"word": "lãi_suất"},
        {"word": "tín_dụng"},
        {"word": "kinh_tế"},
    ]

    result = make_suggested_label(keywords)

    assert result == "ngân hàng – lãi suất – tín dụng"


def test_display_mapping_preserves_topic_ids() -> None:
    """Mapping phải giữ topic ID và thêm outlier -1."""
    keywords = build_topic_keywords(
        make_keyword_table()
    )

    result = build_display_mapping(keywords)

    assert set(result) == {"-1", "0", "1"}
    assert result["-1"]["status"] == "system"
    assert result["0"]["status"] == "provisional"
    assert result["0"]["display_label"]
    assert result["0"]["reviewed_by"] == []