"""Kiểm tra các hàm trực quan hóa BERTopic."""

from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

from src.visualization.bertopic_plots import (
    plot_bertopic_pipeline,
    plot_cluster_evaluation,
    plot_ctfidf_keywords,
    plot_embedding_map,
    plot_experiment_dashboard,
    plot_topic_diversity,
    plot_topic_sizes,
    plot_validation_outlier,
    validate_columns,
)

def make_summary() -> pd.DataFrame:
    """Tạo kết quả nhỏ để kiểm tra biểu đồ."""
    return pd.DataFrame(
        {
            "run_id": [
                "e0_multilingual_minilm",
                "e1_vietnamese_sbert",
                "e2_vietnamese_sbert_neighbors30",
                "e3_vietnamese_sbert_cluster30",
            ],
            "topic_count": [
                111,
                96,
                93,
                57,
            ],
            "validation_outlier_rate": [
                0.42,
                0.36,
                0.52,
                0.39,
            ],
            "topic_diversity": [
                0.70,
                0.68,
                0.70,
                0.75,
            ],
            "runtime_seconds": [
                183.0,
                911.0,
                905.0,
                998.0,
            ],
        }
    )


def make_topic_info() -> pd.DataFrame:
    """Tạo bảng topic nhỏ để kiểm tra biểu đồ."""
    return pd.DataFrame(
        {
            "Topic": [
                -1,
                0,
                1,
                2,
            ],
            "Count": [
                20,
                100,
                80,
                60,
            ],
            "Name": [
                "-1_nhiễu",
                "0_ngân_hàng_lãi_suất",
                "1_bóng_đá_đội_tuyển",
                "2_công_nghệ_dữ_liệu",
            ],
        }
    )

def make_topic_summary() -> pd.DataFrame:
    """Tạo bảng tổng hợp topic nhỏ để kiểm tra."""
    return pd.DataFrame(
        {
            "topic": [0, 1, 2],
            "topic_name": [
                "0_ngân_hàng_lãi_suất",
                "1_bóng_đá_đội_tuyển",
                "2_công_nghệ_dữ_liệu",
            ],
            "document_count": [100, 80, 60],
            "percentage": [0.42, 0.33, 0.25],
            "top_keywords": [
                "ngân hàng, lãi suất",
                "bóng đá, đội tuyển",
                "công nghệ, dữ liệu",
            ],
        }
    )


def make_topic_keywords() -> pd.DataFrame:
    """Tạo bảng từ khóa c-TF-IDF nhỏ."""
    rows = []

    topic_words = {
        0: ["ngân hàng", "lãi suất", "tín dụng"],
        1: ["bóng đá", "đội tuyển", "cầu thủ"],
        2: ["công nghệ", "dữ liệu", "mô hình"],
    }

    for topic, words in topic_words.items():
        for rank, word in enumerate(
            words,
            start=1,
        ):
            rows.append(
                {
                    "topic": topic,
                    "rank": rank,
                    "word": word,
                    "ctfidf_score": (
                        0.10 / rank
                    ),
                }
            )

    return pd.DataFrame(rows)


def make_embedding_projection() -> pd.DataFrame:
    """Tạo tọa độ embedding 2D giả lập."""
    return pd.DataFrame(
        {
            "article_id": [
                "A001",
                "A002",
                "A003",
                "A004",
                "A005",
                "A006",
            ],
            "split": [
                "train",
                "train",
                "train",
                "val",
                "val",
                "val",
            ],
            "topic": [
                0,
                1,
                -1,
                0,
                1,
                2,
            ],
            "x": [
                0.1,
                1.0,
                0.5,
                0.2,
                1.1,
                2.0,
            ],
            "y": [
                0.2,
                1.1,
                0.7,
                0.3,
                1.2,
                2.1,
            ],
            "is_outlier": [
                False,
                False,
                True,
                False,
                False,
                False,
            ],
        }
    )


def make_document_topics() -> pd.DataFrame:
    """Tạo kết quả gán topic cho train và validation."""
    return pd.DataFrame(
        {
            "split": [
                "train",
                "train",
                "train",
                "val",
                "val",
                "val",
            ],
            "topic": [
                0,
                1,
                -1,
                0,
                1,
                -1,
            ],
            "topic_probability": [
                0.92,
                0.84,
                0.0,
                0.80,
                0.76,
                0.0,
            ],
            "is_outlier": [
                False,
                False,
                True,
                False,
                False,
                True,
            ],
        }
    )

@pytest.mark.parametrize(
    ("plot_function", "filename"),
    [
        (
            plot_experiment_dashboard,
            "dashboard.png",
        ),
        (
            plot_validation_outlier,
            "outlier.png",
        ),
        (
            plot_topic_diversity,
            "diversity.png",
        ),
    ],
)
def test_experiment_plots_create_files(
    tmp_path: Path,
    plot_function: Callable,
    filename: str,
) -> None:
    """Kiểm tra biểu đồ thí nghiệm được tạo."""
    output_path = tmp_path / filename

    plot_function(
        make_summary(),
        output_path,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_topic_size_plot_creates_file(
    tmp_path: Path,
) -> None:
    """Kiểm tra biểu đồ kích thước topic được tạo."""
    output_path = (
        tmp_path / "topic_sizes.png"
    )

    plot_topic_sizes(
        make_topic_info(),
        output_path,
        top_n=2,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_validate_columns_rejects_missing_column() -> None:
    """Kiểm tra bảng thiếu cột bị từ chối."""
    data = pd.DataFrame(
        {
            "run_id": ["e0"],
        }
    )

    with pytest.raises(
        ValueError,
        match="Thiếu cột",
    ):
        validate_columns(
            data,
            {
                "run_id",
                "topic_count",
            },
        )

def test_embedding_map_creates_file(
    tmp_path: Path,
) -> None:
    """Kiểm tra bản đồ embedding được tạo."""
    output_path = tmp_path / "embedding.png"

    plot_embedding_map(
        make_embedding_projection(),
        make_topic_summary(),
        output_path,
        top_n=3,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_ctfidf_plot_creates_file(
    tmp_path: Path,
) -> None:
    """Kiểm tra biểu đồ từ khóa c-TF-IDF."""
    output_path = tmp_path / "ctfidf.png"

    plot_ctfidf_keywords(
        make_topic_keywords(),
        make_topic_summary(),
        output_path,
        top_n_topics=3,
        top_n_words=3,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_pipeline_plot_creates_file(
    tmp_path: Path,
) -> None:
    """Kiểm tra sơ đồ pipeline được tạo."""
    output_path = tmp_path / "pipeline.png"

    plot_bertopic_pipeline(
        output_path
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_cluster_evaluation_creates_file(
    tmp_path: Path,
) -> None:
    """Kiểm tra dashboard đánh giá cụm."""
    output_path = (
        tmp_path / "cluster_evaluation.png"
    )

    plot_cluster_evaluation(
        make_document_topics(),
        make_topic_summary(),
        output_path,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0