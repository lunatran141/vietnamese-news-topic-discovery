import pandas as pd
import pytest

from src.data.dataset_loader import (
    load_split_dataset,
    validate_dataset,
)
from src.features.article_text import build_article_text


def make_valid_dataset() -> pd.DataFrame:
    """Tạo dataset nhỏ để kiểm tra các hàm đầu vào."""
    return pd.DataFrame(
        {
            "article_id": ["A001", "A002"],
            "title": ["Tin kinh tế", "Tin thể thao"],
            "content": [
                "Ngân hàng công bố mức lãi suất mới.",
                "Đội tuyển chuẩn bị cho trận đấu.",
            ],
            "processed_title": [
                "tin kinh tế",
                "tin thể thao",
            ],
            "processed_content": [
                "ngân hàng công bố mức lãi suất mới",
                "đội tuyển chuẩn bị cho trận đấu",
            ],
            "text": [
                "tin kinh tế ngân hàng công bố mức lãi suất mới",
                "tin thể thao đội tuyển chuẩn bị cho trận đấu",
            ],
            "published_at": ["2026-10-01", "2026-10-02"],
            "source": ["source_a", "source_b"],
            "url": [
                "https://example.com/1",
                "https://example.com/2",
            ],
        }
    )


def test_valid_dataset_passes() -> None:
    dataset = make_valid_dataset()
    validate_dataset(dataset)


def test_missing_article_id_raises_error() -> None:
    dataset = make_valid_dataset().drop(columns=["article_id"])

    with pytest.raises(ValueError, match="article_id"):
        validate_dataset(dataset)


def test_duplicate_article_id_raises_error() -> None:
    dataset = make_valid_dataset()
    dataset.loc[1, "article_id"] = "A001"

    with pytest.raises(ValueError, match="bị trùng"):
        validate_dataset(dataset)


def test_build_text_preserves_row_order() -> None:
    dataset = make_valid_dataset()

    text = build_article_text(
        dataset,
        text_variant="processed_title_content",
    )

    assert len(text) == len(dataset)
    assert text.index.equals(dataset.index)
    assert "ngân hàng" in text.iloc[0]


def test_load_train_split_from_dataset_v1() -> None:
    """Kiểm tra dữ liệu train được ghép đúng theo split chính thức của D05."""
    dataset = load_split_dataset(
        "data/processed/p1_normalized.csv",
        "data/splits/split_manifest.csv",
        "train",
    )

    assert len(dataset) == 6493
    assert dataset["article_id"].is_unique
    assert dataset["split"].eq("train").all()